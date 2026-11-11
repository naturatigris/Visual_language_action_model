"""Record scripted-expert demonstrations into a local LeRobotDataset (Phase 4, Step 12).

Every control step (20 Hz) becomes one frame:
    observation.state         8 floats: 7 arm joint angles (rad) + gripper opening (m)
    observation.images.front  256x256 RGB from front_camera   (stored as MP4 video)
    observation.images.wrist  256x256 RGB from wrist_camera   (stored as MP4 video)
    action                    8 floats: 7 joint targets (rad) + gripper command (0 closed .. 255 open)
    task                      the instruction, e.g. "Put the red cube in the container"

The observation is captured *before* the action is applied, so each frame says
"in this situation, the expert did this". Failed episodes are discarded.

    uv run python -m data.collect --episodes 100
    uv run python -m data.collect --episodes 5 --name test --overwrite
"""

import argparse
import shutil
import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from data.scripted_expert import ScriptedExpert
from data.tasks import make_instruction
from simulation.camera import CameraRenderer
from simulation.controller import RobotController
from simulation.environment import PROJECT_DIR, TabletopEnv, load_config

DATASET_DIR = PROJECT_DIR / "data" / "lerobot"
REPO_OWNER = "sandhya056"            # HF username; the dataset id is <owner>/<name> (upload: data/push_dataset.py)
CAMERA_KEYS = {"front_camera": "observation.images.front", "wrist_camera": "observation.images.wrist"}
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def make_features(config: dict) -> dict:
    h, w = config["camera"]["height"], config["camera"]["width"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + ["gripper_opening"]},
        "action": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + ["gripper"]},
    }
    for cam in config["camera"]["names"]:
        features[CAMERA_KEYS[cam]] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return features


class EpisodeRecorder:
    """Pairs each action with the observation taken just before it, and adds the frame to the dataset."""

    def __init__(self, dataset: LeRobotDataset, env: TabletopEnv, cameras: CameraRenderer, task: str):
        self.dataset, self.env, self.cameras, self.task = dataset, env, cameras, task
        self.state = RobotController(env.model, env.data)   # used only for its state readout
        self.n_frames = 0
        self.obs = self._observe()

    def _observe(self) -> dict:
        obs = {"observation.state": self.state.state().astype(np.float32)}
        for cam, image in self.cameras.render_all(self.env.data).items():
            obs[CAMERA_KEYS[cam]] = image
        return obs

    def __call__(self, action: np.ndarray) -> None:
        # Called by the expert *after* `action` was applied: self.obs is still the pre-action view.
        self.dataset.add_frame({**self.obs, "action": action.astype(np.float32), "task": self.task})
        self.n_frames += 1
        self.obs = self._observe()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record scripted-expert demos as a LeRobotDataset.")
    parser.add_argument("--episodes", type=int, default=100, help="number of *successful* episodes to keep")
    parser.add_argument("--name", default="panda_pick_place", help="dataset name (folder under data/lerobot/)")
    parser.add_argument("--seed", type=int, default=0, help="first layout seed")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing dataset with this name")
    args = parser.parse_args()

    root = DATASET_DIR / args.name
    eval_seed_start = load_config()["eval_seed_start"]
    if args.seed >= eval_seed_start:
        raise SystemExit(f"--seed {args.seed} is in the held-out eval range (>= {eval_seed_start}).")
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} already exists. Use --overwrite to replace it, or pick another --name.")
        shutil.rmtree(root)

    env = TabletopEnv()
    cfg = env.config
    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_OWNER}/{args.name}", fps=cfg["fps"], features=make_features(cfg), root=root,
        robot_type="franka_panda", use_videos=True, image_writer_threads=4,
    )
    cameras = CameraRenderer(env.model, cfg["camera"]["height"], cfg["camera"]["width"],
                             cameras=tuple(cfg["camera"]["names"]))
    targets = cfg["targets"]
    eval_seed_start = cfg["eval_seed_start"]

    kept, tried, seed, t0 = 0, 0, args.seed, time.time()
    try:
        while kept < args.episodes:
            if seed >= eval_seed_start:
                raise SystemExit(f"seed {seed} is in the held-out eval range (>= {eval_seed_start}); "
                                 "pick a lower --seed or fewer --episodes.")
            env.reset(seed)
            rng = np.random.default_rng(10_000 + seed)          # same per-seed variations as the evaluator
            target = targets[tried % len(targets)]
            task = make_instruction(target, np.random.default_rng(20_000 + seed))
            recorder = EpisodeRecorder(dataset, env, cameras, task)
            result = ScriptedExpert(env, rng).run(target, on_step=recorder)
            tried += 1
            if result.success:
                dataset.save_episode()
                kept += 1
                print(f"[{kept:4d}/{args.episodes}] seed {seed:5d}  {recorder.n_frames:4d} frames  \"{task}\"")
            else:
                dataset.clear_episode_buffer()
                print(f"           seed {seed:5d}  discarded (failed at {result.failed_stage or 'success check'})")
            seed += 1
    finally:
        dataset.finalize()
        cameras.close()

    print(f"\nSaved {kept} episodes / {dataset.num_frames} frames to {root}  "
          f"({tried - kept} discarded, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
