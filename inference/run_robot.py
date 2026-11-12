"""Closed loop: SmolVLA drives the Panda in MuJoCo (Phase 6, Step 22).

Each 20 Hz step: render both cameras + read the 8-D state -> policy -> 8-D action -> clip to joint
limits -> servos + physics for one control period -> check success. The VLA decides and the controller
only executes; there is no scripted fallback.

Episodes use held-out layout seeds (>= eval_seed_start in configs/simulation.yaml) by default.

    # plumbing test with the untrained base model (actions are meaningless, but it should run)
    uv run python -m inference.run_robot --checkpoint lerobot/smolvla_base --episodes 1 --max-seconds 5

    # fine-tuned model, 5 episodes with a video
    uv run python -m inference.run_robot --episodes 5 --video outputs/smolvla/run.mp4

    # watch live (macOS needs mjpython for the viewer)
    uv run mjpython -m inference.run_robot --view --episodes 3
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data.tasks import make_instruction, target_from_instruction
from policies.smolvla_policy import DEFAULT_CHECKPOINT, SmolVLAController
from simulation.camera import CameraRenderer
from simulation.controller import RobotController
from simulation.environment import TabletopEnv

VIDEO_SIZE = 256   # each camera tile in the saved video (front | wrist side by side)


@dataclass
class EpisodeResult:
    seed: int
    target: str | None
    instruction: str
    success: bool
    wrong_object: bool
    n_steps: int
    sim_seconds: float
    chunk_latency_ms: list[float] = field(default_factory=list)   # one per network call
    step_latency_ms: list[float] = field(default_factory=list)    # every policy call, incl. queued ones


def run_episode(env: TabletopEnv, policy: SmolVLAController, cameras: CameraRenderer, seed: int,
                target: str | None, instruction: str, max_steps: int, on_frame=None,
                early_stop: bool = True) -> EpisodeResult:
    """target is only used for scoring; None (instruction names no cube) = just run until the timeout.

    early_stop=False keeps running to the timeout (nicer videos); the episode is still scored by
    whichever happened first, success or wrong object."""
    env.reset(seed)
    policy.reset()
    ctrl = RobotController(env.model, env.data)   # for state readout, joint limits and apply_action
    res = EpisodeResult(seed, target, instruction, False, False, 0, 0.0)

    for _ in range(max_steps):
        images = cameras.render_all(env.data)
        obs = policy.make_observation(ctrl.state(), images)

        replan = policy.will_replan
        t0 = time.perf_counter()
        action = policy.act(obs, instruction)
        ms = (time.perf_counter() - t0) * 1000
        res.step_latency_ms.append(ms)
        if replan:
            res.chunk_latency_ms.append(ms)

        action[:7] = np.clip(action[:7], ctrl.q_low, ctrl.q_high)
        ctrl.apply_action(action)
        res.n_steps += 1
        if on_frame:
            on_frame(images, action)

        if target is None or res.success or res.wrong_object:
            continue
        if env.wrong_object(target):
            res.wrong_object = True
        elif env.check_success(target):
            res.success = True
        if early_stop and (res.success or res.wrong_object):
            break

    res.sim_seconds = res.n_steps * ctrl.control_dt
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SmolVLA checkpoint in the MuJoCo tabletop.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Hub repo id or local folder (e.g. outputs/train/.../checkpoints/005000/pretrained_model)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="first layout seed (default: eval_seed_start)")
    parser.add_argument("--instruction", default=None,
                        help='what to tell the robot, e.g. "Put the blue cube in the box". The cube to score is '
                             "read from its colour word. Default: alternate red/blue with the dataset templates")
    parser.add_argument("--split", choices=["train", "eval"], default="train",
                        help="instruction templates: train wording or held-out paraphrases")
    parser.add_argument("--n-action-steps", type=int, default=25, help="actions executed per predicted chunk (1-50)")
    parser.add_argument("--max-seconds", type=float, default=None, help="episode timeout (default: config timeout_s)")
    parser.add_argument("--device", default=None, help="mps / cpu / cuda (default: best available)")
    parser.add_argument("--video", default=None, help="save all episodes to this mp4 (front | wrist)")
    parser.add_argument("--view", action="store_true", help="watch live (macOS: run with `uv run mjpython`)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="keep running to the timeout after success / wrong object (scored by what happened first)")
    args = parser.parse_args()

    env = TabletopEnv()
    cfg = env.config
    seed0 = cfg["eval_seed_start"] if args.seed is None else args.seed
    if seed0 < cfg["eval_seed_start"]:
        print(f"note: seed {seed0} is a training layout (< {cfg['eval_seed_start']}), not a held-out one")
    max_steps = round((args.max_seconds or cfg["success"]["timeout_s"]) * cfg["fps"])
    if args.instruction:
        fixed_target = target_from_instruction(args.instruction)
        print(f"instruction: \"{args.instruction}\" -> scoring "
              + (fixed_target.replace("_", " ") if fixed_target else "nothing (no single colour named; runs to timeout)"))

    print(f"loading {args.checkpoint} ...")
    policy = SmolVLAController(args.checkpoint, device=args.device, n_action_steps=args.n_action_steps)
    print(f"device {policy.device}, executing {args.n_action_steps}/{policy.config.chunk_size} actions per chunk")
    cameras = CameraRenderer(env.model, cfg["camera"]["height"], cfg["camera"]["width"],
                             cameras=tuple(cfg["camera"]["names"]))

    on_frame, cleanup = _make_frame_sink(env, args, cfg["fps"])
    results = []
    try:
        for ep in range(args.episodes):
            seed = seed0 + ep
            if args.instruction:
                target, instruction = fixed_target, args.instruction
            else:
                target = cfg["targets"][ep % len(cfg["targets"])]
                instruction = make_instruction(target, np.random.default_rng(20_000 + seed), args.split)
            res = run_episode(env, policy, cameras, seed, target, instruction, max_steps, on_frame,
                              early_stop=not args.no_early_stop)
            results.append(res)
            status = ("ran" if target is None else "ok  " if res.success
                      else "WRONG OBJECT" if res.wrong_object else "fail")
            print(f"seed {seed:6d}  {status:<12} {res.sim_seconds:5.1f} s  {res.n_steps:4d} steps  "
                  f"chunk {np.mean(res.chunk_latency_ms):6.0f} ms  \"{instruction}\"")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cleanup()
        cameras.close()

    if results:
        _summary(results, cfg["fps"])


def _make_frame_sink(env: TabletopEnv, args, fps: int):
    """Optional video writer and/or live viewer, called once per control step."""
    writer = viewer = None
    if args.video:
        import cv2
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * VIDEO_SIZE, VIDEO_SIZE))
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    def on_frame(images, _action):
        if writer is not None:
            import cv2
            tiles = [cv2.resize(images[c], (VIDEO_SIZE, VIDEO_SIZE)) for c in ("front_camera", "wrist_camera")]
            writer.write(np.hstack(tiles)[:, :, ::-1])
        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()

    def cleanup():
        if writer is not None:
            writer.release()
            print(f"video -> {args.video}")
        if viewer is not None:
            viewer.close()

    return (on_frame if writer or viewer else None), cleanup


def _summary(results: list[EpisodeResult], fps: int) -> None:
    scored = [r for r in results if r.target is not None]
    ok = [r for r in scored if r.success]
    chunk = np.concatenate([r.chunk_latency_ms[1:] or r.chunk_latency_ms for r in results])  # skip warm-up call
    step = np.concatenate([r.step_latency_ms for r in results])
    if scored:
        n = len(scored)
        print(f"\nsuccess {len(ok)}/{n} = {100 * len(ok) / n:.0f}%   wrong-object {sum(r.wrong_object for r in scored)}"
              + (f"   mean time to success {np.mean([r.sim_seconds for r in ok]):.1f} s" if ok else ""))
    else:
        print("\nno success score: the instruction names no single cube colour")
    print(f"latency: {np.mean(chunk):.0f} ms per chunk (median {np.median(chunk):.0f}), "
          f"{np.mean(step):.0f} ms per step on average -> "
          f"{1000 / np.mean(step):.1f} Hz achievable vs {fps} Hz control")


if __name__ == "__main__":
    main()
