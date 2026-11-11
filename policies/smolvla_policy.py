"""SmolVLA policy wrapper for the MuJoCo tabletop (Phase 6, Step 21).

Turns one simulator observation into one 8-D action, exactly the way the policy saw data in training:

    observation.state         8 floats: 7 arm joints (rad) + gripper opening (m)       from RobotController.state()
    observation.images.front  256x256 RGB uint8 (HWC)                                   front_camera
    observation.images.wrist  256x256 RGB uint8 (HWC)                                   wrist_camera
    task                      the instruction string

    -> action  [q1..q7 targets (rad), gripper (0 closed .. 255 open)]

The policy is loaded the same way `lerobot-train` built it: the pretrained SmolVLA config, the camera
rename map (front -> camera1, wrist -> camera2), and normalization stats from our dataset. That makes
the same code work for both the fine-tuned checkpoint and the untrained `lerobot/smolvla_base`
(useful to test the plumbing before training finishes; its actions are meaningless).

SmolVLA predicts a chunk of 50 actions (2.5 s at 20 Hz). `n_action_steps` sets how many of them are
executed before it looks at the scene again: fewer = reacts sooner, but more inference calls.
"""

from pathlib import Path

import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION

from data.collect import CAMERA_KEYS, DATASET_DIR, REPO_OWNER

DEFAULT_CHECKPOINT = f"{REPO_OWNER}/smolvla_panda_pick_place"
BASE_MODEL = "lerobot/smolvla_base"
DATASET_NAME = "panda_pick_place"
# Must match --rename_map used in lerobot-train.
RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
GRIPPER_THRESHOLD = 127.5   # the model outputs a continuous value; the expert only ever sent 0 or 255


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SmolVLAController:
    """Loads a SmolVLA checkpoint and maps observations -> actions, one control step at a time."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str | None = None,
                 n_action_steps: int = 25, dataset_name: str = DATASET_NAME):
        self.device = torch.device(device or default_device())

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.device = self.device.type
        if not 1 <= n_action_steps <= cfg.chunk_size:
            raise ValueError(f"n_action_steps must be in [1, {cfg.chunk_size}], got {n_action_steps}")
        cfg.n_action_steps = n_action_steps
        self.config = cfg

        # Dataset metadata (features + normalization stats). Read from the local copy if present,
        # otherwise fetched from the Hub (needs `hf auth login`, the dataset is private).
        root = DATASET_DIR / dataset_name
        meta = LeRobotDatasetMetadata(f"{REPO_OWNER}/{dataset_name}", root=root if root.exists() else None)

        self.policy = make_policy(cfg, ds_meta=meta, rename_map=RENAME_MAP)
        self.policy.eval()

        stats = rename_stats(meta.stats, RENAME_MAP)
        features = {**self.policy.config.input_features, **self.policy.config.output_features}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            dataset_stats=stats,
            preprocessor_overrides={
                "device_processor": {"device": self.device.type},
                "rename_observations_processor": {"rename_map": RENAME_MAP},
                "normalizer_processor": {"stats": stats, "features": features,
                                         "norm_map": self.policy.config.normalization_mapping},
            },
            postprocessor_overrides={
                "unnormalizer_processor": {"stats": stats, "features": self.policy.config.output_features,
                                           "norm_map": self.policy.config.normalization_mapping},
            },
        )
        self.camera_keys = dict(CAMERA_KEYS)   # mujoco camera name -> dataset key
        self.reset()

    def reset(self) -> None:
        """Call at the start of every episode: clears the queued action chunk."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @property
    def will_replan(self) -> bool:
        """True if the next `act` call runs the network (the action queue is empty)."""
        return len(self.policy._queues[ACTION]) == 0

    def make_observation(self, state: np.ndarray, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """state: 8 floats; images: {mujoco camera name: HxWx3 uint8} (from CameraRenderer.render_all)."""
        obs = {"observation.state": np.asarray(state, dtype=np.float32)}
        for cam, image in images.items():
            obs[self.camera_keys[cam]] = np.ascontiguousarray(image)
        return obs

    def act(self, observation: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        """One 8-D action [7 joint targets, gripper 0/255] for the current observation."""
        action = predict_action(observation, self.policy, self.device, self.preprocessor, self.postprocessor,
                                use_amp=False, task=instruction, robot_type="franka_panda")
        action = action.squeeze(0).to("cpu", torch.float32).numpy()
        action[7] = 255.0 if action[7] > GRIPPER_THRESHOLD else 0.0
        return action
