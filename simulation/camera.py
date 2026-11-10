"""Camera capture: render RGB images from the scene's named cameras and save them to disk.

Quick check (saves images to outputs/camera/):
    uv run python -m simulation.camera
"""

from pathlib import Path

import cv2
import mujoco
import numpy as np

from simulation.environment import PROJECT_DIR, build_model, reset_to_home

DEFAULT_SIZE = (256, 256)          # (height, width); matches observation.images.front in PLAN.md
DEFAULT_CAMERAS = ("front_camera", "wrist_camera")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "camera"


class CameraRenderer:
    """Wraps one offscreen mujoco.Renderer and renders any of the model's named cameras.

    Create it once and reuse it: making a new Renderer every frame is slow and leaks GL contexts.
    """

    def __init__(self, model: mujoco.MjModel, height: int = DEFAULT_SIZE[0],
                 width: int = DEFAULT_SIZE[1], cameras: tuple[str, ...] = DEFAULT_CAMERAS):
        for name in cameras:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"Camera {name!r} not found in the model")
        self.model = model
        self.cameras = tuple(cameras)
        self.height, self.width = height, width
        self._renderer = mujoco.Renderer(model, height, width)

    def render(self, data: mujoco.MjData, camera: str | None = None) -> np.ndarray:
        """RGB image of shape (height, width, 3), dtype uint8, from `camera` (default: the first one)."""
        self._renderer.update_scene(data, camera=camera or self.cameras[0])
        return self._renderer.render().copy()   # copy: the renderer reuses its buffer

    def render_all(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """One image per configured camera, keyed by camera name."""
        return {name: self.render(data, name) for name in self.cameras}

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "CameraRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save an RGB uint8 image (OpenCV writes BGR, so the channels are flipped here)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image[:, :, ::-1]):
        raise IOError(f"Could not write {path}")
    return path


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    # Training resolution, plus a larger preview that is easier to inspect by eye.
    for (h, w), suffix in ((DEFAULT_SIZE, ""), ((480, 640), "_preview")):
        with CameraRenderer(model, h, w) as cam:
            for name, image in cam.render_all(data).items():
                print(f"saved {save_image(image, OUTPUT_DIR / f'{name}{suffix}.png')}  {image.shape}")


if __name__ == "__main__":
    main()
