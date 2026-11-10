"""Scripted pick-and-place expert: the demonstrator whose trajectories we record for training.

It cheats (reads exact cube and container positions from the simulator), which is fine for
generating demos; the learned policy only gets camera images, robot state and the instruction.

Measure the success rate over many random layouts:
    uv run python -m data.scripted_expert --episodes 100
    uv run python -m data.scripted_expert --episodes 3 --video outputs/expert.mp4

Watch it live (macOS needs mjpython for the passive viewer):
    uv run mjpython -m data.scripted_expert --view --episodes 5
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from simulation.controller import RobotController
from simulation.environment import CUBE_HALF, TABLE_TOP_Z, TabletopEnv

StepCallback = Callable[[np.ndarray], None]

HOVER_HEIGHT = 0.15         # TCP height above the table while approaching / carrying (clears rim + cubes)
PLACE_CLEARANCE = 0.01      # release the cube with its bottom this far above the container floor
GRIPPER_WAIT = 0.6          # seconds to let the fingers close or open


@dataclass
class EpisodeResult:
    success: bool
    wrong_object: bool
    failed_stage: str | None    # first motion stage that timed out, if any
    duration: float             # simulated seconds
    n_steps: int                # control steps = recorded frames


class ScriptedExpert:
    """Runs the 8-stage pick-and-place on a TabletopEnv.

    With `rng` given, each episode gets small random variations (speed, hover height, where in the
    container the cube is dropped) so the demos are not all identical.
    """

    def __init__(self, env: TabletopEnv, rng: np.random.Generator | None = None):
        self.env = env
        self.rng = rng

    def run(self, target: str, on_step: StepCallback | None = None) -> EpisodeResult:
        env, rng = self.env, self.rng
        speed = rng.uniform(0.20, 0.30) if rng else 0.25
        hover = TABLE_TOP_Z + HOVER_HEIGHT + (rng.uniform(-0.02, 0.02) if rng else 0.0)
        drop_offset = np.append(rng.uniform(-0.015, 0.015, size=2), 0) if rng else np.zeros(3)

        ctrl = RobotController(env.model, env.data, control_hz=env.config["fps"], max_speed=speed)
        ctrl.reset_targets()
        steps = 0
        t0 = env.data.time
        timeout = env.config["success"]["timeout_s"]

        def count(action):
            nonlocal steps
            steps += 1
            if on_step:
                on_step(action)

        cube = env.cube_position(target)
        box = env.container_position() + drop_offset
        place_z = env.container_floor_z() + CUBE_HALF + PLACE_CLEARANCE

        stages = [
            ("move above cube",      lambda: ctrl.move_to_position([cube[0], cube[1], hover], on_step=count)),
            ("lower to cube",        lambda: ctrl.move_to_position(cube, on_step=count)),
            ("close gripper",        lambda: (ctrl.close_gripper(), ctrl.wait(GRIPPER_WAIT, count), True)[-1]),
            ("lift",                 lambda: ctrl.move_to_position([cube[0], cube[1], hover], on_step=count)),
            ("move above container", lambda: ctrl.move_to_position([box[0], box[1], hover], on_step=count)),
            ("lower into container", lambda: ctrl.move_to_position([box[0], box[1], place_z], on_step=count)),
            ("open gripper",         lambda: (ctrl.open_gripper(), ctrl.wait(GRIPPER_WAIT, count), True)[-1]),
            ("retreat",              lambda: ctrl.move_to_position([box[0], box[1], hover], on_step=count)),
        ]
        failed = None
        for name, run_stage in stages:
            if not run_stage() or env.data.time - t0 > timeout:
                failed = name
                break

        return EpisodeResult(
            success=failed is None and env.check_success(target),
            wrong_object=env.wrong_object(target),
            failed_stage=failed,
            duration=env.data.time - t0,
            n_steps=steps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the scripted expert over random layouts.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="first layout seed")
    parser.add_argument("--no-noise", action="store_true", help="disable per-episode variations")
    parser.add_argument("--video", type=str, default=None, help="save the episodes as an mp4")
    parser.add_argument("--view", action="store_true",
                        help="watch live in the MuJoCo viewer (macOS: run with `uv run mjpython`)")
    args = parser.parse_args()

    env = TabletopEnv()
    if args.view:
        return view(env, args)
    targets = env.config["targets"]
    writer = cam = None
    if args.video:
        import cv2
        from simulation.camera import CameraRenderer
        cam = CameraRenderer(env.model, 480, 640)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), env.config["fps"], (640, 480))

    def record(_action):
        writer.write(cam.render(env.data)[:, :, ::-1])

    results, wall = [], time.time()
    for ep in range(args.episodes):
        seed = args.seed + ep
        env.reset(seed)
        target = targets[ep % len(targets)]
        rng = None if args.no_noise else np.random.default_rng(10_000 + seed)
        res = ScriptedExpert(env, rng).run(target, on_step=record if writer else None)
        results.append(res)
        status = "ok  " if res.success else f"FAIL ({res.failed_stage or 'success check'})"
        print(f"seed {seed:4d}  {target:<10} {status:<34} {res.duration:5.1f} s  {res.n_steps:4d} steps"
              + ("  WRONG OBJECT" if res.wrong_object else ""))

    if writer:
        writer.release()
        cam.close()
        print(f"video -> {args.video}")
    n_ok = sum(r.success for r in results)
    print(f"\nsuccess {n_ok}/{len(results)} = {100 * n_ok / len(results):.1f}%   "
          f"wrong-object {sum(r.wrong_object for r in results)}   "
          f"mean {np.mean([r.duration for r in results]):.1f} s / {np.mean([r.n_steps for r in results]):.0f} steps per episode   "
          f"(wall {time.time() - wall:.1f} s)")


def view(env: TabletopEnv, args) -> None:
    """Play episodes live in the passive viewer, in real time."""
    import mujoco.viewer

    targets = env.config["targets"]
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        dt = 1.0 / env.config["fps"]

        def show(_action):
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            time.sleep(dt)   # one control step = 1/fps seconds of sim time -> roughly real time

        try:
            for ep in range(args.episodes):
                seed = args.seed + ep
                env.reset(seed)
                viewer.sync()
                target = targets[ep % len(targets)]
                print(f"seed {seed}: put the {target.replace('_', ' ')} in the container")
                rng = None if args.no_noise else np.random.default_rng(10_000 + seed)
                res = ScriptedExpert(env, rng).run(target, on_step=show)
                print(f"  -> {'success' if res.success else 'FAILED'} ({res.duration:.1f} s)")
                time.sleep(1.0)
            print("Done. Close the viewer window to exit.")
            while viewer.is_running():
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
