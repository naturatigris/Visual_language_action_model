"""Tabletop scene: Franka Panda mounted on a table with three coloured cubes and a container.

Run the viewer:  uv run python -m simulation.environment [--seed N]
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "simulation.yaml"
PANDA_XML = PROJECT_DIR / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

TABLE_TOP_Z = 0.40      # height of the table surface (m)
CUBE_HALF = 0.025       # cubes are 5 cm on each side (Panda gripper opens to 8 cm)

# Cube name -> (rgba, xy position on the table). The robot base is at the origin.
CUBES = {
    "red_cube":   ("0.85 0.15 0.15 1", (0.50, -0.15)),
    "green_cube": ("0.15 0.70 0.20 1", (0.50,  0.00)),
    "blue_cube":  ("0.15 0.30 0.85 1", (0.50,  0.15)),
}

# Open-top container (fixed to the table) that the cubes get placed into.
CONTAINER_XY = (0.45, 0.33)
CONTAINER_HALF = 0.07         # outer half-width: 14 x 14 cm footprint, ~12 cm inside
CONTAINER_WALL_HALF = 0.005   # walls and floor are 1 cm thick
CONTAINER_HEIGHT = 0.05       # wall height (m), low enough for the gripper to drop over

# Panda "home" pose from Menagerie's panda.xml: 7 arm joints + 2 fingers, and 8 actuator targets.
HOME_ARM_QPOS = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
HOME_FINGER_QPOS = [0.04, 0.04]
HOME_CTRL = HOME_ARM_QPOS + [255]   # actuator8 = gripper, 255 = fully open
# Robot element names get this prefix when attached, e.g. "panda/joint1", "panda/hand", "panda/actuator8".
ROBOT_PREFIX = "panda/"
TCP_OFFSET = 0.1034        # hand frame -> point between the fingertip pads (m)
TCP_SITE = f"{ROBOT_PREFIX}tcp"
ROBOT_JOINTS = [f"{ROBOT_PREFIX}joint{i}" for i in range(1, 8)] + [
    f"{ROBOT_PREFIX}finger_joint1", f"{ROBOT_PREFIX}finger_joint2"]

# Front camera: in front of the table, above it, aimed at the middle of the randomization region.
# Chosen so a 5 cm cube is ~20 px wide in a 256x256 image and the whole region (y +-0.30) fits.
FRONT_CAM_POS = (1.0, 0.0, 1.15)
FRONT_CAM_TARGET = (0.47, 0.0, 0.40)
FRONT_CAM_FOVY = 45
# Wrist camera on the hand (hand frame: z points along the fingers, fingers slide along y).
# Offset to the side of the hand, tilted to look at the space between and just past the fingertips.
WRIST_CAM_POS = (0.06, 0.0, 0.065)   # just below the hand housing, so the housing stays out of view
WRIST_CAM_TARGET = (0.0, 0.0, TCP_OFFSET + 0.05)
WRIST_CAM_FOVY = 95


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera `xyaxes` (image right, image up) for a camera at `pos` looking at `target`."""
    forward = np.subtract(target, pos, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    image_up = np.cross(right, forward)
    return np.concatenate([right, image_up])


def _vec(v) -> str:
    return " ".join(f"{float(x):.4f}" for x in v)


def _scene_xml() -> str:
    """Everything except the robot: floor, table, container, cubes, lights and camera."""
    top = TABLE_TOP_Z
    leg_h = (top - 0.04) / 2
    cubes = "\n".join(
        f"""
    <body name="{name}" pos="{x} {y} {top + CUBE_HALF}">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            rgba="{rgba}" mass="0.05" friction="1.5 0.01 0.001"/>
    </body>"""
        for name, (rgba, (x, y)) in CUBES.items()
    )
    legs = "\n".join(
        f'      <geom type="box" size="0.025 0.025 {leg_h}" pos="{lx} {ly} {leg_h}" material="table_leg"/>'
        for lx in (-0.15, 0.85) for ly in (-0.40, 0.40)
    )
    return _template(top, legs, cubes, _container_xml())


def _container_xml() -> str:
    """A static open-top box: a thin floor plate plus four walls, with a site at the inside centre."""
    h, t = CONTAINER_HALF, CONTAINER_WALL_HALF
    wall_hz = CONTAINER_HEIGHT / 2
    x, y = CONTAINER_XY
    walls = [
        (f"{t} {h} {wall_hz}",  f"{h - t} 0 {wall_hz}"),    # +x wall
        (f"{t} {h} {wall_hz}",  f"{-(h - t)} 0 {wall_hz}"),  # -x wall
        (f"{h} {t} {wall_hz}",  f"0 {h - t} {wall_hz}"),    # +y wall
        (f"{h} {t} {wall_hz}",  f"0 {-(h - t)} {wall_hz}"),  # -y wall
    ]
    wall_geoms = "\n".join(
        f'      <geom name="container_wall{i}" type="box" size="{size}" pos="{pos}" material="container"/>'
        for i, (size, pos) in enumerate(walls)
    )
    return f"""
    <body name="container" pos="{x} {y} {TABLE_TOP_Z}">
      <geom name="container_floor" type="box" size="{h} {h} {t}" pos="0 0 {t}" material="container"/>
{wall_geoms}
      <site name="container_center" pos="0 0 {2 * t}" size="0.01" rgba="1 1 0 0"/>
    </body>"""


def _template(top: float, legs: str, cubes: str, container: str) -> str:
    return f"""
<mujoco model="tabletop">
  <option integrator="implicitfast"/>  <!-- same integrator as Menagerie's panda.xml -->

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="floor" type="2d" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="table_top" rgba="0.62 0.47 0.33 1"/>
    <material name="table_leg" rgba="0.35 0.27 0.20 1"/>
    <material name="container" rgba="0.85 0.85 0.80 1"/>
  </asset>

  <worldbody>
    <light pos="0.4 0 2.0" dir="0 0 -1" directional="true"/>
    <light pos="1.5 -1.0 2.0" dir="-0.6 0.4 -0.7"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>

    <body name="table">
      <geom name="table_top" type="box" size="0.55 0.45 0.02" pos="0.35 0 {top - 0.02}" material="table_top"/>
{legs}
    </body>
{container}
{cubes}

    <!-- Fixed camera in front of the table, looking down at the workspace (+y is image right) -->
    <camera name="front_camera" pos="{_vec(FRONT_CAM_POS)}" xyaxes="{_vec(look_at_xyaxes(FRONT_CAM_POS, FRONT_CAM_TARGET))}"
            fovy="{FRONT_CAM_FOVY}"/>
  </worldbody>
</mujoco>
"""


def build_model() -> mujoco.MjModel:
    """Compile the table scene with the Panda attached on top of the table."""
    scene = mujoco.MjSpec.from_string(_scene_xml())
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))

    # Drop the Panda's own keyframe (it only covers the robot's 9 joints, not the cubes).
    for key in list(panda.keys):
        panda.delete(key)

    # Tool centre point: between the fingertip pads, 0.1034 m below the hand (the standard Franka TCP).
    panda.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005] * 3, rgba=[1, 0, 1, 0])

    # Gravity compensation, as the real Panda's controller does internally. Without it the position
    # servos sag under gravity and the hand settles a few millimetres below its target.
    for body in panda.bodies:
        body.gravcomp = 1.0

    mount = scene.worldbody.add_frame(pos=[0, 0, TABLE_TOP_Z])
    scene.attach(panda, frame=mount, prefix=ROBOT_PREFIX)

    # Wrist camera, added after attaching so it keeps the plain name "wrist_camera".
    # In the hand frame "up" for the look-at is the hand's -x axis, so image right = hand +y (finger travel).
    xy = look_at_xyaxes(WRIST_CAM_POS, WRIST_CAM_TARGET, up=(-1, 0, 0))
    rot = np.stack([xy[:3], xy[3:], np.cross(xy[:3], xy[3:])], axis=1)   # columns = camera x, y, z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    scene.body(f"{ROBOT_PREFIX}hand").add_camera(
        name="wrist_camera", pos=list(WRIST_CAM_POS), quat=quat.tolist(), fovy=WRIST_CAM_FOVY)

    # Add a full-scene "home" keyframe: robot at its home pose, cubes where the XML placed them.
    # Look up each robot joint's slot in qpos by name (the cubes' freejoints come first).
    model = scene.compile()
    qpos = model.qpos0.copy()
    for name, value in zip(ROBOT_JOINTS, HOME_ARM_QPOS + HOME_FINGER_QPOS):
        qpos[model.joint(name).qposadr[0]] = value
    scene.add_key(name="home", qpos=qpos.tolist(), ctrl=HOME_CTRL)
    return scene.compile()


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TabletopEnv:
    """The scene plus episode logic: randomized reset, object queries and success checks."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.cube_names = list(CUBES)
        self._cube_qpos_adr = {n: self.model.joint(f"{n}_joint").qposadr[0] for n in self.cube_names}
        self._container_id = self.model.body("container").id
        self.reset()

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int | None = None) -> dict:
        """Robot to home, then cubes and container at random non-overlapping spots.

        seed=None keeps the default layout from the XML. Returns {object name: (x, y)}.
        """
        reset_to_home(self.model, self.data)
        layout = self._default_layout() if seed is None else self._sample_layout(np.random.default_rng(seed))

        cx, cy = layout["container"]
        self.model.body_pos[self._container_id] = [cx, cy, TABLE_TOP_Z]
        for name in self.cube_names:
            x, y = layout[name]
            adr = self._cube_qpos_adr[name]
            self.data.qpos[adr:adr + 7] = [x, y, TABLE_TOP_Z + CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return layout

    def _default_layout(self) -> dict:
        layout = {name: xy for name, (_, xy) in CUBES.items()}
        layout["container"] = CONTAINER_XY
        return layout

    def _sample_layout(self, rng: np.random.Generator) -> dict:
        r = self.config["randomization"]
        (x0, x1), (y0, y1) = r["x_range"], r["y_range"]
        # Keep the container footprint on the region, so its centre gets a smaller range.
        m = CONTAINER_HALF
        for _ in range(r["max_attempts"]):
            container = np.array([rng.uniform(x0 + m, x1 - m), rng.uniform(y0 + m, y1 - m)])
            cubes = rng.uniform([x0, y0], [x1, y1], size=(len(self.cube_names), 2))
            if np.any(np.linalg.norm(cubes - container, axis=1) < r["min_container_spacing"]):
                continue
            gaps = [np.linalg.norm(a - b) for i, a in enumerate(cubes) for b in cubes[i + 1:]]
            if min(gaps) < r["min_cube_spacing"]:
                continue
            layout = {n: tuple(xy) for n, xy in zip(self.cube_names, cubes)}
            layout["container"] = tuple(container)
            return layout
        raise RuntimeError("Could not sample a valid layout; loosen the randomization ranges")

    # ---------------------------------------------------------------- queries
    def cube_position(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def container_position(self) -> np.ndarray:
        """Centre of the container's footprint, at table height."""
        return self.data.xpos[self._container_id].copy()

    def container_floor_z(self) -> float:
        return float(self.data.site("container_center").xpos[2])

    def container_rim_z(self) -> float:
        return TABLE_TOP_Z + CONTAINER_HEIGHT

    def in_container(self, name: str) -> bool:
        """Cube centre within the container's inner walls and below its rim."""
        p = self.cube_position(name)
        c = self.container_position()
        inner = CONTAINER_HALF - 2 * CONTAINER_WALL_HALF
        return bool(np.all(np.abs(p[:2] - c[:2]) < inner) and p[2] < self.container_rim_z())

    def check_success(self, target: str) -> bool:
        """Target cube in the container, gripper open, TCP clear above the rim, no distractor in the box."""
        gripper_open = self.data.ctrl[self.model.actuator(f"{ROBOT_PREFIX}actuator8").id] > 127
        tcp_z = self.data.site(TCP_SITE).xpos[2]
        arm_clear = tcp_z > self.container_rim_z() + self.config["success"]["arm_clearance"]
        return self.in_container(target) and gripper_open and arm_clear and not self.wrong_object(target)

    def wrong_object(self, target: str) -> bool:
        """A non-target cube was lifted or ended up in the container."""
        lift = self.config["success"]["wrong_object_lift"]
        for name in self.cube_names:
            if name == target:
                continue
            if self.in_container(name) or self.cube_position(name)[2] > TABLE_TOP_Z + CUBE_HALF + lift:
                return True
        return False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Open the tabletop scene in the MuJoCo viewer.")
    parser.add_argument("--seed", type=int, default=None, help="random layout (default: fixed layout)")
    args = parser.parse_args()

    env = TabletopEnv()
    layout = env.reset(args.seed)
    model, data = env.model, env.data
    print(f"Scene: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators, "
          f"{model.ncam} camera(s)")
    print("Layout:", {k: tuple(round(float(v), 3) for v in xy) for k, xy in layout.items()})
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
