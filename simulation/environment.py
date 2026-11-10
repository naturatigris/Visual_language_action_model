"""Tabletop scene: Franka Panda mounted on a table with three coloured cubes.

Run the viewer:  uv run python -m simulation.environment
"""

from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_DIR = Path(__file__).resolve().parents[1]
PANDA_XML = PROJECT_DIR / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

TABLE_TOP_Z = 0.40      # height of the table surface (m)
CUBE_HALF = 0.025       # cubes are 5 cm on each side (Panda gripper opens to 8 cm)

# Cube name -> (rgba, xy position on the table). The robot base is at the origin.
CUBES = {
    "red_cube":   ("0.85 0.15 0.15 1", (0.50, -0.15)),
    "green_cube": ("0.15 0.70 0.20 1", (0.50,  0.00)),
    "blue_cube":  ("0.15 0.30 0.85 1", (0.50,  0.15)),
}

# Panda "home" pose from Menagerie's panda.xml: 7 arm joints + 2 fingers, and 8 actuator targets.
HOME_ARM_QPOS = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
HOME_FINGER_QPOS = [0.04, 0.04]
HOME_CTRL = HOME_ARM_QPOS + [255]   # actuator8 = gripper, 255 = fully open
# Robot element names get this prefix when attached, e.g. "panda/joint1", "panda/hand", "panda/actuator8".
ROBOT_PREFIX = "panda/"
ROBOT_JOINTS = [f"{ROBOT_PREFIX}joint{i}" for i in range(1, 8)] + [
    f"{ROBOT_PREFIX}finger_joint1", f"{ROBOT_PREFIX}finger_joint2"]


def _scene_xml() -> str:
    """Everything except the robot: floor, table, cubes, lights and camera."""
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
  </asset>

  <worldbody>
    <light pos="0.4 0 2.0" dir="0 0 -1" directional="true"/>
    <light pos="1.5 -1.0 2.0" dir="-0.6 0.4 -0.7"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>

    <body name="table">
      <geom name="table_top" type="box" size="0.55 0.45 0.02" pos="0.35 0 {top - 0.02}" material="table_top"/>
{legs}
    </body>
{cubes}

    <!-- Fixed camera in front of the table, looking back at the robot and cubes -->
    <camera name="front_camera" pos="1.6 0 1.3" xyaxes="0 1 0 -0.485 0 0.875" fovy="55"/>
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

    mount = scene.worldbody.add_frame(pos=[0, 0, TABLE_TOP_Z])
    scene.attach(panda, frame=mount, prefix=ROBOT_PREFIX)

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


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    print(f"Scene: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators, "
          f"{model.ncam} camera(s)")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
