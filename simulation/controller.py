"""RobotController: Cartesian moves for the Panda via differential (Jacobian) IK.

Each control step the controller turns a Cartesian goal for the tool centre point (TCP) into
**absolute joint targets** for the 7 arm position servos, plus a gripper command. That 8-D vector
is the action we record for training:

    action = [q1_target, ..., q7_target, gripper]     gripper: 0 = closed, 255 = open

Orientation is held fixed: the gripper points straight down with the same yaw as the home pose
(fingers close along world x, matching the axis-aligned cubes).

Quick test (prints the robot description, then moves through waypoints and grasps a cube):
    uv run python -m simulation.controller
"""

from typing import Callable

import mujoco
import numpy as np

from simulation.environment import HOME_ARM_QPOS, ROBOT_PREFIX, TCP_SITE, build_model, reset_to_home

CONTROL_HZ = 20                 # policy / action rate; physics substeps run underneath
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
MAX_TRACKING_ERROR = 0.05       # m; caps the IK error so one step can never command a huge jump

ARM_JOINTS = [f"{ROBOT_PREFIX}joint{i}" for i in range(1, 8)]
FINGER_JOINTS = [f"{ROBOT_PREFIX}finger_joint1", f"{ROBOT_PREFIX}finger_joint2"]
ARM_ACTUATORS = [f"{ROBOT_PREFIX}actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = f"{ROBOT_PREFIX}actuator8"

StepCallback = Callable[[np.ndarray], None]


class RobotController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_hz: int = CONTROL_HZ,
                 max_speed: float = 0.25, damping: float = 0.05, nullspace_gain: float = 0.5):
        """
        max_speed:       TCP speed limit (m/s) of the moving reference point that the IK tracks.
        damping:         damped-least-squares lambda; keeps the IK stable near singularities.
        nullspace_gain:  pull of the redundant 7th DoF towards the home posture (avoids drifting into odd elbows).
        """
        self.model, self.data = model, data
        self.n_substeps = round(1.0 / (control_hz * model.opt.timestep))
        self.control_dt = self.n_substeps * model.opt.timestep
        self.max_step = max_speed * self.control_dt
        self.damping = damping
        self.nullspace_gain = nullspace_gain

        self.qpos_idx = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dof_idx = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.finger_qpos_idx = np.array([model.joint(j).qposadr[0] for j in FINGER_JOINTS])
        self.arm_act_idx = np.array([model.actuator(a).id for a in ARM_ACTUATORS])
        self.gripper_act_idx = model.actuator(GRIPPER_ACTUATOR).id
        self.q_low, self.q_high = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS]].T
        self.site_id = model.site(TCP_SITE).id
        self.q_home = np.array(HOME_ARM_QPOS)

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.reset_targets()

    # ---------------------------------------------------------------- state readout
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.dof_idx].copy()

    def ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def ee_velocity(self) -> np.ndarray:
        """Linear TCP velocity in the world frame (m/s)."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self.site_id)
        return self._jacp @ self.data.qvel

    def ee_rotation(self) -> np.ndarray:
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def gripper_opening(self) -> float:
        """Distance between the fingers (m): 0.08 fully open, ~0.05 holding a 5 cm cube."""
        return float(self.data.qpos[self.finger_qpos_idx].sum())

    def state(self) -> np.ndarray:
        """Proprioceptive state for observation.state: 7 arm joints + gripper opening (no object poses)."""
        return np.append(self.joint_positions(), self.gripper_opening())

    # ---------------------------------------------------------------- targets
    def reset_targets(self) -> None:
        """Hold the current pose: goal = current TCP pose, gripper command = current ctrl."""
        mujoco.mj_forward(self.model, self.data)
        self.target_pos = self.ee_position()
        self.ref_pos = self.target_pos.copy()
        self.target_rot = self._top_down_rotation()
        self.gripper_cmd = float(self.data.ctrl[self.gripper_act_idx])

    def set_target(self, pos) -> None:
        self.target_pos = np.asarray(pos, dtype=float).copy()

    def open_gripper(self) -> None:
        self.gripper_cmd = GRIPPER_OPEN

    def close_gripper(self) -> None:
        self.gripper_cmd = GRIPPER_CLOSED

    def position_error(self) -> float:
        return float(np.linalg.norm(self.target_pos - self.ee_position()))

    # ---------------------------------------------------------------- control
    def compute_action(self) -> np.ndarray:
        """One differential-IK step: 8-D action [7 joint targets, gripper command]."""
        m, d = self.model, self.data
        q = self.joint_positions()

        # The reference point advances towards the goal at the speed limit; IK tracks the reference.
        # (Clipping the error directly would make the arm crawl, because the servos lag each command.)
        to_goal = self.target_pos - self.ref_pos
        dist = np.linalg.norm(to_goal)
        self.ref_pos = self.target_pos.copy() if dist <= self.max_step else self.ref_pos + to_goal * (self.max_step / dist)

        # 6-D task-space error: position (bounded for safety) and orientation.
        pos_err = self.ref_pos - self.ee_position()
        norm = np.linalg.norm(pos_err)
        if norm > MAX_TRACKING_ERROR:
            pos_err *= MAX_TRACKING_ERROR / norm
        rot_err = _rotation_error(self.target_rot, self.ee_rotation())
        err = np.concatenate([pos_err, rot_err])

        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site_id)
        J = np.vstack([self._jacp, self._jacr])[:, self.dof_idx]        # 6 x 7

        # Damped least squares, plus a null-space pull towards the home posture.
        JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
        dq = J_pinv @ err
        dq += (np.eye(7) - J_pinv @ J) @ (self.nullspace_gain * (self.q_home - q) * self.control_dt)

        q_target = np.clip(q + dq, self.q_low, self.q_high)
        return np.append(q_target, self.gripper_cmd)

    def apply_action(self, action: np.ndarray) -> None:
        """Send an 8-D action to the actuators and advance physics by one control period."""
        self.data.ctrl[self.arm_act_idx] = action[:7]
        self.data.ctrl[self.gripper_act_idx] = action[7]
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def step(self) -> np.ndarray:
        """Compute the action for the current goal, apply it, and return it (for recording)."""
        action = self.compute_action()
        self.apply_action(action)
        return action

    # ---------------------------------------------------------------- blocking helpers
    def move_to_position(self, pos, tol: float = 0.003, timeout: float = 4.0,
                         on_step: StepCallback | None = None) -> bool:
        """Step until the TCP is within `tol` metres of `pos` and has settled. Returns success."""
        self.set_target(pos)
        for _ in range(int(timeout / self.control_dt)):
            action = self.step()
            if on_step:
                on_step(action)
            # Settled = TCP close and nearly still. (Joint speeds can stay non-zero while the
            # null-space term slowly relaxes the elbow without moving the hand.)
            if self.position_error() < tol and np.linalg.norm(self.ee_velocity()) < 0.01:
                return True
        return False

    def wait(self, seconds: float, on_step: StepCallback | None = None) -> None:
        """Keep holding the current goal (e.g. while the gripper opens or closes)."""
        for _ in range(max(1, round(seconds / self.control_dt))):
            action = self.step()
            if on_step:
                on_step(action)

    def _top_down_rotation(self) -> np.ndarray:
        """Gripper pointing straight down (TCP z = world -z), fingers closing along world x."""
        return np.array([[0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0]])


def _rotation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) that rotates R_current onto R_target."""
    q_t, q_c, q_err = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_t, R_target.ravel())
    mujoco.mju_mat2Quat(q_c, R_current.ravel())
    mujoco.mju_negQuat(q_c, q_c)
    mujoco.mju_mulQuat(q_err, q_t, q_c)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


def describe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Step 8: print the robot's joints, actuators and current end-effector state."""
    print(f"timestep {model.opt.timestep} s, control at {CONTROL_HZ} Hz\n")
    print(f"{'joint':<22}{'qpos idx':>9}{'range (rad / m)':>24}{'qpos':>9}")
    for j in ARM_JOINTS + FINGER_JOINTS:
        jt = model.joint(j)
        lo, hi = jt.range
        print(f"{j:<22}{jt.qposadr[0]:>9}{f'[{lo:+.4f}, {hi:+.4f}]':>24}{data.qpos[jt.qposadr[0]]:>+9.4f}")
    print(f"\n{'actuator':<22}{'ctrlrange':>24}{'kp':>9}{'force limit':>14}")
    for a in ARM_ACTUATORS + [GRIPPER_ACTUATOR]:
        act = model.actuator(a)
        lo, hi = act.ctrlrange
        print(f"{a:<22}{f'[{lo:+.4f}, {hi:+.4f}]':>24}{act.gainprm[0]:>9.4g}{f'±{act.forcerange[1]:g}':>14}")
    ctrl = RobotController(model, data)
    print(f"\nTCP position  {ctrl.ee_position().round(4)}")
    print(f"TCP rotation\n{ctrl.ee_rotation().round(3)}")
    print(f"gripper opening {ctrl.gripper_opening():.4f} m")


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    describe(model, data)

    ctrl = RobotController(model, data)
    cube = data.body("red_cube").xpos.copy()
    above = cube + [0, 0, 0.12]
    print("\nReaching test:")
    for name, goal in [("above red cube", above), ("grasp height", cube), ("back up", above),
                       ("far left", [0.35, 0.30, 0.60]), ("far right", [0.60, -0.30, 0.55])]:
        ok = ctrl.move_to_position(goal)
        print(f"  {name:<16} ok={ok!s:<5} error={ctrl.position_error() * 1000:5.1f} mm  t={data.time:5.2f} s")

    print("\nGrasp test:")
    ctrl.move_to_position(above)
    ctrl.move_to_position(cube)
    ctrl.close_gripper()
    ctrl.wait(0.8)
    print(f"  gripper opening after close: {ctrl.gripper_opening() * 1000:.1f} mm")
    ctrl.move_to_position(cube + [0, 0, 0.20])
    ctrl.wait(1.0)
    lifted = data.body("red_cube").xpos[2] - cube[2]
    print(f"  cube lifted {lifted * 1000:.0f} mm (target 200)  ->  {'held' if lifted > 0.15 else 'DROPPED'}")


if __name__ == "__main__":
    main()
