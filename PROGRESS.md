# Progress Log

The full project plan is in [PLAN.md](PLAN.md). This file records where things stand.

_Last updated: 2026-11-10_

---

## Done

### Planning
- `PLAN.md` is written. Main model: **SmolVLA** fine-tuned with LeRobot. Optional compact CLIP+MLP baseline in Phase 9. Demos recorded directly in LeRobot format. Actions are absolute joint targets (8-D: 7 arm joints + gripper).
- Blog title: *From Simulation to Autonomous Robot: Building and Training a Vision-Language-Action System*.
- First task: "Put the red/blue cube in the container" (the colour choice forces the model to use language).

### Environment setup
- uv project, created by the user with `uv init --package`. Python 3.12 is pinned in `.python-version`.
- Installed: `mujoco` 4.0.0, `lerobot`, `numpy`, `opencv-python`, `matplotlib` (see `pyproject.toml`).
- **Always run with `uv run ...`.** It uses `.venv/` automatically; plain `python` is the system 3.9.
- To clear the IDE warning "Import mujoco could not be resolved", set the interpreter to `.venv/bin/python`.

### Phase 1: MuJoCo works ✅
- A falling box/ball test confirmed physics and the viewer work. `main.py` now opens the tabletop scene instead.

### Phase 2: Robot + tabletop scene ✅
- Only the Panda folder of MuJoCo Menagerie was sparse-cloned, into `assets/mujoco_menagerie/franka_emika_panda/`. It is git-ignored.
  - To re-download: `git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie && cd assets/mujoco_menagerie && git sparse-checkout set franka_emika_panda`
- `simulation/environment.py` contains:
  - A table (top at z = 0.40 m) with the Panda mounted at the origin on the table top.
  - Red, green and blue 5 cm cubes with freejoints at x = 0.5, y = -0.15 / 0 / +0.15.
  - A static open-top `container` (off-white, 14×14 cm outside, 5 cm walls, 1 cm thick) at xy (0.45, 0.33). It has no joint, so `qpos` is unchanged. The site `container_center` sits on the inside floor (z = 0.41), for success checks.
  - `front_camera` at pos (1.6, 0, 1.3), which sees the whole arm, the gripper and all cubes.
  - `build_model()`, which loads `panda.xml` with `MjSpec` and attaches it with the prefix `panda/`. It also adds a full-scene `home` keyframe.
  - `reset_to_home(model, data)`.
- Verified: the arm holds its home pose (drift < 0.01 rad over 2 s), the cubes rest at z = 0.425, and front and side renders look correct.
- Robot names: `panda/joint1..7`, `panda/finger_joint1/2`, `panda/hand`, `panda/actuator1..8`.
  - `actuator8` is the gripper: 0 = closed, 255 = open.
  - The cubes' freejoints come **first** in `qpos`, so always look up a joint's slot with `model.joint(name).qposadr`.

### Step 7: Camera capture ✅
- `simulation/camera.py` has `CameraRenderer(model, height=256, width=256, cameras=("front_camera",))`, with `.render(data, camera)` (returns RGB uint8 HxWx3), `.render_all(data)` and `.close()`. It works as a context manager. Make it once and reuse it.
- `save_image(rgb, path)` flips the image to BGR for OpenCV.
- `uv run python -m simulation.camera` saves a 256×256 image and a 640×480 preview to `outputs/` (git-ignored).
- At 256×256 all three cubes and the container are visible, but each cube is only about 8 px wide. Think about moving `front_camera` closer or narrowing the FOV, or adding a `wrist_camera`.

### Phase 3, Steps 8–9: Inspection + RobotController ✅
- In `environment.py`:
  - added the site `panda/tcp` on the hand, 0.1034 m below it, between the fingertip pads (`TCP_SITE`);
  - turned on **gravity compensation** (`gravcomp=1`) on all Panda bodies, as the real robot does. Arm drift at home is now about 0.
- `simulation/controller.py`, `RobotController(model, data)`:
  - runs at 20 Hz (25 physics substeps of 2 ms);
  - action = 8-D `[q1..q7 targets, gripper]`, where gripper is `actuator8` ctrl: 0 = closed, 255 = open;
  - uses damped least-squares Jacobian IK with a fixed top-down orientation (fingers close along world x), plus a null-space pull towards the home posture;
  - a moving reference point advances at `max_speed` (0.25 m/s) and the IK tracks it. Clipping the error directly made the arm crawl, because the servos lag each command.
  - API: `step()` returns the applied action; `compute_action()` / `apply_action()`; `move_to_position(pos, tol, timeout, on_step)`; `wait(s, on_step)`; `open_gripper()` / `close_gripper()`; state readouts `joint_positions`, `joint_velocities`, `ee_position`, `ee_velocity`, `ee_rotation`, `gripper_opening`, and `state()` (7 joints + gripper opening).
- `uv run python -m simulation.controller` prints the joint and actuator table, then runs the tests. All reaches end within 1 mm. Grasping the red cube and lifting it 20 cm works: the fingers close to 49.5 mm and the cube is held.

### Step 6 (rest): randomization + success check ✅
- `configs/simulation.yaml` sets fps 20, the camera (front, 256×256), targets `[red_cube, blue_cube]` (green is a distractor), the randomization region (x 0.36–0.62, y ±0.30, cube spacing ≥ 9 cm, cube-to-container ≥ 15 cm) and the success settings (20 s timeout, arm 5 cm above the rim). `pyyaml` was added to the dependencies.
- `environment.TabletopEnv()` has:
  - `reset(seed)`: the robot goes home, and all 3 cubes **and the container** get random positions. `seed=None` gives the fixed default layout. The container moves by editing `model.body_pos`.
  - `check_success(target)`: target cube inside the inner walls and below the rim, gripper open, TCP clear above the rim, and no wrong object.
  - `wrong_object(target)`, `in_container(name)`, `cube_position(name)`, `container_position()`, `container_floor_z()`.
- The viewer takes a random layout: `uv run python -m simulation.environment --seed 3`.
- Cube yaw is not randomized yet, because the grasp has a fixed yaw.

### Step 10: Scripted expert ✅ (Milestone 3)
- `data/scripted_expert.py`, `ScriptedExpert(env, rng).run(target, on_step)` → `EpisodeResult`. It runs the 8 stages from PLAN.md, at a hover height 15 cm above the table.
- When `rng` is given, each episode gets small variations: speed 0.20–0.30 m/s, hover height ±2 cm, drop point ±1.5 cm.
- **Success rate: 600/600 = 100% with variations (seeds 0–99 and 1000–1499), and 100/100 without. No wrong objects.** An episode averages about 7.8 s of sim time, i.e. about 156 frames at 20 Hz, and 600 episodes take about 50 s of wall time.
- `uv run python -m data.scripted_expert --episodes 100 [--video outputs/expert.mp4] [--no-noise]`
- **Watch live:** `uv run mjpython -m data.scripted_expert --view --episodes 5`. On macOS the passive viewer needs `mjpython`, not `python`.
- Watching the video: during the grasp the arm hides the cube from `front_camera`. That is another reason to add a `wrist_camera`.

### Cameras fixed ✅
- `front_camera` is now at (1.0, 0, 1.15), aimed at (0.47, 0, 0.40) with fovy 45. A cube is about 20 px at 256×256 (was 8 px). It is built with the helper `look_at_xyaxes(pos, target)`. **+y is image right.**
- `wrist_camera` is new, on `panda/hand` at (0.06, 0, 0.065) in the hand frame, looking just past the fingertips, fovy 95. It shows both fingers, the cube while grasping and the container while carrying. It is added after attaching, so its name has no prefix.
- Both cameras are in `configs/simulation.yaml`. They map to `observation.images.front` and `observation.images.wrist`.

### Phase 4: Recording works ✅ (no full dataset recorded yet)
- **LeRobot is pinned to 0.4.4** in `pyproject.toml`.
- `data/tasks.py` has 6 training instruction templates and 3 **held-out** eval templates (for the unseen-paraphrase test). Use `make_instruction(target, rng, split)`.
- `data/collect.py` records `observation.state` (8), `action` (8), `observation.images.front` / `.wrist` (256×256 video) and `task`. The observation is captured **before** the action. Failed episodes are discarded. Each seed's variations and instruction are reproducible.
  - It writes to `data/lerobot/<name>/` (git-ignored) and refuses to overwrite an existing dataset unless `--overwrite` is given.
  - `uv run python -m data.collect --episodes 100` takes about 6 s per episode.
- Checked on a 3-episode test set (`data/lerobot/test`, 438 frames, 2.6 MB): it loads with `LeRobotDataset`; images come back as (3,256,256) floats in 0–1; gripper actions are {0, 255}; joint action_t ≈ state_t+1 (mean 0.013 rad).
- Harmless warning: `objc ... AVFFrameReceiver implemented in both` comes from Homebrew ffmpeg clashing with PyAV's bundled copy.

**Run the scene:** `uv run python main.py` (or `uv run python -m simulation.environment`)
- In the viewer, use **Simulation → Load key** to return to the home pose. Don't press Backspace: it resets every joint to zero.

---

## Open issue: git push to GitHub

- The local repo is fine: branch `main`, commit `01b4c22 feat: create basic 3d scene with mujoco(arm and cubes)`.
- The remote `origin` is `https://github.com/naturatigris/Visual_language_action_model.git`.
- First error: `repository not found`. From outside, the repo returned 404, so it is private or doesn't exist. The macOS Keychain has a GitHub login for account **`sandhya`**, not `naturatigris`. The nested Menagerie clone is **not** the cause; it is git-ignored.
- Latest error: `Invalid username or token. Password authentication is not supported`. GitHub needs a **Personal Access Token**, not a password.
- **To fix:**
  1. Logged in as `naturatigris` on github.com, make sure the repo exists. If it doesn't, create it empty, without a README.
  2. Create a token: Settings → Developer settings → Personal access tokens. A classic token needs the `repo` scope; a fine-grained one needs Contents: read/write on this repo.
  3. `git remote set-url origin https://naturatigris@github.com/naturatigris/Visual_language_action_model.git`
  4. Run `git push -u origin main` and paste the **token** when asked for a password.
  5. If it still uses the old login, remove the old Keychain entry: Keychain Access → search `github.com` → delete it, then push again.
  6. Optional: set the author for this repo only with `git config user.name "naturatigris"` and `git config user.email "<naturatigris email>"`.

---

## Next steps

1. Fix the git push (above).
2. Record the real dataset (for example 200 episodes, about 20 min), then Step 13: `data/inspect_dataset.py` to check gripper timing, labels and videos.
3. Phase 5: SmolVLA fine-tuning in the cloud (`training/`).

### Open suggestion (not yet added to PLAN.md)
Add "does it really understand?" tests to Phase 7:
- the red/blue swap test on the same scene
- a blank-instruction test
- swapping cube colours or positions
- unseen paraphrases

Also add the rule **never put object positions in `observation.state`**, so the model has to use vision.
