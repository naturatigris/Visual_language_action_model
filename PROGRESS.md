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
2. Add `simulation/camera.py`: a reusable `Renderer` and saving camera images (Step 7).
3. **Phase 3:** inspect joints and actuators (Step 8), write `simulation/controller.py` with Jacobian IK → joint targets plus gripper (Step 9), then the scripted pick-and-place expert (Step 10).

### Open suggestion (not yet added to PLAN.md)
Add "does it really understand?" tests to Phase 7:
- the red/blue swap test on the same scene
- a blank-instruction test
- swapping cube colours or positions
- unseen paraphrases

Also add the rule **never put object positions in `observation.state`**, so the model has to use vision.
