# Progress Log

The full project plan is in [PLAN.md](PLAN.md). This file records where things stand.

_Last updated: 2026-11-11_

---

## Done

### Planning
- `PLAN.md` is written. Main model: **SmolVLA** fine-tuned with LeRobot. Optional compact CLIP+MLP baseline in Phase 9. Demos recorded directly in LeRobot format. Actions are absolute joint targets (8-D: 7 arm joints + gripper).
- Blog title: *From Simulation to Autonomous Robot: Building and Training a Vision-Language-Action System*.
- First task: "Put the red/blue cube in the container" (the colour choice forces the model to use language).

### Environment setup
- uv project, created by the user with `uv init --package`. Python 3.12 is pinned in `.python-version`.
- Installed: `mujoco` 4.0.0, `lerobot`, `numpy`, `opencv-python`, `matplotlib` (see `pyproject.toml`).
- SmolVLA's extra packages were added with `uv add "lerobot[smolvla]==0.4.4"`, which installed `transformers` 4.57.6, `accelerate` 1.15.0, `num2words` and `safetensors`.
  - The SmolVLA **code** ships inside `lerobot` (`lerobot/policies/smolvla/`); the extra only adds what it imports.
  - The **weights** (`lerobot/smolvla_base`, about 1 GB) download from the HF Hub on first use and are cached in `~/.cache/huggingface/`.
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

### Phase 4: Dataset recorded + validated ✅ (Steps 14–15)
- Git push now works; `origin/main` is up to date.
- `configs/simulation.yaml` has `eval_seed_start: 100000`. Seeds at or above it are held out for evaluation, and `data/collect.py` refuses to record them.
- **`data/lerobot/panda_pick_place`**: 200 episodes, 31,255 frames, 173 MB. Seeds 0–199, 0 discarded, 100 red / 100 blue, all 6 train templates used. Episodes are 128–193 frames (7.8 s average). Recording took about 21 min.
- `data/inspect_dataset.py` prints a summary and 9 PASS/FAIL checks:
  - no NaNs;
  - no dropped frames;
  - timestamps spaced 1/fps;
  - gripper commands only {0, 255};
  - the gripper goes open→closed→open exactly once;
  - the fingers hold about 5 cm while closed;
  - action_t ≈ state_t+1;
  - every instruction is a train template;
  - no eval paraphrase leaked in.

  It also saves `lengths.png`, `ep<N>_trajectory.png` and `ep<N>_frames.png` to `outputs/inspect/<name>/`. **All checks pass** on the full dataset.
  - `uv run python -m data.inspect_dataset --name panda_pick_place [--episode N]`
  - Interactive viewer: `uv run lerobot-dataset-viz --repo-id sandhya056/panda_pick_place --root data/lerobot/panda_pick_place --episode-index 0`
- The "does it really understand?" tests and the rule to keep object positions out of `observation.state` are now in PLAN.md Phase 7.

### Step 16: Dataset on the HF Hub ✅
- HF username is **`sandhya056`** (the GitHub user is `naturatigris`). `REPO_OWNER` in `data/collect.py` is now `sandhya056`.
- **https://huggingface.co/datasets/sandhya056/panda_pick_place** is **private**, tagged with the codebase version `v3.0`, and holds 200 episodes / 31,255 frames.
- `data/push_dataset.py` uploads the dataset and then downloads it from the Hub into a temp folder, checking that the episode and frame counts match. Verification **PASSES**.
  - `uv run python -m data.push_dataset --name panda_pick_place [--verify-only] [--public]`
  - The CLI is already logged in (`uv run hf auth whoami` → sandhya056). Cloud machines need `HF_TOKEN` set to read the private dataset.

### Dataset pipeline: record → check → upload

The three commands below rebuild `sandhya056/panda_pick_place` from scratch. For a different dataset, change `--name` in all three.

```bash
# 1. Record 200 successful expert episodes (seeds 0–199, about 21 min) into data/lerobot/panda_pick_place/
#    Add --overwrite to replace an existing dataset; --seed sets the first layout seed (must stay < 100000)
uv run python -m data.collect --episodes 200 --name panda_pick_place

# 2. Print the summary and the 9 PASS/FAIL checks; plots go to outputs/inspect/panda_pick_place/
#    --episode N picks which episode to plot (default 0)
uv run python -m data.inspect_dataset --name panda_pick_place

# 3. Upload to the HF Hub (private) and verify by reloading it from the Hub
#    Needs `uv run hf auth login` once; --verify-only skips the upload, --public makes it public
uv run python -m data.push_dataset --name panda_pick_place
```

Quick test run: `uv run python -m data.collect --episodes 3 --name test --overwrite`, then step 2 with `--name test`. Don't push the test set.

### LeRobot upgraded to 0.7.0 ✅
- `pyproject.toml` now has `lerobot[smolvla,dataset,dataset-viz]==0.7.0`, to match the Kaggle training run (0.7.1 from git main; 0.7.0 is the latest PyPI release). In 0.7 the dataset code is behind the `dataset` extra. `numpy` is relaxed to `>=2.0`, because lerobot 0.7 needs numpy < 2.3. transformers is now 5.22.
- `data/inspect_dataset.py` still passes all 9 checks under 0.7.0.
- Harmless warning: torchcodec can't load its dylib on this Mac, so video decoding falls back to pyav.

### Phase 6, Steps 21–23: closed-loop code ✅ (waiting on the trained checkpoint)
- `policies/smolvla_policy.py`, `SmolVLAController(checkpoint, device, n_action_steps)`:
  - builds the policy the same way `lerobot-train` does, using `make_policy`, the rename map (`front` → `camera1`, `wrist` → `camera2`) and normalization stats from `panda_pick_place`, so it works for both `lerobot/smolvla_base` and the fine-tuned checkpoint;
  - `act(obs, instruction)` returns `[7 joint targets, gripper]`, with the gripper snapped to 0/255;
  - `reset()` runs every episode;
  - `will_replan` is True when the next call runs the network.
- `inference/run_robot.py` runs the closed loop on held-out seeds (≥ 100000) by default: clip to joint limits → `apply_action` → stop on success or a wrong object. It saves an optional front|wrist video and prints latency.
  - `uv run python -m inference.run_robot --checkpoint lerobot/smolvla_base --episodes 2 --max-seconds 5 --video outputs/smolvla/base_test.mp4` runs end to end (0% success, as expected for the untrained model).
  - Options: `--split eval` (held-out paraphrases), `--target red|blue`, `--instruction "..."`, `--n-action-steps N`, `--view` (with `uv run mjpython`).
- **Latency on the M4 Pro (mps):** about 290 ms per 50-action chunk, about 15 ms per step on average with `n_action_steps=25`, i.e. about 66 Hz achievable vs 20 Hz control. The first call warms up and takes about 5 s. The first load downloads about 1 GB of SmolVLM weights (about 8 min); after that it's cached.

### First trained checkpoint tested (2026-11-12)
- `sandhya056/smolvla_panda_pick_place` (private): Kaggle T4, **1,000 steps, batch 32, AMP** (about 1 epoch, the pipeline-test run).
- `main.py` is now the entry point and runs this model by default (all `inference/run_robot.py` options apply). The scene viewer moved to `uv run python -m simulation.environment`.
- **Closed loop: 0/4 success** with `n_action_steps` 25 and with 10. All 4 episodes timed out; none touched the wrong object. The arm reaches down near the cubes and sometimes moves toward the correct colour, but it bumps cubes and never grasps. Video: `outputs/smolvla/trained_1k.mp4`.
- **Offline check on dataset frames:** mean joint error vs the expert is 0.020 rad (the expert moves about 0.013 rad per step), and the gripper is right 95% of the time. So the inference pipeline is correct and the model is undertrained. Small errors add up in closed loop.
- Next: train longer, about 5k steps at batch 64 (about 10 epochs) on a faster GPU, then re-run `main.py`.

---

**Run the scene:** `uv run python main.py` (or `uv run python -m simulation.environment`)
- In the viewer, use **Simulation → Load key** to return to the home pose. Don't press Backspace: it resets every joint to zero.

---

## Next steps

1. Phase 5: SmolVLA fine-tuning in the cloud (`training/`, `training/smolvla_finetune.md`), using `--dataset.repo_id=sandhya056/panda_pick_place`.
   - Reference Colab notebook (official HF SmolVLA training): https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb
   - In the notebook, pin `lerobot[smolvla]==0.4.4` (to match local) and set `HF_TOKEN`, because the dataset is private.
   - First run is on **Kaggle**, with lerobot **0.7.1** from git main and transformers 5.22. Local is now 0.7.0 (see above).
   - Kaggle's T4 in fp32 took about 24 s per step at batch size 64 (258 steps in 1 h 45 min), too slow. Options: `--policy.use_amp=true --batch_size=32` for a short pipeline test, or rent an A100/L4 (about 1 h for 5k steps). Use `--policy.scheduler_decay_steps=<steps>` whenever `--steps` is under 30k.
2. When the checkpoint is on the Hub: `uv run python -m inference.run_robot --episodes 10 --video outputs/smolvla/run.mp4`, then Phase 7 (`inference/evaluate.py`).
   - `smolvla_base` expects the cameras `observation.images.camera1/2/3` and a 6-D state/action. Map ours with `--rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'`. The 8-D state/action is padded to 32 internally.
