# From Simulation to Autonomous Robot: Building and Training a Vision-Language-Action System

*An end-to-end exploration of robot simulation, demonstration learning, VLA fine-tuning and language-guided manipulation using MuJoCo and LeRobot.*

This is the project plan behind the blog post.

---

## Goal

Build a simulated Franka Panda robot in MuJoCo. It should read a natural-language instruction, look at the scene through a camera, and carry out the task autonomously. The robot is driven by a **fine-tuned SmolVLA** policy, trained on demonstrations recorded in **LeRobot** format.

**Example task:**

> Put the red cube in the container.

Loop: receive instruction → capture image → VLA predicts an action chunk → execute in MuJoCo → observe again → repeat until done.

The goal is to understand, and then explain, the full path from **visual observations and language to robot actions**: simulation, demonstrations, dataset, fine-tuning, closed-loop control and evaluation.

---

## Decisions

| Topic | Decision |
|---|---|
| Main model | **SmolVLA** (`lerobot/smolvla_base`), fine-tuned with LeRobot |
| Compact model | **Optional** baseline for the blog's "Do we need a big VLA?" section (Phase 9) |
| Dataset format | **LeRobot format from the start**: record straight into `LeRobotDataset`, with no conversion step |
| Action representation | **Absolute joint-position targets**: 7 arm joints + 1 gripper = 8-D. This matches SmolVLA's usual setup and the Panda's position actuators. Confirm against the SmolVLA config before recording (Step 12) |
| First task scope | Pick-and-place with **both** cubes: "Put the red/blue cube in the container". The colour forces the model to use the language |
| Project root | This folder (`Visual_language_action_model/`) |
| Hardware | Apple M4 Pro, 24 GB RAM, no NVIDIA GPU |
| Compute | Simulation, data collection and inference run locally (PyTorch `mps`). **SmolVLA fine-tuning runs on a cloud GPU** (Colab / Kaggle / rented) |
| Python | Pin 3.12 with `uv python pin 3.12`; the system default is 3.9, which is too old for LeRobot |
| Blog | Capture screenshots, GIFs and metrics **during** each phase (see the "Blog capture" lists) |

---

## Technology Stack

| Component | Technology | Notes |
|---|---|---|
| Physics simulation | MuJoCo | Source of truth for all state |
| Robot model | Franka Emika Panda (MuJoCo Menagerie) | 7-DoF arm + parallel gripper |
| Language | Python (`uv`) | |
| Robot learning framework | Hugging Face LeRobot | Dataset format, training, policies |
| VLA | SmolVLA (~450M params) | SmolVLM backbone + flow-matching action expert; predicts action chunks |
| Deep learning | PyTorch | `mps` locally, CUDA in the cloud |
| Dataset hosting | Hugging Face Hub (private repo) | Moves the dataset to the cloud GPU and the checkpoint back |
| Visualization | MuJoCo Viewer | |
| Frontend | Three.js (Vite) | Visualization only |
| Communication | WebSocket | Python → browser |

---

## Project Structure

```text
Visual_language_action_model/
├── simulation/
│   ├── environment.py        # tabletop scene (MJCF), reset + randomization, success check
│   ├── robot.py              # load / inspect the Panda
│   ├── camera.py             # RGB capture (one reused Renderer per camera)
│   ├── controller.py         # IK → joint targets, gripper
│   └── stream.py             # WebSocket state streaming (Phase 8)
├── assets/
│   └── mujoco_menagerie/     # git clone
├── data/
│   ├── scripted_expert.py    # scripted pick-and-place demonstrator
│   ├── collect.py            # records episodes into LeRobotDataset
│   ├── inspect_dataset.py    # visualization and validation
│   └── lerobot/              # local LeRobot dataset root
├── policies/
│   ├── smolvla_policy.py     # loads fine-tuned SmolVLA, wraps preprocessing
│   └── compact/              # OPTIONAL baseline (Phase 9)
│       ├── model.py
│       └── train.py
├── training/
│   ├── smolvla_finetune.md   # exact cloud commands and config used
│   └── colab_finetune.ipynb  # cloud fine-tuning notebook
├── inference/
│   ├── run_robot.py          # closed-loop VLA control in MuJoCo
│   └── evaluate.py           # batch evaluation, metrics, videos
├── frontend/
│   └── threejs-dashboard/
├── configs/
│   └── simulation.yaml       # randomization ranges, fps, camera settings
├── experiments/              # metrics, videos, checkpoint references
├── blog/
│   ├── assets/               # screenshots, GIFs, plots
│   └── draft.md
├── main.py
├── pyproject.toml
├── PLAN.md
└── README.md
```

---

## Conventions to Fix Early

Collection, fine-tuning and inference must all use the same conventions.

| Item | Convention |
|---|---|
| `action` | 8-D float: target positions for the 7 arm joints (rad) + gripper command (open/closed mapped to the actuator range). This is the vector written to `data.ctrl` |
| `observation.state` | 8-D float: 7 arm joint positions + gripper opening |
| `observation.images.front` | Fixed front camera, RGB, e.g. 256×256 (SmolVLA resizes internally) |
| `observation.images.wrist` | *Optional* wrist camera. A second view often improves grasping a lot |
| `task` | The instruction string, stored per frame (LeRobot uses it as the language input) |
| fps | 10–30 Hz policy rate (pick one, e.g. 20). Physics substeps run underneath |
| Normalization | Handled by LeRobot from dataset statistics. Don't normalize by hand |
| Success | Target cube's centre is inside the container bounds, the gripper is open and the arm is clear, within N seconds |
| Wrong-object | The non-target cube is lifted or moved into the container |

---

## macOS / MuJoCo / LeRobot Gotchas

- `mujoco.viewer.launch(...)` (blocking) works with plain `python`. `launch_passive(...)` needs **`mjpython`** on macOS: `uv run mjpython script.py`.
- Create each `mujoco.Renderer` **once** and reuse it. Don't create one per frame.
- The Menagerie Panda uses position actuators: 7 arm actuators + a gripper tendon actuator (`actuator8`, ctrl 0–255). Verify this against the XML and `model.actuator_ctrlrange`.
- LeRobot's API and CLI change often. **Pin the LeRobot version** in `pyproject.toml` and record it in the blog. The dataset-creation and training commands below are conceptual; check them against the docs for the pinned version.
- Some ops on `mps` may need `PYTORCH_ENABLE_MPS_FALLBACK=1`.

---

## Phase 1: Project Initialization

**Step 1. Initialize**

```bash
uv init
uv python pin 3.12
uv add mujoco numpy opencv-python matplotlib
uv add lerobot          # pin the version; may need extras for smolvla, e.g. lerobot[smolvla]
```

**Step 2. Create folders as each phase needs them.** The structure above is the target layout; there's no need to create empty folders up front.

**Step 3. Verify MuJoCo.** Open a plane with a light in the viewer from `main.py`.

- [ ] **Milestone 1:** MuJoCo works and the viewer opens.

> **Blog capture:** setup commands, first viewer screenshot.

---

## Phase 2: Robot and Simulation Environment

**Step 4. Get the Panda.**

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
```

Relevant files in `franka_emika_panda/`: `panda.xml`, `scene.xml`, `assets/`.

**Step 5. Load and inspect** `scene.xml`. Learn these concepts: `MjModel`, `MjData`, bodies, joints, actuators, sensors, geoms, cameras.

**Step 6. Build the tabletop scene** in `simulation/environment.py`: include `panda.xml`, then add:

- a table
- a red cube
- a blue cube
- a container
- lights
- `front_camera` (plus an optional `wrist_camera` attached to the hand)

Add `reset(seed)`, which randomizes the cube and container positions within ranges from `configs/simulation.yaml`. Add `check_success(target)`.

**Step 7. Camera capture** in `simulation/camera.py`. Save images and check that both cubes and the container are clearly visible.

- [ ] **Milestone 2:** The scene loads, randomizes and produces camera images.

> **Blog capture:** scene screenshots, camera views, a diagram of the scene tree.

---

## Phase 3: Control and the Scripted Expert

**Step 8. Inspect** joints, actuators, limits, ctrl ranges and the EE site. Read joint positions and velocities, the EE pose and the gripper opening.

**Step 9. `RobotController`** in `simulation/controller.py`:

- `move_to_position(pos)`: differential (Jacobian) IK with a fixed top-down orientation produces joint targets
- `open_gripper()` / `close_gripper()`
- `step()`

The controller outputs **joint targets**, which is the action format.

**Step 10. Scripted pick-and-place** in `data/scripted_expert.py`:

1. Move above the target cube.
2. Lower.
3. Close the gripper.
4. Lift.
5. Move above the container.
6. Lower.
7. Open the gripper.
8. Retreat.

Measure the expert's success rate over many random seeds; it should be close to 100%.

- [ ] **Milestone 3:** The scripted expert reliably places either cube.

> **Blog capture:** GIF of the scripted expert, the expert's success rate, the IK explanation.

---

## Phase 4: Record Demonstrations in LeRobot Format

**Step 11. Learn the LeRobot dataset format.** It consists of Parquet files for state, action and task; MP4 videos for camera streams; and metadata with fps, features and stats. Look at an existing SmolVLA fine-tuning dataset on the Hub to see what it expects.

**Step 12. Confirm SmolVLA's input and output expectations** *before recording*:

- camera keys and resolution
- state and action dimensions (SmolVLA pads to a maximum dimension)
- chunk size
- whether actions are absolute joint targets

Adjust the conventions table if anything differs.

**Step 13. Recorder** in `data/collect.py`. Conceptually:

```python
dataset = LeRobotDataset.create(
    repo_id="<user>/panda-pick-place-sim",
    fps=FPS,
    robot_type="franka_panda_sim",
    features={
        "observation.images.front": {...},
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "action": {"dtype": "float32", "shape": (8,)},
    },
)

for episode in range(N):
    env.reset(seed=episode)
    target, instruction = sample_task()
    for step in scripted_expert(env, target):
        dataset.add_frame({
            "observation.images.front": img,
            "observation.state": state,
            "action": ctrl,          # the joint targets sent this step
            "task": instruction,
        })
    if env.check_success(target):
        dataset.save_episode()
    else:
        dataset.clear_episode_buffer()   # keep only successful demos
```

**Step 14. Generate varied demos.**

- Instructions with paraphrases, e.g.:
  - "Put the red cube in the container"
  - "Place the blue cube into the box"
  - "Move the red block into the bin"
- Randomize:
  - cube positions
  - container position
  - initial arm pose
  - which cube is the target
  - (optionally) lighting and colours slightly
- Start with ~50 episodes, then go to 100–200 if evaluation is weak.

> The scripted expert supplies trajectories but no language understanding. SmolVLA has to learn to link the instruction, the scene and the actions.

**Step 15. Validate the dataset** with `data/inspect_dataset.py` and LeRobot's dataset visualizer:

- Watch episode videos and plot state and action trajectories.
- Check that the gripper timing and instruction labels are right.
- Confirm there are no dropped frames.
- Hold out a **test set of seeds and paraphrases** that is never used for training.

**Step 16. Push to the Hugging Face Hub** as a **private** dataset so the cloud GPU can access it.

- [ ] **Milestone 4:** A validated LeRobot dataset on the Hub.

> **Blog capture:** dataset structure, sample frames, trajectory plots, visualizer screenshot.

---

## Phase 5: Fine-tune SmolVLA (Cloud GPU)

**Step 17. Understand SmolVLA.**

- A compact vision-language backbone (SmolVLM) encodes the camera images, the instruction and the robot state.
- A flow-matching "action expert" generates a **chunk** of future actions.
- It is pretrained on community robot datasets and designed to run on modest hardware.

**Step 18. Set up the cloud environment.** In Colab, Kaggle or a rented GPU, install the same pinned LeRobot version and log in to the HF Hub.

**Step 19. Fine-tune.** Conceptually (check the CLI for the pinned version):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<user>/panda-pick-place-sim \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/smolvla_panda \
  --policy.device=cuda
```

Record the exact command, GPU type, training time and cost in `training/smolvla_finetune.md`.

**Step 20. Track training** with W&B or TensorBoard: training loss curve, learning rate and checkpoints. Push the final checkpoint to the Hub (private).

- [ ] **Milestone 5:** A fine-tuned SmolVLA checkpoint.

> **Blog capture:** architecture diagram, config, loss curves, training time and cost.

---

## Phase 6: Close the Loop in MuJoCo

**Step 21. `policies/smolvla_policy.py`**

- Load the checkpoint locally (`mps`, or CPU as a fallback).
- Build the observation dict with the **same keys, image size and state layout** as the dataset.
- Return the next action or action chunk.

**Step 22. `inference/run_robot.py`**

```python
obs = env.reset(seed)
while not done and t < max_steps:
    action = policy.select_action(obs, task=instruction)   # pops from the predicted chunk
    action = clip_to_joint_limits(action)
    env.apply(action)                                      # write to data.ctrl, step physics
    obs = env.observe()
    done = env.check_success(target)
```

The VLA decides and the controller only executes. **There is no scripted fallback during evaluation.**

**Step 23. Measure inference latency** on the M4 Pro: time per chunk and the effective control rate. If it is too slow, execute more steps per chunk.

- [ ] **Milestone 6:** SmolVLA autonomously completes pick-and-place in MuJoCo.

> **Blog capture:** first successful autonomous GIF (and the funniest failure), latency numbers.

---

## Phase 7: Evaluation

**Step 24. `inference/evaluate.py`** runs N episodes per condition on **held-out** seeds and saves metrics and videos to `experiments/`.

| Condition | What it tests |
|---|---|
| Red / blue target | Language grounding |
| Unseen cube positions | Spatial generalization |
| Unseen container positions | Target generalization |
| Unseen paraphrases | Language generalization |
| Repeated seeds | Consistency |

**"Does it really understand?" tests.** These catch a policy that ignores the instruction or uses shortcuts:

| Test | Pass means |
|---|---|
| Red/blue swap on the **same** scene (same seed, only the colour word changes) | It picks a different cube, so the language is really used |
| Blank or nonsense instruction | It hesitates or behaves arbitrarily, rather than confidently doing one fixed thing |
| Swap the cube colours or positions between runs | It follows the colour, not a memorized location |
| Unseen paraphrases (`EVAL_TEMPLATES` in `data/tasks.py`) | The wording generalizes |

Held-out layouts use seeds `>= eval_seed_start` from `configs/simulation.yaml`; `data/collect.py` refuses to record them.

> **Rule:** never put object positions in `observation.state`. It stays as 7 joints + gripper, so the model has to find the cubes visually.

| Metric | Purpose |
|---|---|
| Task success rate | % completed |
| Grasp success rate | % successful grasps |
| Wrong-object rate | Picked the wrong colour |
| Placement accuracy | Cube inside the container |
| Completion time | Seconds / steps |
| Inference latency | ms per chunk |

**Step 25. Failure analysis.** Group the failures (missed grasp, wrong cube, drop, timeout, collision) and show examples. Compare against the scripted expert, which is the upper bound.

**Step 26. (Optional) Data scaling.** Fine-tune on 25 / 50 / 100 / 200 episodes and plot success rate against dataset size.

- [ ] **Milestone 7:** A documented evaluation with metrics, videos and failure analysis.

> **Blog capture:** results tables, success-vs-data plot, failure gallery.

---

## Phase 8: Three.js Dashboard

Start this phase only after the closed loop works. Three.js **only visualizes** MuJoCo state.

**Step 27.** Create the frontend:

```bash
npm create vite@latest frontend/threejs-dashboard -- --template vanilla
cd frontend/threejs-dashboard && npm install && npm install three
```

**Step 28.** Build the 3D scene: Panda meshes (from Menagerie) driven by the streamed joint positions, plus the table, cubes, container, orbit controls and lights.

**Step 29.** Write `simulation/stream.py`, a WebSocket server that sends messages like this:

```json
{
  "joint_positions": [],
  "cube_positions": {"red": [], "blue": []},
  "container_position": [],
  "gripper_state": "open",
  "instruction": "Put the red cube in the container",
  "predicted_action_chunk": [[]],
  "task_status": "running",
  "camera_image": "<base64 jpeg>"
}
```

**Step 30.** Dashboard features:

- 3D view
- instruction input
- start and reset controls
- live camera frame
- predicted actions
- joint states
- gripper status
- task status
- running success metrics

- [ ] **Milestone 8:** Type an instruction, watch SmolVLA drive the robot live in the browser.

> **Blog capture:** dashboard screenshots and screen recording.

---

## Phase 9 (Optional): "Do We Need a Big VLA?" Compact Baseline

Train a small model on the **same LeRobot dataset** and compare it with SmolVLA.

**Step 31. Compact model** in `policies/compact/`:

```text
front image ─▶ frozen CLIP image encoder ─┐
instruction ─▶ frozen CLIP text encoder ──┼─▶ concat + MLP ─▶ 8-D action
robot state ─▶ small MLP ─────────────────┘
```

- Loss: MSE on the joint targets + BCE on the gripper.
- Trains locally on `mps` in minutes.

**Step 32. Evaluate** it with the same `evaluate.py` conditions.

**Step 33. Compare:**

| | Compact model | SmolVLA |
|---|---|---|
| Parameters trained | | |
| Training time / hardware | | |
| Inference latency | | |
| Task success | | |
| Wrong-object rate | | |
| Unseen paraphrases | | |

Either result is a good story. A big gap shows why pretraining matters. A small gap is an honest, surprising finding about simple tasks.

- [ ] **Milestone 9 (optional):** A comparison table and discussion.

---

## Phase 10: Write the Blog

**Step 34. Assemble** `blog/draft.md` from the captured material, following the outline below.

**Step 35. Document** hardware, LeRobot version, dataset size, training time and cost, and limitations: simulation only, a single task family, a scripted expert, no sim-to-real.

**Step 36. Publish the code** (README with reproduction steps) and optionally the dataset and checkpoint.

### Blog outline

1. **Why VLAs?** What the project builds and the final demo GIF up front
2. **Building the simulation:** MuJoCo, the Panda, the tabletop scene, cameras
3. **Teaching by demonstration:** the scripted expert with IK, and recording a LeRobot dataset
4. **Fine-tuning SmolVLA:** architecture, setup, config, cloud training
5. **Closing the loop:** the VLA controlling the robot in MuJoCo
6. **Evaluation:** what worked, what failed and why
7. **(Optional) Do we need a big VLA?** The compact-model comparison
8. **Watching it live:** the Three.js dashboard
9. **Lessons learned and limitations**

---

## Recommended Development Order

| # | Task | Done when |
|---|---|---|
| 1 | Initialize Python and MuJoCo | Viewer opens |
| 2 | Import the Panda | Robot loads |
| 3 | Tabletop environment | Objects, randomization and camera work |
| 4 | Controller and scripted expert | Expert succeeds ~100% |
| 5 | Confirm SmolVLA input/output format | Conventions locked |
| 6 | Record the LeRobot dataset | Dataset validated and on the Hub |
| 7 | Fine-tune SmolVLA (cloud) | Checkpoint trained |
| 8 | Closed loop in MuJoCo | VLA completes the task |
| 9 | Evaluate | Metrics and failure analysis done |
| 10 | Three.js dashboard | Live visualization works |
| 11 | (Optional) compact baseline | Comparison table |
| 12 | Write the blog | Published |

## Important Technical Principles

1. **Get the simulation and expert solid first.** Bad demos mean a bad policy.
2. **Lock the data conventions before recording.** Re-recording is expensive, so match SmolVLA's expectations from day one.
3. **Scripted demos + fine-tuning is imitation learning, not RL.**
4. **Keep the model and the controller separate.** The VLA predicts joint targets and the simulator executes them.
5. **MuJoCo is the source of truth.** Three.js only renders.
6. **Keep it small:** one robot, two cubes, one container and a handful of instructions.
7. **Capture blog material as you go.**

---

## Open Questions (defaults chosen, revisit later)

| Question | Default | Alternative |
|---|---|---|
| Wrist camera | Add it (likely improves grasps) | Front camera only |
| Action space | Absolute joint targets (8-D) | EE pose / Cartesian deltas plus an adapter |
| Number of demos | 50 → 200 as needed | Data-scaling study |
| Cloud provider | Colab (free/Pro) | Kaggle, Lambda, RunPod |
| Inference location | Local on M4 Pro (`mps`) | Remote GPU server |
| Three.js robot geometry | Menagerie meshes + streamed `qpos` | MuJoCo-WASM viewer |
| Dashboard framework | Vanilla Vite | React |
| Publish dataset/checkpoint | Decide before publishing the blog | Keep private |

---

## Resources

| Resource | Purpose |
|---|---|
| https://mujoco.readthedocs.io/ | MuJoCo docs |
| https://github.com/google-deepmind/mujoco_menagerie | Robot models |
| https://github.com/huggingface/lerobot | LeRobot framework |
| https://huggingface.co/docs/lerobot | Dataset and policy docs |
| https://huggingface.co/docs/lerobot/smolvla | SmolVLA fine-tuning docs |
| https://huggingface.co/blog/smolvla | SmolVLA introduction |
| https://huggingface.co/lerobot/smolvla_base | Base checkpoint |
| https://threejs.org/docs/ | Three.js docs |
