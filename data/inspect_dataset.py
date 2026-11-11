"""Validate a recorded LeRobotDataset and make plots for eyeballing it (Phase 4, Step 15).

Prints a summary and PASS/FAIL integrity checks, and writes figures to outputs/inspect/<name>/:
    lengths.png               histogram of episode lengths
    ep<N>_trajectory.png      joint states vs. joint targets, gripper command vs. gripper opening
    ep<N>_frames.png          front + wrist frames at 6 moments of the episode, titled with the instruction

    uv run python -m data.inspect_dataset --name test
    uv run python -m data.inspect_dataset --name panda_pick_place --episode 7

LeRobot's own interactive viewer (rerun) also works on the local dataset:
    uv run lerobot-dataset-viz --repo-id sandhya056/panda_pick_place --root data/lerobot/panda_pick_place --episode-index 0
"""

import argparse
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from data.collect import DATASET_DIR, REPO_OWNER
from data.tasks import COLOR_OF, EVAL_TEMPLATES, TRAIN_TEMPLATES
from simulation.environment import PROJECT_DIR, load_config

GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CUBE_SIZE = 0.05                      # a held cube keeps the fingers ~5 cm apart
ACTION_STATE_TOL = 0.05               # rad: mean |action_t - state_{t+1}| per episode must stay below this


def template_regex(template: str) -> re.Pattern:
    return re.compile("^" + re.escape(template).replace(re.escape("{color}"), "(\\w+)") + "$")


TRAIN_RE = [template_regex(t) for t in TRAIN_TEMPLATES]
EVAL_RE = [template_regex(t) for t in EVAL_TEMPLATES]


def task_color(task: str) -> str | None:
    for rx in TRAIN_RE + EVAL_RE:
        if m := rx.match(task):
            return m.group(1)
    return None


class Checks:
    def __init__(self):
        self.failed = 0

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        self.failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load_columns(ds: LeRobotDataset) -> dict[str, np.ndarray]:
    hf = ds.hf_dataset.with_format("numpy")
    return {k: np.stack(hf[k]) if k in ("observation.state", "action") else np.asarray(hf[k])
            for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "task_index")}


def episode_slices(cols: dict) -> dict[int, np.ndarray]:
    ep = cols["episode_index"]
    return {int(e): np.flatnonzero(ep == e) for e in np.unique(ep)}


def summarize(ds: LeRobotDataset, cols: dict, eps: dict, tasks: dict[int, str]) -> None:
    lengths = np.array([len(ix) for ix in eps.values()])
    print(f"\nDataset {ds.repo_id}  ({ds.root})")
    print(f"  episodes {len(eps)}   frames {len(cols['index']) if 'index' in cols else lengths.sum()}   fps {ds.fps}")
    print(f"  frames/episode  min {lengths.min()}  mean {lengths.mean():.1f}  max {lengths.max()}"
          f"   ({lengths.mean() / ds.fps:.1f} s avg)")
    ep_tasks = [tasks[int(cols["task_index"][ix[0]])] for ix in eps.values()]
    colors = Counter(task_color(t) for t in ep_tasks)
    print("  targets   " + "   ".join(f"{c}: {n}" for c, n in sorted(colors.items(), key=str)))
    print("  instructions (episodes):")
    for task, n in Counter(ep_tasks).most_common():
        print(f"    {n:4d}  {task}")


def run_checks(ds: LeRobotDataset, cols: dict, eps: dict, tasks: dict[int, str], config: dict) -> int:
    print("\nChecks")
    check = Checks()
    state, action = cols["observation.state"], cols["action"]
    fps = ds.fps

    check("no NaN / inf in state or action", np.isfinite(state).all() and np.isfinite(action).all())

    bad = [e for e, ix in eps.items() if not np.array_equal(cols["frame_index"][ix], np.arange(len(ix)))]
    check("frame indices contiguous from 0 (no dropped frames)", not bad, f"bad episodes {bad[:5]}" if bad else "")

    bad = [e for e, ix in eps.items() if not np.allclose(np.diff(cols["timestamp"][ix]), 1 / fps, atol=1e-4)]
    check(f"timestamps spaced 1/{fps} s", not bad, f"bad episodes {bad[:5]}" if bad else "")

    values = set(np.unique(action[:, 7]).tolist())
    check("gripper action only {0, 255}", values <= {GRIPPER_OPEN, GRIPPER_CLOSED}, f"values {sorted(values)[:6]}")

    bad_pattern, bad_grasp = [], []
    for e, ix in eps.items():
        g = action[ix, 7]
        switches = np.flatnonzero(np.diff(g) != 0)
        if not (g[0] == GRIPPER_OPEN and g[-1] == GRIPPER_OPEN and len(switches) == 2):
            bad_pattern.append(e)
            continue
        close_end = ix[switches[1]]               # last frame commanded closed
        opening = state[close_end, 7]             # fingers should be stopped on the cube, not shut
        if not (0.6 * CUBE_SIZE < opening < 1.4 * CUBE_SIZE):
            bad_grasp.append((e, round(float(opening), 4)))
    check("gripper goes open -> closed -> open exactly once", not bad_pattern,
          f"bad episodes {bad_pattern[:5]}" if bad_pattern else "")
    check("fingers hold a cube (~5 cm) while closed", not bad_grasp, f"{bad_grasp[:5]}" if bad_grasp else "")

    errs = {e: float(np.abs(action[ix[:-1], :7] - state[ix[1:], :7]).mean()) for e, ix in eps.items()}
    worst = max(errs, key=errs.get)
    check(f"joint action_t ~ state_t+1 (mean < {ACTION_STATE_TOL} rad)", errs[worst] < ACTION_STATE_TOL,
          f"overall {np.mean(list(errs.values())):.4f}, worst ep {worst}: {errs[worst]:.4f}")

    allowed = {COLOR_OF[t] for t in config["targets"]}
    used = set(np.unique(cols["task_index"]).tolist())
    bad = [tasks[i] for i in used if task_color(tasks[i]) not in allowed or not any(rx.match(tasks[i]) for rx in TRAIN_RE)]
    check(f"every instruction is a train template naming one of {sorted(allowed)}", not bad, f"{bad[:3]}" if bad else "")

    leaked = [tasks[i] for i in used if any(rx.match(tasks[i]) for rx in EVAL_RE)]
    check("no held-out eval paraphrase in the dataset", not leaked, f"{leaked[:3]}" if leaked else "")

    return check.failed


def plot_lengths(eps: dict, fps: int, out_dir) -> None:
    lengths = [len(ix) for ix in eps.values()]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths, bins=min(30, max(5, len(lengths) // 5)), color="#4C78A8")
    ax.set_xlabel(f"frames per episode ({fps} Hz)")
    ax.set_ylabel("episodes")
    ax.set_title(f"Episode lengths (n={len(lengths)})")
    fig.tight_layout()
    fig.savefig(out_dir / "lengths.png", dpi=120)
    plt.close(fig)


def plot_trajectory(cols: dict, ix: np.ndarray, episode: int, task: str, fps: int, out_dir) -> None:
    t = np.arange(len(ix)) / fps
    state, action = cols["observation.state"][ix], cols["action"][ix]
    fig, (ax_j, ax_g) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for j in range(7):
        color = f"C{j}"
        ax_j.plot(t, state[:, j], color=color, lw=1.6, label=f"joint{j + 1}")
        ax_j.plot(t, action[:, j], color=color, lw=1, ls="--")
    ax_j.set_ylabel("rad")
    ax_j.set_title(f"Episode {episode}: \"{task}\"   (solid = state, dashed = action target)")
    ax_j.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)

    ax_g.plot(t, state[:, 7] * 1000, color="C0", lw=1.6, label="gripper opening (mm)")
    ax_g.set_ylabel("opening (mm)")
    ax_cmd = ax_g.twinx()
    ax_cmd.step(t, action[:, 7], where="post", color="C3", lw=1, ls="--", label="command (0 closed, 255 open)")
    ax_cmd.set_ylabel("command")
    ax_g.set_xlabel("time (s)")
    lines = ax_g.get_lines() + ax_cmd.get_lines()
    ax_g.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"ep{episode}_trajectory.png", dpi=120)
    plt.close(fig)


def plot_frames(ds: LeRobotDataset, ix: np.ndarray, episode: int, task: str, out_dir, n: int = 6) -> None:
    cams = [k for k in ds.meta.camera_keys]
    picks = np.linspace(0, len(ix) - 1, n).round().astype(int)
    fig, axes = plt.subplots(len(cams), n, figsize=(2.2 * n, 2.3 * len(cams)), squeeze=False)
    for col, k in enumerate(picks):
        item = ds[int(ix[k])]
        for row, cam in enumerate(cams):
            ax = axes[row, col]
            ax.imshow(item[cam].permute(1, 2, 0).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={k / ds.fps:.1f}s  grip={int(item['action'][7])}", fontsize=8)
            if col == 0:
                ax.set_ylabel(cam.split(".")[-1])
    fig.suptitle(f"Episode {episode}: \"{task}\"")
    fig.tight_layout()
    fig.savefig(out_dir / f"ep{episode}_frames.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and visualize a recorded LeRobotDataset.")
    parser.add_argument("--name", default="panda_pick_place", help="dataset folder under data/lerobot/")
    parser.add_argument("--episode", type=int, default=0, help="episode to plot")
    args = parser.parse_args()

    ds = LeRobotDataset(f"{REPO_OWNER}/{args.name}", root=DATASET_DIR / args.name)
    config = load_config()
    if ds.fps != config["fps"]:
        print(f"WARNING: dataset fps {ds.fps} != config fps {config['fps']}")
    tasks = {int(i): t for t, i in ds.meta.tasks["task_index"].items()}
    cols = load_columns(ds)
    eps = episode_slices(cols)

    summarize(ds, cols, eps, tasks)
    failed = run_checks(ds, cols, eps, tasks, config)

    out_dir = PROJECT_DIR / "outputs" / "inspect" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ix = eps[args.episode]
    task = tasks[int(cols["task_index"][ix[0]])]
    plot_lengths(eps, ds.fps, out_dir)
    plot_trajectory(cols, ix, args.episode, task, ds.fps, out_dir)
    plot_frames(ds, ix, args.episode, task, out_dir)
    print(f"\nFigures in {out_dir}")
    print("All checks passed." if not failed else f"{failed} check(s) FAILED.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
