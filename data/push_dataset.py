"""Upload a local LeRobotDataset to the Hugging Face Hub (Phase 4, Step 16).

The dataset is uploaded as PRIVATE unless --public is given. Log in first:
    uv run hf auth login

    uv run python -m data.push_dataset --name panda_pick_place
    uv run python -m data.push_dataset --name panda_pick_place --verify-only   # just reload from the Hub

After uploading, the script downloads the dataset back from the Hub into a temporary
folder and checks that the episode and frame counts match the local copy.
"""

import argparse
import tempfile

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from data.collect import DATASET_DIR, REPO_OWNER


def verify(repo_id: str, local: LeRobotDataset) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        remote = LeRobotDataset(repo_id, root=tmp)
        ok = (remote.meta.total_episodes == local.meta.total_episodes
              and remote.meta.total_frames == local.meta.total_frames)
        frame = remote[0]
        print(f"Hub copy: {remote.meta.total_episodes} episodes, {remote.meta.total_frames} frames; "
              f"frame 0 keys: {sorted(k for k in frame if k.startswith(('observation', 'action')))}")
        print(f"  task: {frame['task']!r}")
    print("VERIFY", "PASS" if ok else "FAIL (counts differ from local copy)")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="panda_pick_place")
    p.add_argument("--owner", default=REPO_OWNER)
    p.add_argument("--public", action="store_true", help="make the Hub dataset public (default: private)")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    repo_id = f"{args.owner}/{args.name}"
    local = LeRobotDataset(repo_id, root=DATASET_DIR / args.name)
    print(f"Local {local.root}: {local.meta.total_episodes} episodes, {local.meta.total_frames} frames")

    if not args.verify_only:
        print(f"Uploading to https://huggingface.co/datasets/{repo_id} ({'public' if args.public else 'private'}) ...")
        local.push_to_hub(private=not args.public, tags=["lerobot", "franka_panda", "mujoco", "simulation"])
        print("Upload done.")

    raise SystemExit(0 if verify(repo_id, local) else 1)


if __name__ == "__main__":
    main()
