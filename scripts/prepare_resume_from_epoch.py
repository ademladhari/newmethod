"""
Trim checkpoints so `train_moe.py continue` resumes from a specific epoch.

`continue` always loads the LATEST checkpoint in the folder. To restart from
epoch N, delete every checkpoint with epoch > N.

Usage:
  python scripts/prepare_resume_from_epoch.py --run-folder PATH/TO/RUN --epoch 20
  python scripts/prepare_resume_from_epoch.py --run-folder PATH/TO/RUN --epoch 30 --dry-run
"""
import argparse
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Trim checkpoints after target epoch for resume.")
    p.add_argument("--run-folder", required=True, type=Path, help="Run folder containing checkpoints/")
    p.add_argument("--epoch", required=True, type=int, help="Resume from this epoch (delete later ckpts)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be deleted")
    args = p.parse_args()

    chk_dir = args.run_folder / "checkpoints"
    if not chk_dir.is_dir():
        raise SystemExit(f"No checkpoints folder: {chk_dir}")

    target = None
    to_delete = []
    for f in sorted(chk_dir.glob("*.pyt")):
        m = re.search(r"epoch-(\d+)", f.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep == args.epoch:
            target = f
        elif ep > args.epoch:
            to_delete.append(f)

    if target is None:
        raise SystemExit(
            f"No checkpoint for epoch {args.epoch} in {chk_dir}. "
            f"Saved epochs: use --epoch one of existing files."
        )

    print(f"Resume checkpoint: {target.name}")
    if not to_delete:
        print("Nothing to delete — already at target epoch.")
        return

    print(f"Will delete {len(to_delete)} newer checkpoint(s):")
    for f in to_delete:
        print(f"  {f.name}")
    if args.dry_run:
        print("(dry-run — no files removed)")
        return
    for f in to_delete:
        f.unlink()
    print("Done. Next: train_moe.py continue --folder ... --save-every 1")


if __name__ == "__main__":
    main()
