"""Move legacy_early runs to coco20k/ when train_folder says coco20k."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "results" / "experiments" / "legacy_early"


def dataset_from_log(run_dir: Path) -> str:
    for log in run_dir.glob("*.log"):
        text = log.read_text(encoding="utf-8", errors="replace")[:100000]
        m = re.search(r"'train_folder': '([^']+)'", text)
        if m:
            tf = m.group(1)
            if "coco20k" in tf:
                return "coco20k"
            if "coco100k" in tf:
                return "coco100k"
    return "unknown"


def main():
    moved = []
    for train in list(LEGACY.rglob("train.csv")):
        run_dir = train.parent
        rel = run_dir.relative_to(LEGACY)
        if rel.parts[0] not in ("coco100k", "coco20k"):
            continue
        actual = dataset_from_log(run_dir)
        current = rel.parts[0]
        if actual == "unknown" or actual == current:
            continue
        dest = LEGACY / actual / Path(*rel.parts[1:])
        if dest.exists():
            print("SKIP exists:", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(dest))
        moved.append((str(rel), str(dest.relative_to(LEGACY))))
        print("MOVE", rel, "->", dest.relative_to(LEGACY))

    print(f"\nMoved {len(moved)} runs to correct dataset folder.")


if __name__ == "__main__":
    main()
