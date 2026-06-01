from pathlib import Path
import re

root = Path("results/experiments/legacy_early")
for log in sorted(root.rglob("*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")[:80000]
    m = re.search(r"'train_folder': '([^']+)'", text)
    exp = re.search(r"'experiment_name': '([^']+)'", text)
    tf = m.group(1) if m else "?"
    ds = "coco20k" if "coco20k" in tf else ("coco100k" if "coco100k" in tf else "other")
    print(f"{ds:10} {exp.group(1) if exp else '?':40} {log.parent.relative_to(root)}")
