"""
Reorganize results/ into experiments/frozen_moe/... with readable names.
Run from repo root: python scripts/reorganize_results.py
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# source run dir (must contain train.csv) -> dest relative to RESULTS
MOVES: list[tuple[Path, str]] = [
    (
        RESULTS / "runs" / "moe_sym_bal08_v1 2026.06.01--12-11-36",
        "experiments/frozen_moe/coco100k/4exp_k1/batch128/"
        "sym_bal08_jitter0_bal08warm5_temp14to10_ep20_colab_2026-06-01",
    ),
    (
        RESULTS / "moe8" / "newmethod" / "hidden_frozen178_moe" / "runs"
        / "moe_sym_t14_t09_v1 2026.06.01--11-01-05",
        "experiments/frozen_moe/coco100k/4exp_k1/batch32/"
        "sym_t14_temp09_bal004warm10_ep20_kaggle_2026-06-01",
    ),
    (
        RESULTS / "newmethod" / "hidden_frozen178_moe" / "runs"
        / "moe_routing_isolation_v1 2026.05.31--13-16-56",
        "experiments/frozen_moe/coco100k/4exp_k1/batch32/"
        "routing_isolation_jitter005_adv0_ep40_kaggle_2026-05-31",
    ),
    (
        RESULTS / "newmethod" / "hidden_frozen178_moe" / "runs"
        / "moe_collapse_diag_8exp_v1 2026.05.31--13-16-41",
        "experiments/frozen_moe/coco100k/8exp_k2/batch32/"
        "collapse_diag_jitter02_loadpen01_bal06_ep34_kaggle_2026-05-31",
    ),
]

DUPLICATE_SOURCES = [
    (
        RESULTS / "moecollapsev1" / "newmethod" / "hidden_frozen178_moe" / "runs"
        / "moe_routing_isolation_v1 2026.05.31--13-16-56",
        "duplicates/routing_isolation_v1_copy_moecollapsev1",
    ),
    (
        RESULTS / "moe_collapse_diag_v1 2026.05.31--08-55-29" / "newmthod1"
        / "hidden_frozen178_moe" / "runs"
        / "moe_routing_isolation_v1 2026.05.31--13-16-56",
        "duplicates/routing_isolation_v1_copy_moe_collapse_diag_v1",
    ),
]

PARTIAL = [
    (
        RESULTS / "moe_collapse_diag_v1 2026.05.31--08-55-29 (1)" / "newmethod"
        / "hidden_frozen178_moe" / "runs" / "moe_sym_t14_v1 2026.06.01--12-23-13",
        "partial/moe_sym_t14_v1_checkpoints_only_kaggle_2026-06-01",
    ),
]

LOOSE_FILES = [
    (RESULTS / "train.txt", "logs/moe_sym_bal08_v1_colab_console.txt"),
    (RESULTS / "runs_comparison_bitwise_error.png", "assets/comparisons/runs_comparison_bitwise_error.png"),
    (RESULTS / "runs_comparison_loss.png", "assets/comparisons/runs_comparison_loss.png"),
    (RESULTS / "moecollapse" / "runs_comparison_bitwise_error.png", "assets/comparisons/moecollapse_runs_comparison_bitwise_error.png"),
    (RESULTS / "train.csv", "archive/loose/root_train.csv"),
    (RESULTS / "validation.csv", "archive/loose/root_validation.csv"),
    (RESULTS / "validation copy.csv", "archive/loose/root_validation_copy.csv"),
]

LOOSE_DIRS = [
    (RESULTS / "moecollapse", "archive/loose/moecollapse_csv_exports"),
    (RESULTS / "newmethod", "archive/loose/newmethod_empty_after_moves"),
    (RESULTS / "moe8", "archive/loose/moe8_empty_after_moves"),
    (RESULTS / "runs", "archive/loose/runs_empty_after_moves"),
    (RESULTS / "moecollapsev1", "archive/loose/moecollapsev1_empty_after_moves"),
    (RESULTS / "moe_collapse_diag_v1 2026.05.31--08-55-29", "archive/loose/moe_collapse_diag_v1_empty_after_moves"),
    (RESULTS / "moe_collapse_diag_v1 2026.05.31--08-55-29 (1)", "archive/loose/moe_collapse_diag_v1_(1)_after_partial_move"),
]


def _read_config_from_log(run_dir: Path) -> dict:
    logs = list(run_dir.glob("*.log"))
    if not logs:
        return {}
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    cfg = {}
    for key in (
        "num_experts",
        "top_k",
        "router_jitter_noise",
        "balance_loss_weight",
        "balance_loss_start_weight",
        "balance_loss_warmup_epochs",
        "load_penalty_weight",
        "adversarial_loss",
        "freeze_hidden_backbone",
    ):
        m = re.search(r"'" + key + r"': ([^\n,]+)", text)
        if m:
            cfg[key] = m.group(1).strip()
    m = re.search(r"'batch_size': (\d+)", text)
    if m:
        cfg["batch_size"] = int(m.group(1))
    m = re.search(r"'experiment_name': '([^']+)'", text)
    if m:
        cfg["experiment_name"] = m.group(1)
    m = re.search(r"'train_folder': '([^']+)'", text)
    if m:
        cfg["train_folder"] = m.group(1)
    if "coco100k" in cfg.get("train_folder", ""):
        cfg["dataset"] = "coco100k"
    return cfg


def _epoch_count(run_dir: Path) -> int | None:
    train_csv = run_dir / "train.csv"
    if not train_csv.is_file():
        return None
    return sum(1 for _ in open(train_csv, encoding="utf-8")) - 1


def safe_move(src: Path, dest_rel: str) -> None:
    if not src.exists():
        print("SKIP missing:", src)
        return
    dest = RESULTS / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("SKIP dest exists:", dest)
        return
    print("MOVE", src.relative_to(RESULTS), "->", dest_rel)
    shutil.move(str(src), str(dest))


def write_manifest():
    manifest = []
    for dest_rel, _ in [(m[1], m[0]) for m in MOVES]:
        dest = RESULTS / dest_rel
        if not dest.is_dir():
            continue
        meta = _read_config_from_log(dest)
        meta["path"] = dest_rel
        meta["epochs_completed"] = _epoch_count(dest)
        manifest.append(meta)
    idx_path = RESULTS / "experiments" / "INDEX.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    if not RESULTS.is_dir():
        raise SystemExit(f"Missing {RESULTS}")

    for src, dest_rel in MOVES:
        safe_move(src, dest_rel)

    for src, dest_rel in DUPLICATE_SOURCES:
        safe_move(src, f"archive/{dest_rel}")

    for src, dest_rel in PARTIAL:
        safe_move(src, f"archive/{dest_rel}")

    for src, dest_rel in LOOSE_FILES:
        safe_move(src, dest_rel)

    for src, dest_rel in LOOSE_DIRS:
        if src.exists() and any(src.iterdir()):
            safe_move(src, dest_rel)
        elif src.exists():
            try:
                src.rmdir()
            except OSError:
                pass

    readme = RESULTS / "README.md"
    readme.write_text(
        """# Results layout

## Active experiments

```
experiments/frozen_moe/coco100k/
  4exp_k1/batch128/   # 4 experts, top-k=1, batch 128
  4exp_k1/batch32/    # 4 experts, top-k=1, batch 32
  8exp_k2/batch32/    # 8 experts, top-k=2, batch 32
```

Each run folder name encodes: recipe + platform + date.
Inside: `train.csv`, `validation.csv`, `validation_noisy.csv`, `checkpoints/`, `images/`, `.log`, `options-and-config.pickle`.

See `experiments/INDEX.json` for parsed configs.

## Archive

- `archive/duplicates/` — duplicate uploads of the same Kaggle run
- `archive/partial/` — incomplete runs (checkpoints only)
- `archive/loose/` — old zip folders (`newmethod/`, `moe8/`) and stray CSVs

## Assets

- `assets/comparisons/` — comparison plots
- `logs/` — long console captures (e.g. Colab training log)

## Naming key

| Token | Meaning |
|-------|---------|
| `sym_bal08` | balance 0.08, start 0.02, warmup 5 |
| `sym_t14` | symmetric router, temp 1.4→1.0 |
| `sym_t14_temp09` | temp end 0.9 |
| `routing_isolation` | jitter 0.005, adv 0, ep 40 |
| `collapse_diag` | old 8-exp diagnostic (jitter, load penalty) |
| `colab` / `kaggle` | where it was trained |

**Note:** All current full runs use **COCO 100k** (`coco100k/train`). No verified coco20k run in archive; `moecollapse_csv_exports` is legacy 8-exp train-only export.

Reorganized: """
        + datetime.now().strftime("%Y-%m-%d %H:%M")
        + "\n",
        encoding="utf-8",
    )
    write_manifest()
    print("Done. See results/README.md and results/experiments/INDEX.json")


if __name__ == "__main__":
    main()
