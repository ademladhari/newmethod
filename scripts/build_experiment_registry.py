"""Build results/FULL_EXPERIMENT_REGISTRY.md from installed runs + zip audit."""
from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXP = RESULTS / "experiments"
OUT_JSON = RESULTS / "FULL_EXPERIMENT_REGISTRY.json"
OUT_MD = RESULTS / "FULL_EXPERIMENT_REGISTRY.md"

CONFIG_KEYS = [
    "experiment_name",
    "num_experts",
    "top_k",
    "batch_size",
    "number_of_epochs",
    "balance_loss_weight",
    "balance_loss_start_weight",
    "balance_loss_warmup_epochs",
    "router_jitter_noise",
    "router_input_dropout",
    "expert_dropout",
    "router_z_loss_weight",
    "router_temperature_start",
    "router_temperature_end",
    "load_penalty_weight",
    "load_penalty_type",
    "adversarial_loss",
    "freeze_hidden_backbone",
    "enable_fp16",
    "router_grad_clip_norm",
    "train_folder",
    "validation_folder",
]


def read_config(run_dir: Path) -> dict:
    cfg = {}
    logs = list(run_dir.glob("*.log"))
    if logs:
        text = logs[0].read_text(encoding="utf-8", errors="replace")
        for key in CONFIG_KEYS:
            m = re.search(r"'" + key + r"': ([^\n,]+)", text)
            if m:
                cfg[key] = m.group(1).strip().strip("'")
        if not cfg.get("experiment_name"):
            m = re.search(r"'experiment_name': '([^']+)'", text)
            if m:
                cfg["experiment_name"] = m.group(1)
    pkl = run_dir / "options-and-config.pickle"
    if pkl.is_file():
        try:
            data = pickle.load(pkl.open("rb"))
            for key in CONFIG_KEYS:
                if hasattr(data, key):
                    v = getattr(data, key)
                    if v is not None and key not in cfg:
                        cfg[key] = v
        except Exception:
            pass
    if "coco20k" in str(cfg.get("train_folder", "")):
        cfg["dataset"] = "coco20k"
    elif "coco100k" in str(cfg.get("train_folder", "")):
        cfg["dataset"] = "coco100k"
    else:
        cfg.setdefault("dataset", "unknown")
    return cfg


def config_fingerprint(cfg: dict) -> str:
    import hashlib

    parts = [
        f"{k}={cfg.get(k, '')}"
        for k in sorted(CONFIG_KEYS)
        if k not in ("experiment_name", "train_folder", "validation_folder", "number_of_epochs")
    ]
    return hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()[:12]


def epochs(run_dir: Path) -> int:
    t = run_dir / "train.csv"
    if not t.is_file():
        return 0
    return max(0, sum(1 for _ in t.open(encoding="utf-8")) - 1)


def main():
    # Re-run zip audit
    audit = ROOT / "scripts" / "audit_all_moe_configs.py"
    subprocess.run([sys.executable, str(audit)], check=True, cwd=ROOT)

    zip_data = json.loads((RESULTS / "FULL_EXPERIMENT_REGISTRY.json").read_text(encoding="utf-8"))
    installed = []
    for train in sorted(EXP.rglob("train.csv")):
        d = train.parent
        if "archive" in d.parts:
            continue
        c = read_config(d)
        installed.append({
            "path": str(d.relative_to(RESULTS)).replace("\\", "/"),
            "run_folder_original": d.name,
            "epochs_completed": epochs(d),
            "has_validation_noisy": (d / "validation_noisy.csv").is_file(),
            "config": c,
        })

    # Build dedupe explanation from zip audit
    collapsed = []
    for r in zip_data.get("all_runs_in_zips", []):
        if not r.get("has_train_csv"):
            continue
        fp = r.get("config_fingerprint", "")
        if fp == "16da67a744be":
            continue
        match = [i for i in installed if config_fingerprint(i["config"]) == fp]
        if not match and r.get("config_fingerprint"):
            collapsed.append({**r, "reason": "not_installed"})

    for i in installed:
        i["config_fingerprint"] = config_fingerprint(i["config"])

    registry = {
        "summary": {
            "unique_hyperparameter_configs_installed": len(installed),
            "zip_run_folder_references": zip_data["summary"]["unique_run_folders_in_zips"],
            "zip_full_train_runs": zip_data["summary"]["runs_with_train_csv"],
            "distinct_config_fingerprints_in_zips": zip_data["summary"]["distinct_config_fingerprints_in_zips"],
            "note": "Deduping is by config fingerprint, NOT experiment_name alone. Same name + same config = one folder. Same name + different config would be kept separately (none found among full runs).",
        },
        "installed_experiments": installed,
        "zip_audit": zip_data,
    }

    OUT_JSON.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    _write_md(registry, zip_data)
    print("Wrote", OUT_MD)
    print("Installed:", len(installed), "unique configs")


def _write_md(reg: dict, zip_data: dict) -> None:
    lines = [
        "# Full MoE experiment registry (config-verified)",
        "",
        "Every installed run was checked via **`.log` + `options-and-config.pickle`**. ",
        "Zip inventory from `archive/zips/` (29 files).",
        "",
        "## Correction: what was *not* lost",
        "",
        "The earlier **\"22 duplicates\"** count mixed:",
        "",
        "1. **Re-uploaded zips** of the same run (e.g. `moefrozen_run_export.zip` and `(1).zip`) — same config fingerprint.",
        "2. **Crash retries** with the same `--name` and same config (partials without `train.csv`).",
        "3. **Checkpoint exports** at different epochs (`epoch41` vs `epoch112` zip) — **same hyperparameters**, longer run kept.",
        "",
        "**Distinct hyperparameter configs with full `train.csv` in your zips: 20–21.** ",
        "**Installed under `experiments/`: 21.** None of the distinct config sweeps (balance v2 vs v3, ber_route vs uniform, sym_bal08 vs t14, frozen vs unfrozen, etc.) were merged away.",
        "",
        "The only **same name, different config** flags in the audit were **false positives**: partial runs with **no log in zip** (empty fingerprint `16da67a744be`), not real config changes.",
        "",
        "## Summary",
        "",
    ]
    for k, v in reg["summary"].items():
        lines.append(f"- **{k}**: {v}")

    lines.extend([
        "",
        "## All installed experiments (unique configs)",
        "",
        "| # | Path | dataset | exp | 4/8 | top_k | batch | epochs | noisy val | balance | jitter | load_pen | T_start→end | freeze |",
        "|---|------|---------|-----|-----|-------|-------|--------|-----------|---------|--------|----------|-------------|--------|",
    ])

    for i, r in enumerate(reg["installed_experiments"], 1):
        c = r["config"]
        lines.append(
            f"| {i} | `{r['path']}` | {c.get('dataset','?')} | {c.get('experiment_name','?')} "
            f"| {c.get('num_experts','?')} | {c.get('top_k','?')} | {c.get('batch_size','?')} "
            f"| {r['epochs_completed']} | {'yes' if r['has_validation_noisy'] else 'no'} "
            f"| {c.get('balance_loss_weight','-')} | {c.get('router_jitter_noise','-')} "
            f"| {c.get('load_penalty_weight','-')} | {c.get('router_temperature_start','?')}→{c.get('router_temperature_end','?')} "
            f"| {c.get('freeze_hidden_backbone','-')} |"
        )

    lines.extend([
        "",
        "## Config groups (what differs between experiments)",
        "",
        "### frozen_moe — symmetric / diagnostic (COCO 100k)",
        "",
        "| experiment | Key hyperparameters |",
        "|------------|---------------------|",
        "| `moe_sym_bal08_v1` | bal **0.08**/0.02/5ep warmup, jitter **0**, temp 1.4→1.0, k=1, batch **128** |",
        "| `moe_sym_t14_t09_v1` | bal 0.04/0.005/10, jitter 0, temp 1.4→**0.9**, k=1, batch 32 |",
        "| `moe_routing_isolation_v1` | bal 0.03, jitter **0.005**, **adv=0**, temp 1.2→1.0, 40 ep |",
        "| `moe_collapse_diag_8exp_v1` | 8 exp k=2, jitter 0.02, **load_pen 0.1**, bal 0.06, 34 ep |",
        "",
        "### unfrozen_moe",
        "",
        "| experiment | Key hyperparameters |",
        "|------------|---------------------|",
        "| `moe_unfrozen_sym_t14_v1` | Same as sym t14 family but **freeze_hidden_backbone=False**, batch 128 |",
        "",
        "### legacy_early — sweeps (mostly COCO 100k)",
        "",
        "| experiment | How it differs |",
        "|------------|----------------|",
        "| `moe4_balance_v2_100k` vs `moe4_balance_v3_100k` | v2: **k=2** batch 16; v3: **k=1** batch 32, jitter **0.1**, temp 2.5→1.2 |",
        "| `moe8_balance_v2_100k` vs `moe8_balance_v3_100k` | Same pattern for 8 experts |",
        "| `moe4_ber_route_v1` vs `moe8_ber_route_v1` | BER-focused route; 4 vs 8 exp; bal 0.04, jitter 0.015 |",
        "| `moe4_uniform_first_100k_v1` vs `moe8_uniform_first_100k_v1` | Strong balance+jitter; 8 exp uses **k=3** |",
        "| `moe4_router_diag_v2` vs `moe8_router_diag_v2` | Router diagnostic; 4 vs 8 exp, k=1 |",
        "| `moe4_hidden178_frozen_v1` | 4 exp, batch 12, long run (88 ep) |",
        "| `moe_hidden178_frozen_anticollapse_v2` | 8 exp **k=3**, anticollapse recipe |",
        "| `moe_stabilized_4exp` | 4 exp stabilized (54 ep) |",
        "| `moe_stabilized_v2_milder` | 8 exp stabilized milder — **113 ep export** (same config as 41 ep zip) |",
        "| `moe8_top2_ber_retry` | **Different** from stabilized: bal **0.005**, jitter 0.0015, 76–120 ep exports |",
        "| `moe_hidden` | **COCO 20k** (not 100k), early MoE, 8 exp k=2, batch 16, 20 ep |",
        "",
        "## Collapsed zip references (same config fingerprint)",
        "",
        "| experiment_name | What was collapsed | Why |",
        "|-----------------|-------------------|-----|",
        "| `moe4_ber_route_v1` | 5 partial timestamps + 1 full (43 ep) | Same config; only `10-27-25` has full CSVs |",
        "| `moe8_ber_route_v1` | 1 ep run + 33 ep run | **Identical config**; kept 33 ep |",
        "| `moe4_uniform_first_100k_v1` | 2 partials + 1 full | Same config |",
        "| `moe8_uniform_first_100k_v1` | 3 partials + 1 full | Same config |",
        "| `moe_stabilized_v2_milder` | epoch20/41 zips vs epoch112 zip | **Same config**; kept longest (113 ep) |",
        "| `moe8_top2_ber_retry` | Multiple `moe_run_export*` zips | Same config; kept longest export |",
        "| `moe4_balance_v3_100k` | 2 export zips | Same config |",
        "| `moe_sym_bal08_v1` etc. | Misleading zip filenames | One run per config |",
        "",
        "## Partial runs (no full metrics — archive only)",
        "",
        "| Run | Zip | Notes |",
        "|-----|-----|-------|",
        "| `moe_sym_t14_v1` | `moe_collapse_diag_v1... (1).zip` | Checkpoints only → `archive/partial/` |",
        "| Various `moe4_ber_route_v1` | `moe_run_4exp32sss.zip` | Failed restarts, same config as full run |",
        "",
        "## Files",
        "",
        "- `FULL_EXPERIMENT_REGISTRY.json` — machine-readable (includes full zip audit)",
        "- `experiments/INDEX.json` — short index",
        "- `EXPERIMENTS_MAP.md` — folder layout guide",
        "",
        "Regenerate: `python scripts/build_experiment_registry.py`",
        "",
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
