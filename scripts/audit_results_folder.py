#!/usr/bin/env python3
"""Audit results/ — list all MoE runs with config + ep20 metrics + verdict."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT_MD = ROOT / "RESULTS_AUDIT.md"
OUT_CSV = ROOT / "RESULTS_AUDIT.csv"


def g(row, *keys):
    for k in keys:
        for c in (k, k + " "):
            if c in row and row[c] not in ("", None):
                return float(row[c])
    return None


def metrics_at_epoch(val_path: Path, ep: int | None = 20):
    if not val_path.exists():
        return None
    with val_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if ep is not None:
        match = [r for r in rows if int(r["epoch"]) == ep]
        r = match[0] if match else rows[-1]
    else:
        r = rows[-1]
    ep = int(r["epoch"])
    ber = g(r, "bitwise-error", "val_ber")
    return {
        "ep": ep,
        "clean_ber_pct": round(ber * 100, 3) if ber is not None else None,
        "max_use": g(r, "expert_max_use"),
        "load_l1": g(r, "train_val_load_l1"),
    }


def noisy_at_epoch(run_dir: Path, ep: int):
    p = run_dir / "validation_noisy.csv"
    if not p.exists():
        return None
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    match = [r for r in rows if int(r["epoch"]) == ep]
    if not match:
        return None
    ber = g(match[0], "bitwise-error")
    return round(ber * 100, 3) if ber is not None else None


def infer_config(rel: str, name: str):
    frozen = "frozen_moe" in rel or "sym_bal08" in rel
    if "unfrozen" in rel or "unfrozen" in name:
        frozen = False
    experts = 8 if ("8exp" in rel or "8exp" in name or name.startswith("moe8")) else 4
    if "4exp" in name:
        experts = 4
    top_k = 1
    if "top2" in name or "/8exp_k2/" in rel or ("k2" in rel and "8exp" in rel):
        top_k = 2
    if "8exp_k3" in rel:
        top_k = 3
    if "4exp_sparse_b128" in name and "8exp" not in name and "top2" not in name:
        top_k = 4  # dense misnamed
    if "top2" in name:
        top_k = 2
    batch = 128 if "batch128" in rel else 32 if "batch32" in rel else 16 if "batch16" in rel else 12 if "batch12" in rel else "?"
    dataset = "coco20k" if "coco20k" in rel else "coco100k"
    if "300k" in name or "200k" in name:
        dataset = "~200-300k"
    attack_train = "attack" in name.lower()
    return experts, top_k, batch, "frozen" if frozen else "unfrozen", dataset, attack_train


def verdict_for(name: str, rel: str, m: dict | None, experts: int, top_k: int, frozen: str):
    if m is None or m.get("clean_ber_pct") is None:
        return "INCOMPLETE"
    ber = m["clean_ber_pct"]
    l1 = m.get("load_l1") or 99
    mu = m.get("max_use") or 1
    if "continue" in name.lower():
        return "BAD — overfit continue"
    if "sym_t14_unfrozen" in rel and m["ep"] == 20 and ber <= 0.5:
        return "GOOD — thesis main (R0)"
    if ber <= 0.5 and l1 < 0.15 and mu < 0.4:
        return "GOOD"
    if ber <= 1.0 and l1 < 0.3:
        return "OK — minor ablation"
    if ber > 2.0 or l1 > 0.5 or mu > 0.55:
        return "BAD — collapse / weak"
    if experts == 8 or top_k > 1:
        return "BAD — routing ablation"
    return "MARGINAL"


ATTACK_CSV = {
    "sym_t14_unfrozen_bal004warm10": "attack_summary_main_ep20.csv",
    "8exp_sparse_b128 2026.06.02": "attack_summary_8exp_k1_ep20.csv",
    "8exp_sparse_b128 2026.06.03": "attack_summary_8exp_300k_ep15.csv",
    "top2_b128": "attack_summary_top2_ep20.csv",
    "4exp_sparse_b128 2026.06.02--21-14": "attack_summary_dense_4exp_k4_ep20.csv",
}


def find_attack_csv(rel: str, name: str) -> str:
    for k, v in ATTACK_CSV.items():
        if k in rel or k in name:
            return v
    if "attack_summary.csv" in rel:
        return "attack_summary.csv (legacy)"
    return ""


def collect_runs():
    rows = []
    skip_parts = ("archive\\duplicates", "archive/duplicates", "moecollapse_csv_exports")
    for val in sorted(ROOT.rglob("validation.csv")):
        rel = str(val.parent.relative_to(ROOT)).replace("\\", "/")
        if any(s in rel for s in skip_parts):
            continue
        run_dir = val.parent
        name = run_dir.name
        experts, top_k, batch, frozen, dataset, attack_train = infer_config(rel, name)
        m = metrics_at_epoch(val, 20)
        rep_ep = m["ep"] if m else "?"
        noisy = noisy_at_epoch(run_dir, rep_ep) if isinstance(rep_ep, int) else None
        v = verdict_for(name, rel, m, experts, top_k, frozen)
        rows.append(
            {
                "tier": rel.split("/")[0] if "/" in rel else rel,
                "verdict": v,
                "run_name": name,
                "rel_path": rel,
                "dataset": dataset,
                "num_experts": experts,
                "top_k": top_k,
                "batch": batch,
                "backbone": frozen,
                "attack_training": "yes" if attack_train else "no",
                "report_epoch": rep_ep,
                "clean_val_ber_pct": m["clean_ber_pct"] if m else "",
                "noisy_val_ber_pct": noisy if noisy is not None else "",
                "expert_max_use": round(m["max_use"], 3) if m and m.get("max_use") else "",
                "train_val_load_l1": round(m["load_l1"], 3) if m and m.get("load_l1") else "",
                "attack_eval": find_attack_csv(rel, name),
                "has_checkpoint": "yes" if list(run_dir.glob("checkpoints/*.pyt")) else "csv only",
            }
        )
    return rows


ZIP_ONLY = [
    ("exp4continue.zip", "moe_unfrozen_sym_t14_v1_continue", "coco100k", 4, 1, 128, "unfrozen", "no", "21-60", "BAD — overfit", "—"),
    ("moe_unfrozen_8exp_sparse_b128 2026.06.02--21-36-20.zip", "8-exp 100k", "coco100k", 8, 1, 128, "unfrozen", "no", 20, "BAD — 1.7% @20", "attack_summary_8exp_k1_ep20.csv"),
    ("moe_unfrozen_8exp_sparse_b128 2026.06.03--09-06-45.zip", "8-exp ~300k", "~240k", 8, 1, 128, "unfrozen", "no", "15-20", "MARGINAL — use ep15", "attack_summary_8exp_300k_ep15.csv"),
    ("moe_unfrozen_4exp_sparse_b128_run.zip", "dense k=4 (misnamed sparse)", "coco100k", 4, 4, 128, "unfrozen", "no", 20, "BAD — 2.62%", "attack_summary_dense_4exp_k4_ep20.csv"),
    ("moe_collapse_diag_v1 2026.05.31--08-55-29.zip", "top-2 b128", "coco100k", 4, 2, 128, "unfrozen", "no", "20-30", "BAD — 1.91% @20", "attack_summary_top2_ep20.csv"),
]

IN_PROGRESS = [
    ("Kaggle log / dataset", "moe_unfrozen_sym_t14_attack_v1", "coco100k", 4, 1, 128, "unfrozen", "yes", "37+/80", "IN PROGRESS — R1", "after finish"),
    ("Kaggle log", "moe_unfrozen_sym_t14_200k_v1", "~240k", 4, 1, 128, "unfrozen", "no", "11+/30-60", "IN PROGRESS — data scale", "after finish"),
    ("trainexp4.txt", "moe_sym_bal08_v1", "coco100k", 4, 1, 128, "frozen", "no", "30", "IN PROGRESS — frozen", "after finish"),
]


def main():
    rows = collect_runs()
    # write csv
    fields = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Results folder audit",
        "",
        f"Generated from `{ROOT}` — {len(rows)} runs with `validation.csv`.",
        "",
        "## Thesis track (use in write-up)",
        "",
        "| Role | Verdict | Run | Experts | top-k | Batch | Frozen | Attack train | Clean @ep |",
        "|------|---------|-----|---------|-------|-------|--------|--------------|-----------|",
    ]
    thesis = [r for r in rows if "thesis main" in r["verdict"] or r["verdict"].startswith("GOOD — thesis")]
    for r in thesis:
        lines.append(
            f"| **Primary** | {r['verdict']} | `{r['run_name'][:40]}` | {r['num_experts']} | {r['top_k']} | {r['batch']} | {r['backbone']} | {r['attack_training']} | {r['clean_val_ber_pct']}% |"
        )

    lines += [
        "",
        "## All installed / extracted runs (sorted by verdict)",
        "",
        "| Verdict | Experts | k | Batch | Frozen | Data | Atk train | Ep | Clean% | Noisy% | max_use | load_l1 | Attack eval CSV | Location |",
        "|---------|---------|---|-------|--------|------|-----------|-----|--------|--------|---------|---------|-----------------|----------|",
    ]
    order = {"GOOD": 0, "OK": 1, "MARGINAL": 2, "BAD": 3, "INCOMPLETE": 4}
    for r in sorted(rows, key=lambda x: (order.get(x["verdict"][:4], 9), x["clean_val_ber_pct"] or 999)):
        lines.append(
            f"| {r['verdict']} | {r['num_experts']} | {r['top_k']} | {r['batch']} | {r['backbone']} | {r['dataset']} | {r['attack_training']} | {r['report_epoch']} | {r['clean_val_ber_pct']} | {r['noisy_val_ber_pct']} | {r['expert_max_use']} | {r['train_val_load_l1']} | {r['attack_eval']} | `{r['rel_path'][:55]}` |"
        )

    lines += [
        "",
        "## Zips in `results/` (not only under `experiments/`)",
        "",
        "| Zip | Description | Experts | k | Batch | Frozen | Clean @ep | Verdict | Attack CSV |",
        "|-----|-------------|---------|---|-------|--------|-----------|---------|------------|",
    ]
    for z in ZIP_ONLY:
        lines.append(f"| `{z[0]}` | {z[1]} | {z[3]} | {z[4]} | {z[5]} | {z[6]} | {z[7]} | {z[8]} | {z[9]} | {z[10]} |")

    lines += [
        "",
        "## In progress (Kaggle / logs — not full folder here)",
        "",
        "| Source | Name | Experts | k | Batch | Frozen | Attack train | Status |",
        "|--------|------|---------|---|-------|--------|--------------|--------|",
    ]
    for x in IN_PROGRESS:
        lines.append(f"| {x[0]} | {x[1]} | {x[3]} | {x[4]} | {x[5]} | {x[6]} | {x[7]} | {x[8]} |")

    lines += [
        "",
        "## Folder layout",
        "",
        "| Path | Contents |",
        "|------|----------|",
        "| `experiments/frozen_moe/` | 4 frozen COCO-100k runs (installed) |",
        "| `experiments/unfrozen_moe/` | **Main** unfrozen sym_t14 @ ep20 |",
        "| `experiments/legacy_early/` | 17 older sweeps (coco20k + coco100k) |",
        "| `attack_eval_runs/` | Extracted zips for 8-attack eval |",
        "| `exp4continue_extracted/` | Continue 21–60 (BAD) |",
        "| `comparison_hidden_vs_moe/` | Attack vs HiDDeN CSVs |",
        "| `plots/`, `thesis_figures/` | Generated figures |",
        "| `archive/` | Duplicate zips, partial runs |",
        "",
        "See also: `FULL_EXPERIMENT_REGISTRY.md`, `EXPERIMENTS_MAP.md`.",
        "",
        f"Regenerate: `python scripts/audit_results_folder.py`",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
