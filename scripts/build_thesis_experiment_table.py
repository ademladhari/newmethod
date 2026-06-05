#!/usr/bin/env python3
"""Build complete thesis experiment table — every run, including failed attempts."""
from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REGISTRY = RESULTS / "FULL_EXPERIMENT_REGISTRY.json"
OUT_CSV = RESULTS / "THESIS_ALL_EXPERIMENTS.csv"
OUT_MD = RESULTS / "THESIS_ALL_EXPERIMENTS.md"

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
    "adversarial_loss",
    "freeze_hidden_backbone",
    "apply_training_noise",
    "train_folder",
]


def read_config(run_dir: Path) -> dict:
    cfg: dict = {}
    logs = list(run_dir.glob("*.log"))
    if logs:
        text = logs[0].read_text(encoding="utf-8", errors="replace")
        for key in CONFIG_KEYS:
            m = re.search(r"'" + key + r"': ([^\n,]+)", text)
            if m:
                cfg[key] = m.group(1).strip().strip("'\"")
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
                        cfg[key] = str(v)
        except Exception:
            pass
    tf = str(cfg.get("train_folder", ""))
    if "unlabeled" in tf or "200k" in tf or "240k" in tf:
        cfg["dataset"] = "coco~240k"
    elif "coco20k" in tf:
        cfg["dataset"] = "coco20k"
    elif "coco100k" in tf or "train2017" in tf:
        cfg["dataset"] = "coco100k"
    else:
        cfg.setdefault("dataset", "")
    return cfg


def g(row, *keys):
    for k in keys:
        for c in (k, k + " "):
            if c in row and row[c] not in ("", None):
                return float(row[c])
    return None


def val_metrics(val_path: Path) -> dict:
    if not val_path.is_file():
        return {}
    rows = list(csv.DictReader(val_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {}
    best = min(rows, key=lambda r: g(r, "bitwise-error", "val_ber") or 999)
    ep20 = next((r for r in rows if int(r["epoch"]) == 20), None)
    final = rows[-1]

    def pack(r):
        ber = g(r, "bitwise-error", "val_ber")
        return {
            "epoch": int(r["epoch"]),
            "clean_pct": round(ber * 100, 3) if ber is not None else None,
            "max_use": g(r, "expert_max_use"),
            "load_l1": g(r, "train_val_load_l1"),
        }

    out = {"epochs_logged": int(final["epoch"]), "best": pack(best), "final": pack(final)}
    if ep20:
        out["ep20"] = pack(ep20)
    noisy_path = val_path.parent / "validation_noisy.csv"
    if noisy_path.is_file() and ep20:
        nrow = next(
            (r for r in csv.DictReader(noisy_path.open(newline="", encoding="utf-8")) if int(r["epoch"]) == 20),
            None,
        )
        if nrow:
            nb = g(nrow, "bitwise-error")
            out["noisy_ep20_pct"] = round(nb * 100, 3) if nb is not None else None
    return out


def branch_for(rel: str) -> str:
    if rel.startswith("experiments/"):
        return rel.split("/")[1]
    if rel.startswith("attack_eval"):
        return "attack_eval_runs"
    if "continue" in rel:
        return "continue_run"
    if rel.startswith("archive/"):
        return "archive"
    return rel.split("/")[0]


def thesis_role(name: str, cfg: dict, m: dict, rel: str) -> tuple[str, str, str]:
    """Returns (outcome, thesis_use, why_explanation)."""
    exp = cfg.get("experiment_name", name)
    experts = int(cfg.get("num_experts") or 4)
    top_k = int(cfg.get("top_k") or 1)
    frozen = str(cfg.get("freeze_hidden_backbone", "")).lower() == "true"
    ep20 = m.get("ep20") or {}
    best = m.get("best") or {}
    ber20 = ep20.get("clean_pct")
    ber_best = best.get("clean_pct")
    ber_report = ber20 if ber20 is not None else ber_best
    rep_ep = ep20.get("epoch") or best.get("epoch") or "?"
    mu = (ep20 or best).get("max_use") or 0
    l1 = (ep20 or best).get("load_l1") or 99

    if exp == "moe_collapse_diag_8exp_v1":
        return (
            "FAILED",
            "Diagnostic — 8-exp collapse study",
            f"8 experts top-2 with load penalty: clean BER ok early ({ber_report}% @ep{rep_ep}) but load_l1={l1:.2f}, max_use={mu:.2f} — routing stress test, not a production config.",
        )

    if "continue" in name.lower():
        ep60 = m.get("ep60", {})
        b60 = ep60.get("clean_pct")
        return (
            "FAILED",
            "Negative control — do not use",
            f"Same config as main but trained ep21–60; val BER rises (ep20 {ber20}% → ep60 {b60}%): overfitting / routing drift, not improved robustness.",
        )

    if exp == "moe_unfrozen_sym_t14_v1" and not frozen and top_k == 1 and experts == 4:
        if ber20 is not None and ber20 <= 0.5:
            return (
                "SUCCESS",
                "Primary model (R0)",
                "Unfrozen backbone, symmetric router bal=0.04, sparse top-1: best clean BER and stable load (max_use≈0.31, load_l1≈0.08).",
            )

    if "attack" in exp.lower() and cfg.get("apply_training_noise", "").lower() == "true":
        return (
            "IN PROGRESS",
            "Planned R1 (attack curriculum)",
            "80-epoch run with training-time attacks ramping after ep20; compare to R0 after completion.",
        )

    if "200k" in exp.lower():
        return (
            "IN PROGRESS",
            "Data-scale ablation",
            "~240k train images vs 100k; tests whether more data improves BER without changing architecture.",
        )

    if exp == "moe_sym_bal08_v1" and frozen:
        if ber20 and ber20 > 1.5:
            return (
                "FAILED (frozen baseline)",
                "Thesis comparison — frozen backbone",
                "Frozen HiDDeN encoder limits adaptation; higher clean BER (~2.7%) despite similar routing — used for frozen vs unfrozen discussion.",
            )

    if experts == 8 and top_k == 1:
        return (
            "FAILED",
            "Negative — 8 experts",
            f"More experts + same top-1: worse BER ({ber20 or ber_best}%), harder load balancing; 100k data insufficient for 8-way routing.",
        )

    if top_k >= 4:
        return (
            "FAILED",
            "Negative — dense routing",
            f"top-k={top_k} (dense): BER {ber20 or ber_best}%; defeats sparse specialization, router averages experts.",
        )

    if top_k == 2:
        return (
            "FAILED",
            "Negative — top-2 routing",
            f"top-k=2: BER {ber20 or ber_best}%; split gradients across two experts per token, weaker than top-1.",
        )

    if "300k" in rel or "2026.06.03" in name:
        return (
            "FAILED",
            "Negative — large data 8-exp",
            f"~300k images, 8 experts: best ~ep15 then collapse by ep17+ (max_use→0.5+); scale did not fix routing.",
        )

    if mu >= 0.55 or l1 >= 0.5 or (ber20 and ber20 > 2.0):
        return (
            "FAILED",
            "Collapsed / unstable routing",
            f"Expert collapse or imbalance: clean BER {ber20 or ber_best}%, max_use={mu:.2f}, load_l1={l1:.2f}.",
        )

    if ber20 and ber20 > 1.0:
        return (
            "FAILED",
            "Legacy sweep — weak BER",
            f"Early recipe (strong balance/jitter or wrong batch): BER {ber20}% at ep20; superseded by sym_t14 family.",
        )

    if ber20 and ber20 <= 1.0 and l1 < 0.35:
        return (
            "MARGINAL",
            "Optional ablation / diagnostic",
            f"Usable BER ({ber20}%) but not primary; kept for hyperparameter exploration narrative.",
        )

    return (
        "FAILED",
        "Legacy / exploratory",
        f"Did not meet thesis quality bar (BER {ber20 or ber_best}%, routing metrics).",
    )


ATTACK_MAP = {
    "sym_t14_unfrozen": "attack_summary_main_ep20.csv",
    "8exp_sparse_b128 2026.06.02": "attack_summary_8exp_k1_ep20.csv",
    "8exp_sparse_b128 2026.06.03": "attack_summary_8exp_300k_ep15.csv",
    "top2_b128": "attack_summary_top2_ep20.csv",
    "4exp_sparse_b128 2026.06.02--21-14": "attack_summary_dense_4exp_k4_ep20.csv",
}


def attack_csv_for(rel: str, name: str) -> str:
    for k, v in ATTACK_MAP.items():
        if k in rel or k in name:
            return v
    return ""


def row_from_run_dir(run_dir: Path, rel: str, *, duplicate_note: str = "") -> dict:
    cfg = read_config(run_dir)
    m = val_metrics(run_dir / "validation.csv")
    name = run_dir.name
    exp = cfg.get("experiment_name", name)

    # continue ep60
    if "continue" in name.lower():
        rows = list(csv.DictReader((run_dir / "validation.csv").open()))
        ep60 = next((r for r in rows if int(r["epoch"]) == 60), None)
        if ep60:
            m["ep60"] = {
                "epoch": 60,
                "clean_pct": round(g(ep60, "bitwise-error") * 100, 3),
                "max_use": g(ep60, "expert_max_use"),
                "load_l1": g(ep60, "train_val_load_l1"),
            }

    outcome, thesis_use, why = thesis_role(name, cfg, m, rel)
    ep20 = m.get("ep20") or {}
    best = m.get("best") or {}
    report = ep20 if ep20 else best

    frozen = cfg.get("freeze_hidden_backbone", "")
    if frozen == "":
        frozen = "True" if "frozen_moe" in rel and "unfrozen" not in name else "False"

    return {
        "row_type": "metrics_on_disk",
        "duplicate_note": duplicate_note,
        "experiment_name": exp,
        "run_folder": name,
        "location": rel,
        "branch": branch_for(rel),
        "dataset": cfg.get("dataset", ""),
        "num_experts": cfg.get("num_experts", ""),
        "top_k": cfg.get("top_k", ""),
        "batch_size": cfg.get("batch_size", ""),
        "epochs_planned": cfg.get("number_of_epochs", ""),
        "epochs_logged": m.get("epochs_logged", ""),
        "backbone_frozen": frozen,
        "balance_w": cfg.get("balance_loss_weight", ""),
        "balance_start": cfg.get("balance_loss_start_weight", ""),
        "balance_warmup": cfg.get("balance_loss_warmup_epochs", ""),
        "router_jitter": cfg.get("router_jitter_noise", ""),
        "load_penalty": cfg.get("load_penalty_weight", "0"),
        "temp_start_end": f"{cfg.get('router_temperature_start','?')}→{cfg.get('router_temperature_end','?')}",
        "attack_training": cfg.get("apply_training_noise", "False"),
        "outcome": outcome,
        "thesis_use": thesis_use,
        "why_failed_or_role": why,
        "clean_ber_pct_ep20": ep20.get("clean_pct", ""),
        "clean_ber_pct_report": report.get("clean_pct", ""),
        "report_epoch": report.get("epoch", ""),
        "noisy_ber_pct_ep20": m.get("noisy_ep20_pct", ""),
        "expert_max_use_ep20": round(report["max_use"], 4) if report.get("max_use") is not None else "",
        "load_l1_ep20": round(report["load_l1"], 4) if report.get("load_l1") is not None else "",
        "clean_ber_pct_best": best.get("clean_pct", ""),
        "best_val_epoch": best.get("epoch", ""),
        "clean_ber_pct_final": m.get("final", {}).get("clean_pct", ""),
        "final_epoch": m.get("final", {}).get("epoch", ""),
        "attack_eval_csv": attack_csv_for(rel, name),
        "has_checkpoint": "yes" if list(run_dir.glob("checkpoints/*.pyt")) else "no",
    }


def load_registry_by_path() -> dict[str, dict]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_path = {}
    for e in data.get("installed_experiments", []):
        by_path[e["path"].replace("\\", "/")] = e
    return by_path, data


# Zip folders with empty audit fingerprint → installed run that has the real metrics
ZIP_RECOVERED_METRICS = {
    "moe4_ber_route_v1": "experiments/legacy_early/coco20k/4exp_k2/batch32/moe4_ber_route_v1_ep43_2026-05-29",
    "moe4_uniform_first_100k_v1": "experiments/legacy_early/coco100k/4exp_k2/batch24/moe4_uniform_first_100k_v1_ep19_2026-05-29",
    "moe_hidden": "experiments/legacy_early/coco20k/8exp_k2/batch16/moe_hidden_ep20_2026-05-18",
    "moe_hidden 2026.05.18--18-07-20_export": "experiments/legacy_early/coco20k/8exp_k2/batch16/moe_hidden_ep20_2026-05-18",
}


def zip_audit_rows(data: dict, on_disk_exps: set[str]) -> list[dict]:
    """Every run folder ever seen in a zip (35 entries) — includes crash retries."""
    rows = []
    for zr in data.get("zip_audit", {}).get("all_runs_in_zips", []):
        cfg = zr.get("config") or {}
        exp = cfg.get("experiment_name", zr.get("run_folder", "?").split()[0])
        has_val = zr.get("has_validation", False)
        has_train = zr.get("has_train_csv", False)
        fp = zr.get("config_fingerprint", "")

        if fp == "16da67a744be":
            why = (
                "Zip folder had no .log/pickle/validation.csv (audit empty fingerprint). "
                "Usually a failed Kaggle restart — see results/ZIP_CRASH_SIX_CLARIFIED.md for recovered metrics."
            )
            outcome = "ZIP_RETRY_EMPTY"
        elif not has_train and not has_val:
            why = "Partial export — checkpoints or logs only, no train/validation CSV."
            outcome = "PARTIAL"
        elif has_train and has_val and exp in on_disk_exps:
            why = "Full run — same config installed under experiments/ (metrics on disk)."
            outcome = "INSTALLED"
        elif has_train and has_val:
            why = "Full run in zip but not copied to experiments/ tree."
            outcome = "ZIP_FULL"
        else:
            why = "Incomplete training export."
            outcome = "PARTIAL"

        recovered_path = ZIP_RECOVERED_METRICS.get(exp) or ZIP_RECOVERED_METRICS.get(
            zr.get("run_folder", "").split()[0]
        )
        rec_metrics = {}
        if recovered_path:
            rec_dir = RESULTS / recovered_path
            if rec_dir.is_dir():
                rec_metrics = val_metrics(rec_dir / "validation.csv")
                if fp == "16da67a744be":
                    outcome = "ZIP_RETRY_EMPTY"
                    why = (
                        f"Empty zip folder; metrics recovered from installed run `{recovered_path}`."
                    )

        rep = rec_metrics.get("ep20") or rec_metrics.get("best") or {}
        rows.append(
            {
                "row_type": "zip_history",
                "duplicate_note": f"fingerprint={fp}" if fp else "",
                "experiment_name": exp,
                "run_folder": zr.get("run_folder", ""),
                "location": zr.get("zip_file", ""),
                "branch": "zip_archive",
                "dataset": cfg.get("dataset", ""),
                "num_experts": cfg.get("num_experts", ""),
                "top_k": cfg.get("top_k", ""),
                "batch_size": cfg.get("batch_size", ""),
                "epochs_planned": cfg.get("number_of_epochs", ""),
                "epochs_logged": zr.get("epochs_completed", ""),
                "backbone_frozen": cfg.get("freeze_hidden_backbone", ""),
                "balance_w": cfg.get("balance_loss_weight", ""),
                "balance_start": cfg.get("balance_loss_start_weight", ""),
                "balance_warmup": cfg.get("balance_loss_warmup_epochs", ""),
                "router_jitter": cfg.get("router_jitter_noise", ""),
                "load_penalty": cfg.get("load_penalty_weight", ""),
                "temp_start_end": (
                    f"{cfg.get('router_temperature_start', '')}→{cfg.get('router_temperature_end', '')}"
                    if cfg.get("router_temperature_start")
                    else ""
                ),
                "attack_training": "",
                "outcome": outcome,
                "thesis_use": "Upload / retry history",
                "why_failed_or_role": why,
                "metrics_recovered_from": recovered_path or "",
                "clean_ber_pct_ep20": rep.get("clean_pct", ""),
                "clean_ber_pct_report": rep.get("clean_pct", ""),
                "report_epoch": rep.get("epoch", ""),
                "noisy_ber_pct_ep20": "",
                "expert_max_use_ep20": round(rep["max_use"], 4) if rep.get("max_use") is not None else "",
                "load_l1_ep20": round(rep["load_l1"], 4) if rep.get("load_l1") is not None else "",
                "clean_ber_pct_best": (rec_metrics.get("best") or {}).get("clean_pct", ""),
                "best_val_epoch": (rec_metrics.get("best") or {}).get("epoch", ""),
                "clean_ber_pct_final": (rec_metrics.get("final") or {}).get("clean_pct", ""),
                "final_epoch": (rec_metrics.get("final") or {}).get("epoch", ""),
                "attack_eval_csv": "",
                "has_checkpoint": "train" if has_train else "no",
            }
        )
    return rows


IN_PROGRESS = [
    {
        "experiment_name": "moe_unfrozen_sym_t14_attack_v1",
        "run_folder": "(Kaggle continue)",
        "location": "Kaggle / continuetrain dataset",
        "dataset": "coco100k",
        "num_experts": "4",
        "top_k": "1",
        "batch_size": "128",
        "epochs_planned": "80",
        "epochs_logged": "37+",
        "backbone_frozen": "False",
        "balance_w": "0.04",
        "attack_training": "True",
        "outcome": "IN PROGRESS",
        "thesis_use": "R1 attack-trained model",
        "why_failed_or_role": "Training-time noise curriculum; ep37 ~0.56% clean — finish to ep80 then attack eval vs R0.",
    },
    {
        "experiment_name": "moe_unfrozen_sym_t14_200k_v1",
        "run_folder": "(Kaggle)",
        "location": "Kaggle coco200k",
        "dataset": "coco~240k",
        "num_experts": "4",
        "top_k": "1",
        "batch_size": "128",
        "epochs_planned": "30-60",
        "epochs_logged": "11+",
        "backbone_frozen": "False",
        "balance_w": "0.04",
        "attack_training": "False",
        "outcome": "IN PROGRESS",
        "thesis_use": "Data-scale ablation",
        "why_failed_or_role": "Same architecture as R0 on ~241k train images; tests data scaling hypothesis.",
    },
    {
        "experiment_name": "moe_sym_bal08_v1",
        "run_folder": "(Colab/Kaggle retrain)",
        "location": "results/logs/moe_sym_bal08_v1_colab_console.txt",
        "dataset": "coco100k",
        "num_experts": "4",
        "top_k": "1",
        "batch_size": "128",
        "epochs_planned": "30",
        "epochs_logged": "30",
        "backbone_frozen": "True",
        "balance_w": "0.08",
        "attack_training": "False",
        "outcome": "IN PROGRESS",
        "thesis_use": "Frozen backbone comparison",
        "why_failed_or_role": "Re-run frozen bal08 for fair attack eval; installed copy has 2.72% @ep20.",
    },
]

HIDDEN_BASELINE = {
    "row_type": "baseline",
    "experiment_name": "HiDDeN (no MoE)",
    "run_folder": "my_hidden_experiment--epoch-177",
    "location": "hiddenepcoh177/",
    "branch": "baseline",
    "dataset": "coco (paper)",
    "num_experts": "0",
    "top_k": "0",
    "batch_size": "12",
    "epochs_planned": "177",
    "epochs_logged": "177",
    "backbone_frozen": "n/a",
    "balance_w": "",
    "balance_start": "",
    "balance_warmup": "",
    "router_jitter": "",
    "load_penalty": "",
    "temp_start_end": "",
    "attack_training": "False",
    "outcome": "BASELINE",
    "thesis_use": "HiDDeN-177 comparison in 8-attack eval",
    "why_failed_or_role": "Non-MoE baseline for attack comparison table (identity/jpeg/resize/etc.).",
    "clean_ber_pct_ep20": "",
    "noisy_ber_pct_ep20": "",
    "expert_max_use_ep20": "",
    "load_l1_ep20": "",
    "clean_ber_pct_best": "",
    "best_val_epoch": "177",
    "clean_ber_pct_final": "",
    "final_epoch": "177",
    "attack_eval_csv": "comparison_hidden_vs_moe/*.csv",
        "has_checkpoint": "yes",
        "duplicate_note": "",
        "metrics_recovered_from": "",
    }


def fields_default():
    return [
        "row_type",
        "duplicate_note",
        "experiment_name",
        "run_folder",
        "location",
        "branch",
        "dataset",
        "num_experts",
        "top_k",
        "batch_size",
        "epochs_planned",
        "epochs_logged",
        "backbone_frozen",
        "balance_w",
        "balance_start",
        "balance_warmup",
        "router_jitter",
        "load_penalty",
        "temp_start_end",
        "attack_training",
        "outcome",
        "thesis_use",
        "why_failed_or_role",
        "clean_ber_pct_ep20",
        "clean_ber_pct_report",
        "report_epoch",
        "noisy_ber_pct_ep20",
        "expert_max_use_ep20",
        "load_l1_ep20",
        "clean_ber_pct_best",
        "best_val_epoch",
        "clean_ber_pct_final",
        "final_epoch",
        "attack_eval_csv",
        "has_checkpoint",
        "metrics_recovered_from",
    ]


def main():
    by_path, reg = load_registry_by_path()
    rows: list[dict] = []

    for val in sorted(RESULTS.rglob("validation.csv")):
        rel = str(val.parent.relative_to(RESULTS)).replace("\\", "/")
        dup = ""
        if "archive/duplicates" in rel:
            dup = "duplicate copy in archive"
        elif "archive/loose" in rel:
            dup = "legacy loose export"
        run_dir = val.parent
        r = row_from_run_dir(run_dir, rel, duplicate_note=dup)
        for p, meta in by_path.items():
            if rel.endswith(p.split("/")[-1]) or p in rel:
                cfg = meta.get("config", {})
                r["dataset"] = r["dataset"] or cfg.get("dataset", "")
        rows.append(r)

    on_disk_exps = {r["experiment_name"] for r in rows if r.get("experiment_name")}
    rows.extend(zip_audit_rows(reg, on_disk_exps))

    for ip in IN_PROGRESS:
        ip_row = {k: "" for k in fields_default()}
        ip_row.update({"row_type": "in_progress"})
        ip_row.update(ip)
        rows.append(ip_row)

    hid = {k: "" for k in fields_default()}
    hid.update(HIDDEN_BASELINE)
    rows.append(hid)

    order = {"SUCCESS": 0, "BASELINE": 1, "IN PROGRESS": 2, "MARGINAL": 3, "FAILED": 4, "INCOMPLETE": 5, "CRASH": 6, "PARTIAL": 7, "INSTALLED": 8, "ZIP_FULL": 9}
    rows.sort(
        key=lambda r: (
            order.get(r.get("outcome", "FAILED"), 9),
            r.get("clean_ber_pct_ep20") if r.get("clean_ber_pct_ep20") != "" else 999,
            r.get("experiment_name", ""),
        )
    )

    # normalize keys
    all_fields = fields_default()
    for r in rows:
        for k in all_fields:
            r.setdefault(k, "")

    fields = all_fields
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter

    lines = [
        "# Complete MoE experiment table (thesis)",
        "",
        f"**{len(rows)} rows total** in `THESIS_ALL_EXPERIMENTS.csv`.",
        "",
        "| Section | Rows | Description |",
        "|---------|------|-------------|",
        f"| **A — Metrics on disk** | {sum(1 for r in rows if r['row_type']=='metrics_on_disk')} | Every `validation.csv` (incl. archive copies) |",
        f"| **B — Zip upload history** | {sum(1 for r in rows if r['row_type']=='zip_history')} | All 35 run folders from Kaggle/Colab zips |",
        f"| **C — In progress** | {sum(1 for r in rows if r['row_type']=='in_progress')} | Not yet exported locally |",
        f"| **D — Baseline** | {sum(1 for r in rows if r['row_type']=='baseline')} | HiDDeN epoch-177 |",
        "",
        "Regenerate: `python scripts/build_thesis_experiment_table.py`",
        "",
        "## Summary by outcome (on-disk runs only)",
        "",
    ]
    c = Counter(r["outcome"] for r in rows if r["row_type"] == "metrics_on_disk")
    for k, v in sorted(c.items(), key=lambda x: order.get(x[0], 9)):
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## A — All runs with validation metrics", ""])
    hdr = ["#", "Outcome", "Experiment", "E", "k", "Batch", "Data", "Frz", "Ep", "BER%", "@ep", "Best%", "max_use", "load_l1", "Why failed / role"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    n = 0
    for r in rows:
        if r["row_type"] != "metrics_on_disk":
            continue
        n += 1
        lines.append(
            "| {i} | {outcome} | `{exp}` | {e} | {k} | {b} | {d} | {f} | {elog} | {cr} | {rep} | {cb} | {mu} | {l1} | {why} |".format(
                i=n,
                outcome=r.get("outcome", ""),
                exp=(r.get("experiment_name", "") or "")[:32],
                e=r.get("num_experts", ""),
                k=r.get("top_k", ""),
                b=r.get("batch_size", ""),
                d=str(r.get("dataset", ""))[:10],
                f=str(r.get("backbone_frozen", ""))[:5],
                elog=r.get("epochs_logged", ""),
                cr=r.get("clean_ber_pct_report", r.get("clean_ber_pct_ep20", "")),
                rep=r.get("report_epoch", ""),
                cb=r.get("clean_ber_pct_best", ""),
                mu=r.get("expert_max_use_ep20", ""),
                l1=r.get("load_l1_ep20", ""),
                why=(r.get("why_failed_or_role", "") or "")[:100],
            )
        )
    zips = [r for r in rows if r["row_type"] == "zip_history"]
    lines.extend(["", "## B — Zip upload / retry history (35 entries)", ""])
    lines.append(
        "See **`ZIP_CRASH_SIX_CLARIFIED.md`** for the old mislabel “6 CRASH” — those are empty zip folders, not lost experiments."
    )
    lines.append("")
    lines.append("| # | Outcome | Experiment | Zip | BER% (recovered) | Recovered from | Notes |")
    lines.append("|---|---------|------------|-----|------------------|----------------|-------|")
    for i, r in enumerate(zips, 1):
        lines.append(
            f"| {i} | {r['outcome']} | `{r['experiment_name']}` | `{r['location']}` | {r.get('clean_ber_pct_report','')} | `{r.get('metrics_recovered_from','')[:40]}` | {r['why_failed_or_role'][:70]} |"
        )
    prog = [r for r in rows if r["row_type"] in ("in_progress", "baseline")]
    lines.extend(["", "## C/D — Baseline & in-progress", ""])
    lines.append("| Outcome | Experiment | Location | Thesis use |")
    lines.append("|---------|------------|----------|------------|")
    for r in prog:
        lines.append(f"| {r['outcome']} | `{r['experiment_name']}` | {r['location']} | {r['thesis_use']} |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")
    print(f"Wrote {OUT_MD}")
    on_disk = sum(1 for r in rows if r["row_type"] == "metrics_on_disk")
    print(f"  metrics on disk: {on_disk}")
    print(f"  zip history: {sum(1 for r in rows if r['row_type']=='zip_history')}")
    print(f"  in progress: {sum(1 for r in rows if r['row_type']=='in_progress')}")


if __name__ == "__main__":
    main()
