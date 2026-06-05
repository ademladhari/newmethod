#!/usr/bin/env python3
"""Build grouped thesis experiment tables from THESIS_ALL_EXPERIMENTS.csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC = RESULTS / "THESIS_ALL_EXPERIMENTS.csv"
OUT_MD = RESULTS / "THESIS_GROUPED_TABLES.md"
OUT_DIR = RESULTS / "thesis_tables_grouped"


def norm_dataset(d: str, experiment_name: str = "") -> str:
    d = (d or "").lower()
    name = (experiment_name or "").lower()
    if "240" in d or "200" in d or "200k" in name or "300k" in name:
        return "coco~240k"
    if "20k" in d:
        return "coco20k"
    if "100k" in d:
        return "coco100k"
    if "paper" in d:
        return "baseline"
    return "coco100k" if d == "" and "100k" in name else (d or "unknown")


def float_or_none(x):
    if x in ("", None):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_rows():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    return rows


def canonical_metrics(rows: list[dict]) -> list[dict]:
    """One row per on-disk run; skip archive duplicates and loose CSV dupes."""
    out = []
    seen_paths = set()
    for r in rows:
        if r["row_type"] != "metrics_on_disk":
            continue
        if "duplicate copy" in (r.get("duplicate_note") or ""):
            continue
        if r["experiment_name"] == "moecollapse_csv_exports":
            continue
        loc = r.get("location", "")
        if loc in seen_paths:
            continue
        seen_paths.add(loc)
        r = dict(r)
        r["dataset_group"] = norm_dataset(r.get("dataset", ""), r.get("experiment_name", ""))
        r["experts_n"] = r.get("num_experts") or "?"
        r["ber_sort"] = float_or_none(r.get("clean_ber_pct_report")) or float_or_none(
            r.get("clean_ber_pct_best")
        ) or 999.0
        out.append(r)
    out.sort(key=lambda x: (x["dataset_group"], int(x["experts_n"]) if x["experts_n"].isdigit() else 9, x["ber_sort"]))
    return out


def fmt(r, key, default="—"):
    v = r.get(key, "")
    return v if v not in ("", None) else default


def md_table(title: str, rows: list[dict], cols: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return [f"### {title}", "", "_No runs in this group._", ""]
    lines = [f"### {title}", "", "| " + " | ".join(c[0] for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in enumerate(rows, 1):
        cells = []
        for hdr, key in cols:
            if key == "#":
                cells.append(str(i))
            elif key == "experiment":
                cells.append(f"`{fmt(r, 'experiment_name')[:40]}`")
            elif key == "outcome_short":
                o = fmt(r, "outcome")
                cells.append(o.replace("FAILED (frozen baseline)", "frozen BAD").replace("FAILED", "FAIL")[:12])
            else:
                cells.append(fmt(r, key))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


COLS_FULL = [
    ("#", "#"),
    ("Outcome", "outcome_short"),
    ("Experiment", "experiment"),
    ("E", "num_experts"),
    ("k", "top_k"),
    ("Batch", "batch_size"),
    ("Frz", "backbone_frozen"),
    ("Ep", "epochs_logged"),
    ("BER%", "clean_ber_pct_report"),
    ("@ep", "report_epoch"),
    ("Best%", "clean_ber_pct_best"),
    ("max_use", "expert_max_use_ep20"),
    ("load_l1", "load_l1_ep20"),
    ("Noisy%", "noisy_ber_pct_ep20"),
    ("Attack CSV", "attack_eval_csv"),
]

COLS_COMPACT = [
    ("#", "#"),
    ("Experiment", "experiment"),
    ("E", "num_experts"),
    ("k", "top_k"),
    ("Batch", "batch_size"),
    ("Frz", "backbone_frozen"),
    ("BER%", "clean_ber_pct_report"),
    ("Best%", "clean_ber_pct_best"),
    ("max_use", "expert_max_use_ep20"),
    ("Outcome", "outcome_short"),
]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    all_rows = load_rows()
    canon = canonical_metrics(rows := all_rows)
    in_prog = [r for r in all_rows if r["row_type"] == "in_progress"]
    baseline = [r for r in all_rows if r["row_type"] == "baseline"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Grouped MoE experiment tables (thesis)",
        "",
        "Generated from `THESIS_ALL_EXPERIMENTS.csv` — **canonical on-disk runs only** "
        "(excludes archive duplicate folders and zip-only retry rows).",
        "",
        f"**{len(canon)} unique runs** with `validation.csv`. Regenerate: `python scripts/build_thesis_grouped_tables.py`",
        "",
        "---",
        "",
        "## Table of contents",
        "",
        "1. [By dataset](#1-by-dataset-coco20k--coco100k--coco240k)",
        "2. [4 experts vs 8 experts](#2-four-experts-vs-eight-experts)",
        "3. [By top-k routing](#3-by-top-k-routing)",
        "4. [Frozen vs unfrozen backbone](#4-frozen-vs-unfrozen-backbone-coco100k-family)",
        "5. [Thesis track only](#5-thesis-track-unfrozen--frozen--ablations)",
        "6. [By batch size](#6-by-batch-size-on-coco100k)",
        "7. [Outcome summary](#7-outcome-summary-counts)",
        "8. [In progress & baseline](#8-in-progress--baseline)",
        "",
        "---",
        "",
        "## 1. By dataset (coco20k / coco100k / coco~240k)",
        "",
    ]

    dataset_order = ["coco20k", "coco100k", "coco~240k", "other"]
    by_ds: dict[str, list] = defaultdict(list)
    for r in canon:
        by_ds[r["dataset_group"]].append(r)

    for ds in dataset_order:
        group = by_ds.get(ds, [])
        if not group:
            continue
        group.sort(key=lambda x: x["ber_sort"])
        write_csv(OUT_DIR / f"by_dataset_{ds.replace('~', '')}.csv", group)
        lines += md_table(f"Dataset: **{ds}** ({len(group)} runs)", group, COLS_FULL)

    lines += ["---", "", "## 2. Four experts vs eight experts", ""]

    for n, label in [(4, "4 experts"), (8, "8 experts")]:
        g = [r for r in canon if r.get("num_experts") == str(n)]
        g.sort(key=lambda x: (x["dataset_group"], x["ber_sort"]))
        write_csv(OUT_DIR / f"by_{n}experts.csv", g)
        lines += md_table(f"**{label}** — all datasets ({len(g)} runs)", g, COLS_COMPACT)

    lines += ["---", "", "## 3. By top-k routing", ""]

    for k in ["1", "2", "3", "4"]:
        g = [r for r in canon if r.get("top_k") == k]
        if not g:
            continue
        g.sort(key=lambda x: x["ber_sort"])
        label = { "1": "top-k = 1 (sparse)", "2": "top-k = 2", "3": "top-k = 3", "4": "top-k = 4 (dense)"}[k]
        write_csv(OUT_DIR / f"by_topk_{k}.csv", g)
        lines += md_table(f"**{label}** ({len(g)} runs)", g, COLS_COMPACT)

    lines += ["---", "", "## 4. Frozen vs unfrozen backbone (coco100k family)", ""]

    g100 = [r for r in canon if r["dataset_group"] == "coco100k"]
    for frz, label in [("False", "Unfrozen backbone"), ("True", "Frozen backbone")]:
        g = [r for r in g100 if str(r.get("backbone_frozen")).lower() == frz.lower()]
        g.sort(key=lambda x: x["ber_sort"])
        write_csv(OUT_DIR / f"coco100k_{'unfrozen' if frz=='False' else 'frozen'}.csv", g)
        lines += md_table(f"coco100k — **{label}** ({len(g)} runs)", g, COLS_FULL)

    lines += ["---", "", "## 5. Thesis track (unfrozen + frozen + ablations)", ""]

    thesis_branches = {
        "Primary & frozen (thesis configs)": lambda r: r.get("branch") in (
            "unfrozen_moe",
            "frozen_moe",
        )
        and "continue" not in r.get("location", ""),
        "100k routing ablations (attack_eval)": lambda r: r.get("branch") == "attack_eval_runs",
        "Negative controls": lambda r: "continue" in r.get("location", ""),
    }
    for title, pred in thesis_branches.items():
        g = [r for r in canon if pred(r)]
        g.sort(key=lambda x: x["ber_sort"])
        slug = title.split()[0].lower()
        write_csv(OUT_DIR / f"thesis_{slug}.csv", g)
        lines += md_table(f"{title} ({len(g)} runs)", g, COLS_FULL)

    lines += ["---", "", "## 6. By batch size (on coco100k)", ""]

    batches = sorted({r.get("batch_size") for r in g100 if r.get("batch_size")}, key=lambda x: int(x) if x.isdigit() else 0)
    for b in batches:
        g = [r for r in g100 if r.get("batch_size") == b]
        g.sort(key=lambda x: x["ber_sort"])
        lines += md_table(f"coco100k — **batch {b}** ({len(g)} runs)", g, COLS_COMPACT)

    lines += ["---", "", "## 7. Outcome summary (counts)", "", "| Dataset | 4-exp | 8-exp | Best BER (4-exp) | Best BER (8-exp) |", "|---------|-------|-------|------------------|------------------|"]
    for ds in ["coco20k", "coco100k"]:
        g = by_ds.get(ds, [])
        e4 = [r for r in g if r.get("num_experts") == "4"]
        e8 = [r for r in g if r.get("num_experts") == "8"]
        b4 = min((r["ber_sort"] for r in e4), default=None)
        b8 = min((r["ber_sort"] for r in e8), default=None)
        lines.append(
            f"| {ds} | {len(e4)} | {len(e8)} | {b4 if b4 != 999 else '—'}% | {b8 if b8 != 999 else '—'}% |"
        )
    lines.append("")

    lines += [
        "---",
        "",
        "## 8. In progress & baseline",
        "",
        "### Baseline (non-MoE)",
        "",
    ]
    if baseline:
        lines.append("| Experiment | Notes |")
        lines.append("|------------|-------|")
        for r in baseline:
            lines.append(f"| `{r['experiment_name']}` | {r.get('why_failed_or_role', '')[:80]} |")
        lines.append("")

    lines.append("### In progress (no local validation.csv yet)")
    lines.append("")
    lines.append("| Experiment | Dataset | E | k | Batch | Frozen | Status | Thesis use |")
    lines.append("|------------|---------|---|---|-------|--------|--------|------------|")
    for r in in_prog:
        lines.append(
            f"| `{r['experiment_name']}` | {r.get('dataset','')} | {r.get('num_experts','')} | {r.get('top_k','')} | {r.get('batch_size','')} | {r.get('backbone_frozen','')} | {r.get('outcome','')} | {r.get('thesis_use','')} |"
        )
    lines.append("")

    lines += [
        "---",
        "",
        "## Cross-tab: dataset × experts (canonical runs)",
        "",
        "| | **4 experts** (count) | **4-exp best BER** | **8 experts** (count) | **8-exp best BER** |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for ds in ["coco20k", "coco100k"]:
        g = by_ds.get(ds, [])
        e4 = [r for r in g if r.get("num_experts") == "4"]
        e8 = [r for r in g if r.get("num_experts") == "8"]
        b4 = min((r["ber_sort"] for r in e4), default=None)
        b8 = min((r["ber_sort"] for r in e8), default=None)
        lines.append(
            f"| **{ds}** | {len(e4)} | {f'{b4:.2f}%' if b4 and b4 < 900 else '—'} | {len(e8)} | {f'{b8:.2f}%' if b8 and b8 < 900 else '—'} |"
        )
    lines.append("")
    lines += [
        "### coco100k — sorted by BER (full list)",
        "",
    ]
    g100.sort(key=lambda x: x["ber_sort"])
    write_csv(OUT_DIR / "coco100k_all_sorted.csv", g100)
    lines += md_table("All coco100k runs (best → worst)", g100, COLS_FULL)

    g20 = by_ds.get("coco20k", [])
    g20.sort(key=lambda x: x["ber_sort"])
    write_csv(OUT_DIR / "coco20k_all_sorted.csv", g20)
    lines += ["### coco20k — sorted by BER", ""]
    lines += md_table("All coco20k runs (best → worst)", g20, COLS_FULL)

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote CSVs under {OUT_DIR}/ ({len(list(OUT_DIR.glob('*.csv')))} files)")
    print(f"Canonical runs: {len(canon)}")


if __name__ == "__main__":
    main()
