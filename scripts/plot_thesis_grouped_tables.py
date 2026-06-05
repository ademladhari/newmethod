#!/usr/bin/env python3
"""Render grouped thesis CSVs as visual tables (figures only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPED_DIR = ROOT / "results" / "thesis_tables_grouped"
OUT = [ROOT / "results" / "plots" / "grouped", ROOT / "thesis" / "figures" / "grouped"]

# Columns shown in each table figure (readable subset)
DISPLAY_COLS = [
    "experiment_name",
    "num_experts",
    "top_k",
    "batch_size",
    "backbone_frozen",
    "epochs_logged",
    "clean_ber_pct_report",
    "clean_ber_pct_best",
    "expert_max_use_ep20",
    "load_l1_ep20",
    "outcome",
]

HEADER_LABELS = {
    "experiment_name": "Experiment",
    "num_experts": "E",
    "top_k": "k",
    "batch_size": "Batch",
    "backbone_frozen": "Frozen",
    "epochs_logged": "Ep",
    "clean_ber_pct_report": "BER%",
    "clean_ber_pct_best": "Best%",
    "expert_max_use_ep20": "max_use",
    "load_l1_ep20": "load_l1",
    "outcome": "Outcome",
}


def df_for_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        out[c] = out[c].fillna("—").astype(str)
        if c == "experiment_name":
            out[c] = out[c].str.replace("moe_", "", regex=False)
        if c == "backbone_frozen":
            out[c] = out[c].map({"True": "Y", "False": "N", "true": "N", "false": "N"}).fillna(out[c])
        if c == "outcome":
            out[c] = out[c].str.replace("FAILED (frozen baseline)", "frozen fail", regex=False)
            out[c] = out[c].str[:18]
    return out


def render_table_figure(df: pd.DataFrame, title: str, slug: str):
    headers = [HEADER_LABELS.get(c, c) for c in df.columns]
    cells = df.values.tolist()
    nrows, ncols = len(cells), len(headers)
    fig_w = max(10, ncols * 1.15)
    fig_h = max(2.5, 0.38 * nrows + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)
    # Grayscale print style (no color)
    for j in range(ncols):
        tbl[(0, j)].set_facecolor("#d8d8d8")
        tbl[(0, j)].set_text_props(weight="bold", fontsize=8, color="black")
    outcome_idx = list(df.columns).index("outcome") if "outcome" in df.columns else None
    for i in range(nrows):
        row_ok = outcome_idx is not None and "SUCCESS" in str(cells[i][outcome_idx])
        for j in range(ncols):
            cell = tbl[(i + 1, j)]
            if row_ok:
                cell.set_facecolor("#ececec")
                cell.set_linewidth(1.2)
            elif i % 2 == 0:
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#f4f4f4")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"table_{slug}.png", bbox_inches="tight", facecolor="white", dpi=150)
        fig.savefig(d / f"table_{slug}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not GROUPED_DIR.is_dir():
        raise SystemExit(f"Run build_thesis_grouped_tables.py first — missing {GROUPED_DIR}")

    jobs = [
        ("by_dataset_coco20k.csv", "COCO-20k experiments", "dataset_coco20k"),
        ("by_dataset_coco100k.csv", "COCO-100k experiments", "dataset_coco100k"),
        ("by_dataset_coco240k.csv", "COCO ~240k experiments", "dataset_coco240k"),
        ("by_4experts.csv", "4 experts (all datasets)", "4experts"),
        ("by_8experts.csv", "8 experts (all datasets)", "8experts"),
        ("by_topk_1.csv", "top-k = 1 (sparse)", "topk_1"),
        ("by_topk_2.csv", "top-k = 2", "topk_2"),
        ("by_topk_3.csv", "top-k = 3", "topk_3"),
        ("by_topk_4.csv", "top-k = 4 (dense)", "topk_4"),
        ("coco100k_unfrozen.csv", "COCO-100k — unfrozen backbone", "coco100k_unfrozen"),
        ("coco100k_frozen.csv", "COCO-100k — frozen backbone", "coco100k_frozen"),
        ("coco100k_all_sorted.csv", "COCO-100k — all runs (BER sorted)", "coco100k_sorted"),
        ("coco20k_all_sorted.csv", "COCO-20k — all runs (BER sorted)", "coco20k_sorted"),
    ]

    # Remove old chart-style figures so this folder is tables only
    for d in OUT:
        if d.is_dir():
            for old in d.glob("viz_*.*"):
                old.unlink(missing_ok=True)

    made = []
    for filename, title, slug in jobs:
        path = GROUPED_DIR / filename
        if not path.is_file():
            continue
        df = df_for_table(path)
        if df.empty:
            continue
        render_table_figure(df, title, slug)
        made.append(slug)

    readme = (
        "# Grouped table figures\n\n"
        "Visual copies of `results/thesis_tables_grouped/*.csv`.\n\n"
        "Regenerate: `python scripts/plot_thesis_grouped_tables.py`\n\n"
        + "\n".join(f"- `table_{s}.png`" for s in made)
    )
    for d in OUT:
        (d / "README.md").write_text(readme, encoding="utf-8")

    print(f"Rendered {len(made)} table figures → {OUT[0]}")


if __name__ == "__main__":
    main()
