#!/usr/bin/env python3
"""
Generates thesis table figures with clean styling and vertical column headers.
Outputs to:
  thesis/figures/tables/        (live path used by main.tex)
  thesis/figures/new_figures/   (review copy — all freshly generated files)

Updated figures:
  01_clean_channel_fidelity.png
  02_attack_robustness_r0_vs_hidden.png  (bar chart, vertical x-axis labels)
  03_backbone_comparison_b128.png
  08_full_attack_comparison.png  (14-attack BER + Δ BitAcc table)

Usage (from repo root):
    py -3 scripts/plot_thesis_tables_new.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "thesis" / "figures" / "tables"
OUT_NEW = ROOT / "thesis" / "figures" / "new_figures"
OUT.mkdir(parents=True, exist_ok=True)
OUT_NEW.mkdir(parents=True, exist_ok=True)

ATTACK_MAIN = ROOT / "results" / "comparison_hidden_vs_moe" / "attack_summary_main_ep20_hard_top1.csv"
ATTACK_DELTA_PP = 1.0   # win/loss colouring threshold for figure 02


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save to both tables/ and new_figures/ with high DPI."""
    for dest in (OUT, OUT_NEW):
        path = dest / name
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"  Saved: {name}")

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size":   8,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  14-attack BER + Δ BitAcc table  (figure 08)
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_COMBINED = ROOT / "results" / "benchmark_hidden" / "results_combined.csv"
BENCHMARK_VARIANTS = ROOT / "results" / "benchmark_hidden_variants"
BENCHMARK_WAVES = ROOT / "results" / "benchmark_moe_variants" / "results_combined_all.csv"
HIDDEN_METHOD = "HiDDeN-ep177"
MOE_METHOD = "MoE-R0-soft"
DELTA_WIN = 0.015   # 1.5 pp on bit-accuracy scale
HIDDEN_NOISE_ATTACKS = [
    "identity", "jpeg", "quant", "crop", "cropout", "dropout", "resize", "gaussian", "combined",
]
WAVES_EXTRA_ATTACKS = ["rotation", "resized_crop", "erasing", "brightness", "contrast"]

ATTACKS_14 = [
    "identity", "jpeg", "quant", "crop", "cropout", "dropout", "resize",
    "gaussian", "combined", "rotation", "resized_crop", "erasing",
    "brightness", "contrast",
]

ATTACK_LABELS_14 = {
    "identity": "Identity",
    "jpeg": "JPEG",
    "quant": "Quant",
    "crop": "Crop",
    "cropout": "Cropout",
    "dropout": "Dropout",
    "resize": "Resize",
    "gaussian": "Gaussian",
    "combined": "Combined",
    "rotation": "Rotation",
    "resized_crop": "Resized crop",
    "erasing": "Erasing",
    "brightness": "Brightness",
    "contrast": "Contrast",
}

def _load_hidden_bit_acc_14() -> dict[str, float]:
    if not BENCHMARK_COMBINED.is_file():
        raise FileNotFoundError(BENCHMARK_COMBINED)
    hidden: dict[str, float] = {}
    for _, r in pd.read_csv(BENCHMARK_COMBINED).iterrows():
        if r["method"] == HIDDEN_METHOD:
            hidden[str(r["attack"])] = float(r["bit_accuracy"])
    return hidden


def _load_waves_bit_acc() -> dict[tuple[str, str], float]:
    if not BENCHMARK_WAVES.is_file():
        return {}
    out: dict[tuple[str, str], float] = {}
    for _, r in pd.read_csv(BENCHMARK_WAVES).iterrows():
        out[(str(r["method"]), str(r["attack"]))] = float(r["bit_accuracy"])
    return out


def _load_variant_hidden_benchmark(method: str) -> dict[str, float] | None:
    """Per-variant full 14-attack hidden benchmark if run via run_hidden_benchmark_batch.py."""
    safe = method.replace("/", "_")
    path = BENCHMARK_VARIANTS / f"results_{safe}.csv"
    if not path.is_file():
        return None
    moe: dict[str, float] = {}
    for _, r in pd.read_csv(path).iterrows():
        if str(r["method"]) == method:
            moe[str(r["attack"])] = float(r["bit_accuracy"])
    return moe or None


def _load_moe_bit_acc_14(
    method: str,
    attack_csv: Path | None,
    waves_method: str | None,
) -> dict[str, float]:
    full = _load_variant_hidden_benchmark(method)
    if full and len(full) >= len(ATTACKS_14):
        return full

    moe: dict[str, float] = {}
    if method == MOE_METHOD and BENCHMARK_COMBINED.is_file():
        for _, r in pd.read_csv(BENCHMARK_COMBINED).iterrows():
            if r["method"] == MOE_METHOD:
                moe[str(r["attack"])] = float(r["bit_accuracy"])
        if len(moe) >= len(ATTACKS_14):
            return moe

    if attack_csv and attack_csv.is_file():
        for _, r in pd.read_csv(attack_csv).iterrows():
            moe[str(r["attack"])] = float(r["moe_bit_acc"])

    waves = _load_waves_bit_acc()
    wm = waves_method or method
    for atk in WAVES_EXTRA_ATTACKS:
        key = (wm, atk)
        if key in waves:
            moe[atk] = waves[key]
    return moe


def _delta_bg(delta: float) -> str:
    if delta > DELTA_WIN:
        return "#c8e6c9"
    if delta < -DELTA_WIN:
        return "#ffcdd2"
    return "#fff9c4"


def _build_attack_table_rows(
    hidden: dict[str, float],
    moe: dict[str, float],
) -> tuple[list[list[str]], list[list[str]], int, int, int]:
    rows: list[list[str]] = []
    row_bgs: list[list[str]] = []
    wins = losses = ties = 0
    for atk in ATTACKS_14:
        if atk not in hidden or atk not in moe:
            continue
        h_ba, m_ba = hidden[atk], moe[atk]
        h_ber, m_ber = (1.0 - h_ba) * 100.0, (1.0 - m_ba) * 100.0
        delta = m_ba - h_ba
        if delta > DELTA_WIN:
            wins += 1
        elif delta < -DELTA_WIN:
            losses += 1
        else:
            ties += 1
        rows.append([
            atk,
            f"{h_ber:.2f}%",
            f"{m_ber:.2f}%",
            f"{delta:+.3f}",
        ])
        row_bgs.append(["white", "white", "white", _delta_bg(delta)])
    return rows, row_bgs, wins, losses, ties


# (label, LaTeX label slug)
EXPERIMENT_ATTACK_META: list[tuple[str, str, str, Path | None, str | None]] = [
    ("R0 primary (ep20)", "r0", "MoE-R0-soft", None, None),
    ("Top-$k{=}2$ (ep30)", "top2", "MoE-top2-ep30",
     ROOT / "results/comparison_hidden_vs_moe/attack_summary_top2_ep20_soft_router.csv",
     "MoE-top2-ep30"),
    ("Dense $k{=}4$ (ep20)", "dense", "MoE-dense-k4-ep20",
     ROOT / "results/comparison_hidden_vs_moe/attack_summary_dense_4exp_k4_ep20_soft_router.csv",
     "MoE-dense-k4-ep20"),
    ("8 experts run A (ep20)", "8exp-a", "MoE-8exp-A-ep20",
     ROOT / "results/comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20_soft_router.csv",
     "MoE-8exp-A-ep20"),
    ("8 experts run B (ep15)", "8exp-b", "MoE-8exp-B-ep15",
     ROOT / "results/comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15_soft_router.csv",
     "MoE-8exp-B-ep15"),
    ("Frozen top-$k{=}1$ (ep30)", "frozen", "MoE-frozen-4exp-ep30",
     ROOT / "results/ber_500_attacks/attack_summary_frozen_ep30_soft_router.csv",
     "MoE-frozen-4exp-ep16"),
    ("Attack-trained (ep30)", "attack", "MoE-attack-ep30",
     ROOT / "results/comparison_hidden_vs_moe/attack_summary_attack_v1_ep30_soft_router.csv",
     "MoE-attack-ep65"),
]

LATEX_OUT = ROOT / "thesis" / "generated" / "attack_comparison_14.tex"

HDR_BG = "#2c5f7a"
GROUP_BG = "#1a1a2e"


def _draw_styled_attack_table(
    ax,
    *,
    x0: float,
    y_top: float,
    width: float,
    title: str,
    rows: list[list[str]],
    row_bgs: list[list[str]],
    wins: int,
    losses: int,
    ties: int,
    group_h: float = 0.34,
    hdr_h: float = 0.46,
    row_h: float = 0.30,
    title_h: float = 0.34,
    note_h: float = 0.22,
) -> float:
    """Full-width attack table (routing-table style). Returns y bottom."""
    n_rows = len(rows)
    col_widths = [0.26, 0.22, 0.22, 0.30]
    col_x = [x0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w * width)
    col_w = [w * width for w in col_widths]

    title_top = y_top
    group_top = title_top - title_h
    hdr_top = group_top - group_h
    body_top = hdr_top - hdr_h
    body_bot = body_top - n_rows * row_h
    foot_bot = body_bot - note_h

    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, foot_bot), width, title_top - foot_bot,
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#222222", facecolor="none", zorder=3,
    ))

    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, group_top), width, title_h,
        boxstyle="square,pad=0", linewidth=0, facecolor=GROUP_BG, edgecolor="none",
    ))
    ax.text(
        x0 + 0.06, group_top + title_h / 2, title,
        ha="left", va="center", fontsize=8.2, fontweight="bold", color="white",
    )
    ax.text(
        x0 + width - 0.06, group_top + title_h / 2,
        f"MoE {wins}W / {losses}L / {ties}T",
        ha="right", va="center", fontsize=7.5, color="#dddddd",
    )

    # Group headers over HiDDeN / MoE / Comparison columns
    group_specs = [
        ("HiDDeN", col_x[1], col_w[1]),
        ("MoE", col_x[2], col_w[2]),
        ("Comparison", col_x[3], col_w[3]),
    ]
    for glabel, gx, gw in group_specs:
        ax.add_patch(mpatches.FancyBboxPatch(
            (gx, group_top - group_h), gw, group_h,
            boxstyle="square,pad=0", linewidth=0, facecolor=GROUP_BG, edgecolor="none",
        ))
        ax.text(
            gx + gw / 2, group_top - group_h / 2, glabel,
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="white",
        )

    metric_labels = ["Attack", "BER (%)", "BER (%)", r"$\Delta$ BitAcc"]
    for ci, (label, cx, cw) in enumerate(zip(metric_labels, col_x, col_w)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, body_top), cw, hdr_h,
            boxstyle="square,pad=0", linewidth=0, facecolor=HDR_BG, edgecolor="none",
        ))
        ax.text(
            cx + cw / 2, body_top + hdr_h / 2, label,
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="white",
        )

    for ri, (row_vals, bgs) in enumerate(zip(rows, row_bgs)):
        ry = body_top - (ri + 1) * row_h
        for ci, (val, cx, cw, bg) in enumerate(zip(row_vals, col_x, col_w, bgs)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, ry), cw, row_h,
                boxstyle="square,pad=0", linewidth=0, facecolor=bg, edgecolor="none",
            ))
            ax.text(
                cx + (0.06 * cw if ci == 0 else cw / 2),
                ry + row_h / 2, val,
                ha="left" if ci == 0 else "center", va="center",
                fontsize=7.8, fontweight="bold" if ci == 0 else "normal",
            )

    for ri in range(n_rows + 1):
        y = body_top - ri * row_h
        ax.plot([x0, x0 + width], [y, y], color="#555555", linewidth=1.0 if ri == 0 else 0.35, zorder=2)
    for boundary in [x0, col_x[1], col_x[2], col_x[3], x0 + width]:
        ax.plot([boundary, boundary], [foot_bot, title_top], color="#555555", linewidth=0.9, zorder=2)
    ax.plot([x0, x0 + width], [group_top, group_top], color="#555555", linewidth=0.9, zorder=2)
    ax.plot([x0, x0 + width], [hdr_top, hdr_top], color="#555555", linewidth=0.9, zorder=2)

    return foot_bot


def _collect_attack_panels(hidden: dict[str, float]) -> list[dict]:
    panels: list[dict] = []
    for label, slug, method, attack_csv, waves_method in EXPERIMENT_ATTACK_META:
        moe = _load_moe_bit_acc_14(method, attack_csv, waves_method)
        rows, row_bgs, wins, losses, ties = _build_attack_table_rows(hidden, moe)
        if not rows:
            print(f"  WARNING figure 08: no rows for {label}")
            continue
        panels.append({
            "label": label,
            "slug": slug,
            "rows": rows,
            "row_bgs": row_bgs,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        })
    return panels


# Short column headers for the combined 14-attack survey table
VARIANT_COL_HEADERS: list[tuple[str, str]] = [
    ("R0 ep20", "r0"),
    ("Top-$k{=}2$", "top2"),
    ("Dense $k{=}4$", "dense"),
    ("8exp-A", "8exp-a"),
    ("8exp-B", "8exp-b"),
    ("Frozen", "frozen"),
    ("Atk-tr ep30", "attack"),
]


def write_attack_comparison_latex(panels: list[dict]) -> None:
    """Emit one combined booktabs table: attacks × BER (%) per method."""
    LATEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not panels:
        return

    by_slug = {p["slug"]: p for p in panels}
    active_variants = [(hdr, slug) for hdr, slug in VARIANT_COL_HEADERS if slug in by_slug]
    n_cols = 1 + 1 + len(active_variants)  # attack + HiDDeN + MoE variants
    # Tight gap between Attack and HiDDeN; normal spacing for method columns
    col_spec = "@{}l@{\\hskip 0.45em}r" + " r" * (n_cols - 2) + "@{}"

    header = "Attack & HiDDeN" + "".join(f" & {name}" for name, _ in active_variants) + " \\\\\n"
    lines: list[str] = [
        "% Auto-generated by scripts/plot_thesis_tables_new.py -- do not edit by hand.\n",
        "\\begin{table}[H]\n\\centering\n\\small\n",
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n",
        header,
        "\\midrule\n",
    ]

    ref_rows = panels[0]["rows"]
    for atk, h_ber, _m0, _d0 in ref_rows:
        atk_tex = atk.replace("_", r"\_")
        cells = [atk_tex, h_ber.replace("%", "")]
        for _hdr, slug in active_variants:
            panel = by_slug[slug]
            row_map = {r[0]: r for r in panel["rows"]}
            if atk not in row_map:
                cells.append("---")
                continue
            cells.append(row_map[atk][2].replace("%", ""))
        lines.append(" & ".join(cells) + " \\\\\n")

    lines.append(
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Full 14-attack benchmark: bit error rate (\\%) per method and attack. "
        "HiDDeN and R0 use 1{,}000 images/attack (\\texttt{benchmark\\_hidden}); "
        "other MoE variants combine nine HiDDeN-noise soft-router evals with five WAVES PIL extras. "
        "Soft router at inference for all MoE columns.}\n"
        "\\label{tab:attack-14-all}\n\\end{table}\n"
    )

    LATEX_OUT.write_text("".join(lines), encoding="utf-8")
    print(f"  Wrote: {LATEX_OUT.relative_to(ROOT)}")
    patch_script = ROOT / "scripts" / "patch_combined_attack_table.py"
    if patch_script.is_file():
        import subprocess
        subprocess.run([sys.executable, str(patch_script)], check=False, cwd=ROOT)


def plot_attack_ber_delta_table() -> None:
    """Figure 08 — stacked full-width BER + Δ tables (terminal layout, routing-table style)."""
    try:
        hidden = _load_hidden_bit_acc_14()
    except FileNotFoundError as exc:
        print(f"  SKIP figure 08: {exc}")
        return

    panels = _collect_attack_panels(hidden)
    if not panels:
        print("  SKIP figure 08: no experiment panels")
        return

    write_attack_comparison_latex(panels)

    fw = 8.8
    title_h, group_h, hdr_h, row_h, note_h, gap = 0.34, 0.34, 0.46, 0.30, 0.22, 0.28
    block_h = title_h + group_h + hdr_h + 14 * row_h + note_h
    fh = len(panels) * (block_h + gap) + 0.55

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")

    y = fh - 0.15
    for panel in panels:
        y = _draw_styled_attack_table(
            ax,
            x0=0.0,
            y_top=y,
            width=fw,
            title=panel["label"],
            rows=panel["rows"],
            row_bgs=panel["row_bgs"],
            wins=panel["wins"],
            losses=panel["losses"],
            ties=panel["ties"],
            group_h=group_h,
            hdr_h=hdr_h,
            row_h=row_h,
            title_h=title_h,
            note_h=note_h,
        )
        y -= gap

    leg_y = 0.10
    for px, bg, lbl in [
        (0.08, "#c8e6c9", "MoE wins ($>1.5$ pp)"),
        (2.55, "#fff9c4", "Near tie ($\\pm 1.5$ pp)"),
        (5.05, "#ffcdd2", "HiDDeN wins ($>1.5$ pp)"),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, leg_y), 0.20, 0.10,
            boxstyle="round,pad=0.02", linewidth=0.4,
            facecolor=bg, edgecolor="#555555",
        ))
        ax.text(px + 0.24, leg_y + 0.05, lbl, ha="left", va="center", fontsize=7.0)

    save_fig(fig, "08_full_attack_comparison.png")


def plot_waves_heatmap() -> None:
    """Backward-compatible alias."""
    plot_attack_ber_delta_table()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Backbone comparison  (figure 03)
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_COLS = [
    "Run", "Backbone",
    "BER ep20 (%)", "BER final (%)",
    "Final epoch", "max_use",
]

BACKBONE_ROWS = [
    ["R0 (reported)",     "unfrozen", "0.32", "0.32", "20", "0.306"],
    ["Frozen sym_bal08",  "frozen",   "2.72", "7.04", "30", "0.638"],
]

# row colours: unfrozen = light green, frozen = light salmon
ROW_COLORS = ["#d6edd5", "#f5d0c5"]
HDR_COL_BG = "#2c5f7a"


def plot_backbone_table() -> None:
    N_COLS  = len(BACKBONE_COLS)
    N_ROWS  = len(BACKBONE_ROWS)

    # Column widths (relative, total = 1)
    COL_WIDTHS = [0.285, 0.13, 0.14, 0.14, 0.13, 0.175]
    assert abs(sum(COL_WIDTHS) - 1.0) < 1e-3

    FW   = 7.6
    HDR_H = 0.44
    ROW_H = 0.38
    NOTE_H = 0.34
    FH    = HDR_H + N_ROWS * ROW_H + NOTE_H + 0.08

    fig = plt.figure(figsize=(FW, FH))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")

    col_x = [0.0]
    for w in COL_WIDTHS[:-1]:
        col_x.append(col_x[-1] + w * FW)
    col_w = [w * FW for w in COL_WIDTHS]

    BODY_TOP = FH - HDR_H
    BODY_BOT = BODY_TOP - N_ROWS * ROW_H

    # ── Outer border ──────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, BODY_BOT - NOTE_H), FW, FH - (BODY_BOT - NOTE_H),
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#222222", facecolor="none", zorder=3
    ))

    # ── Column headers ────────────────────────────────────────────────────────
    for ci, (label, cx, cw) in enumerate(zip(BACKBONE_COLS, col_x, col_w)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, BODY_TOP), cw, HDR_H,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=HDR_COL_BG, edgecolor="none"
        ))
        fs = 7.4 if ci >= 2 else 8.0
        ax.text(
            cx + cw / 2, BODY_TOP + HDR_H / 2,
            label, ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color="white", linespacing=1.15,
        )

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, (row_vals, bg) in enumerate(zip(BACKBONE_ROWS, ROW_COLORS)):
        ry = BODY_TOP - (ri + 1) * ROW_H
        for ci, (val, cx, cw) in enumerate(zip(row_vals, col_x, col_w)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, ry), cw, ROW_H,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, edgecolor="none"
            ))
            fw = "bold" if ci <= 1 else "normal"
            ax.text(
                cx + cw / 2, ry + ROW_H / 2, val,
                ha="center", va="center",
                fontsize=8.5, fontweight=fw,
            )

    # ── Grid lines ────────────────────────────────────────────────────────────
    # horizontal
    for ri in range(N_ROWS + 1):
        y  = BODY_TOP - ri * ROW_H
        lw = 1.0 if ri == 0 else 0.4
        ax.plot([0, FW], [y, y], color="#555555", linewidth=lw, zorder=2)

    # vertical
    for ci in range(N_COLS + 1):
        x  = col_x[ci] if ci < N_COLS else FW
        lw = 0.9 if ci in (0, N_COLS) else 0.4
        ax.plot([x, x], [BODY_BOT, FH], color="#555555", linewidth=lw, zorder=2)

    note_top = BODY_BOT - NOTE_H
    title_y = note_top + NOTE_H * 0.72
    legend_y = note_top + NOTE_H * 0.28
    ax.text(
        FW / 2, title_y,
        "Backbone freezing ablation ($N{=}4$, $k{=}1$, batch 128, COCO-100k)",
        ha="center", va="center", fontsize=7.2, fontweight="bold", color="#222222",
    )
    legend_items = [
        ("Unfrozen backbone (R0)", ROW_COLORS[0]),
        ("Frozen backbone", ROW_COLORS[1]),
    ]
    patch_w, patch_h, item_gap = 0.22, 0.10, 0.55
    text_offsets = [2.05, 1.45]
    block_w = sum(text_offsets) + patch_w * 2 + item_gap
    block_x0 = (FW - block_w) / 2
    px = block_x0
    for (label, bg), text_off in zip(legend_items, text_offsets):
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, legend_y - patch_h / 2), patch_w, patch_h,
            boxstyle="round,pad=0.02", linewidth=0.5,
            facecolor=bg, edgecolor="#555555",
        ))
        ax.text(px + patch_w + 0.06, legend_y, label,
                ha="left", va="center", fontsize=6.8)
        px += text_off + patch_w + item_gap

    for dest in (OUT, OUT_NEW):
        path = dest / "03_backbone_comparison_b128.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200, pad_inches=0.12)
    plt.close(fig)
    print("  Saved: 03_backbone_comparison_b128.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Clean-channel fidelity table  (figure 01)  — vertical column headers
# ─────────────────────────────────────────────────────────────────────────────

# Transposed: metrics as ROWS, models as COLUMNS (tall/narrow layout)
CC_HDR_COLS = ["Metric", "HiDDeN ep177", "MoE R0 (unfrozen)"]
CC_HDR_BG   = ["#2c5f7a", "#4a6f8a", "#3d7a52"]   # teal / blue-grey / green
CC_DATA_ROWS = [
    ("Train images",   "≈ 10k",           "≈ 118k"),
    ("BER (%)",        "0.33",            "0.32"),
    ("Bit Acc. (%)",   "99.67",           "99.68"),
    ("Enc. MSE",       "0.0026",          "0.0026"),
    ("PSNR (dB)",      "≈ 35",            "≈ 34"),
    ("Epochs",         "177",             "20"),
]
CC_COL_DATA_BG = ["#eef2f6", "#e8f4ec"]   # HiDDeN col / MoE col


def plot_clean_channel_table() -> None:
    n_cols = len(CC_HDR_COLS)
    n_rows = len(CC_DATA_ROWS)

    col_widths = [0.34, 0.33, 0.33]
    fw = 5.0
    hdr_h = 0.50
    row_h = 0.38
    note_h = 0.0
    fh = hdr_h + n_rows * row_h + 0.06

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")

    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w * fw)
    col_w = [w * fw for w in col_widths]

    body_top = fh - hdr_h
    body_bot = body_top - n_rows * row_h

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, body_bot), fw, fh - body_bot,
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#222222", facecolor="none", zorder=3,
    ))

    # column headers (models across the top)
    for ci, (label, cx, cw, bg) in enumerate(zip(CC_HDR_COLS, col_x, col_w, CC_HDR_BG)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, body_top), cw, hdr_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg, edgecolor="none",
        ))
        ax.text(
            cx + cw / 2, body_top + hdr_h / 2, label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white",
        )

    # data rows (metrics down the left)
    for ri, (metric, hidden_val, moe_val) in enumerate(CC_DATA_ROWS):
        ry = body_top - (ri + 1) * row_h
        row_vals = [metric, hidden_val, moe_val]
        row_bgs  = ["#d8e4ec", CC_COL_DATA_BG[0], CC_COL_DATA_BG[1]]

        for ci, (val, cx, cw, bg) in enumerate(zip(row_vals, col_x, col_w, row_bgs)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, ry), cw, row_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, edgecolor="none",
            ))
            fw_text = "bold" if ci == 0 else "normal"
            ax.text(
                cx + cw / 2, ry + row_h / 2, val,
                ha="center", va="center",
                fontsize=8.5, fontweight=fw_text,
            )

    for ri in range(n_rows + 1):
        y = body_top - ri * row_h
        ax.plot([0, fw], [y, y], color="#555555", linewidth=1.0 if ri == 0 else 0.4, zorder=2)
    for ci in range(n_cols + 1):
        x = col_x[ci] if ci < n_cols else fw
        ax.plot([x, x], [body_bot, fh], color="#555555", linewidth=0.9 if ci in (0, n_cols) else 0.4, zorder=2)

    save_fig(fig, "01_clean_channel_fidelity.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Attack robustness table  (figure 02)  — numeric table, colour-coded Δ
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_BLUE = "#5B7FA5"
MOE_GREEN   = "#2E8B57"
MOE_LOSS    = "#C0392B"
NEAR_TIE    = "#B0B0B0"


def _load_attack_df() -> pd.DataFrame:
    df = pd.read_csv(ATTACK_MAIN)
    df = df[df["attack"] != "combined"].copy()
    order = ["identity", "jpeg", "quant", "crop", "cropout", "dropout", "resize", "gaussian"]
    df["attack"] = pd.Categorical(df["attack"], categories=order, ordered=True)
    return df.sort_values("attack")


def plot_attack_robustness_table() -> None:
    if not ATTACK_MAIN.is_file():
        print(f"  SKIP figure 02 table: {ATTACK_MAIN} not found")
        return

    df = _load_attack_df()
    cols = ["Attack", "HiDDeN BA (%)", "MoE R0 BA (%)", "Δ MoE−HiDDeN (pp)"]
    rows = []
    row_bgs = []

    for _, r in df.iterrows():
        h = r["hidden_bit_acc"] * 100
        m = r["moe_bit_acc"] * 100
        d = r["moe_minus_hidden_acc"] * 100
        d_str = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
        if d > ATTACK_DELTA_PP:
            delta_bg = "#c8e6c9"
        elif d < -ATTACK_DELTA_PP:
            delta_bg = "#ffcdd2"
        else:
            delta_bg = "#fff9c4"
        rows.append([r["attack"].capitalize(), f"{h:.2f}", f"{m:.2f}", d_str])
        row_bgs.append(["white", "white", "white", delta_bg])

    n_rows = len(rows)
    n_cols = len(cols)
    col_widths = [0.20, 0.24, 0.24, 0.32]
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    fw = 8.0
    hdr_h = 0.48
    row_h = 0.36
    note_h = 0.22
    fh = hdr_h + n_rows * row_h + note_h + 0.28

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")

    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w * fw)
    col_w = [w * fw for w in col_widths]
    body_top = fh - hdr_h
    body_bot = body_top - n_rows * row_h

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, body_bot), fw, fh - body_bot,
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#222222", facecolor="none", zorder=3,
    ))

    for ci, (label, cx, cw) in enumerate(zip(cols, col_x, col_w)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, body_top), cw, hdr_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor="#2c5f7a", edgecolor="none",
        ))
        ax.text(
            cx + cw / 2, body_top + hdr_h / 2, label,
            ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="white",
        )

    for ri, (row_vals, bgs) in enumerate(zip(rows, row_bgs)):
        ry = body_top - (ri + 1) * row_h
        for ci, (val, cx, cw, bg) in enumerate(zip(row_vals, col_x, col_w, bgs)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, ry), cw, row_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, edgecolor="none",
            ))
            ax.text(cx + cw / 2, ry + row_h / 2, val,
                    ha="center", va="center", fontsize=8.0)

    for ri in range(n_rows + 1):
        y = body_top - ri * row_h
        ax.plot([0, fw], [y, y], color="#555555", linewidth=1.0 if ri == 0 else 0.4, zorder=2)
    for ci in range(n_cols + 1):
        x = col_x[ci] if ci < n_cols else fw
        ax.plot([x, x], [body_bot, fh], color="#555555", linewidth=0.9 if ci in (0, n_cols) else 0.4, zorder=2)

    legend_y = body_bot - 0.20
    for px, bg, label in [
        (0.08, "#c8e6c9", "MoE wins (>1 pp)"),
        (2.05, "#fff9c4", "Near tie (±1 pp)"),
        (4.05, "#ffcdd2", "HiDDeN wins (>1 pp)"),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, legend_y), 0.20, 0.10,
            boxstyle="round,pad=0.02", linewidth=0.5,
            facecolor=bg, edgecolor="#555555",
        ))
        ax.text(px + 0.24, legend_y + 0.05, label, ha="left", va="center", fontsize=6.8)

    ax.text(
        0.08, body_bot - 0.14,
        "HiDDeN ep177 vs MoE R0 (hard top-1 router). Nine HiDDeN-noise attacks; 10,000 images/attack.",
        ha="left", va="top", fontsize=6.5, style="italic", color="#444444",
    )

    save_fig(fig, "02_attack_robustness_r0_vs_hidden.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RQ2 attack robustness bar chart  — one figure, clear model colours
# ─────────────────────────────────────────────────────────────────────────────

def plot_rq2_bar_chart() -> None:
    if not ATTACK_MAIN.is_file():
        print(f"  SKIP RQ2 chart: {ATTACK_MAIN} not found")
        return

    df = _load_attack_df()
    # bar chart order: easy-to-read grouping (clean → distortions → hard)
    chart_order = ["identity", "quant", "dropout", "cropout", "crop", "resize", "gaussian", "jpeg"]
    df = df.set_index("attack").loc[chart_order].reset_index()

    h_acc = df["hidden_bit_acc"].values * 100
    m_acc = df["moe_bit_acc"].values * 100
    delta = df["moe_minus_hidden_acc"].values * 100
    n_win  = int((delta >  0.5).sum())
    n_loss = int((delta < -0.5).sum())
    n_tie  = len(delta) - n_win - n_loss

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))

    x = np.arange(len(df))
    w = 0.36

    bars_h = axes[0].bar(
        x - w / 2, h_acc, w,
        label="HiDDeN ep177 (baseline)",
        color=HIDDEN_BLUE, edgecolor="black", linewidth=0.8,
    )
    bars_m = axes[0].bar(
        x + w / 2, m_acc, w,
        label="MoE R0 ep20 (this work)",
        color=MOE_GREEN, edgecolor="black", linewidth=0.8,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["attack"].str.upper(), rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Bit accuracy (%)")
    axes[0].set_ylim(0, 108)
    axes[0].axhline(50, color="0.5", linestyle=":", linewidth=1)
    axes[0].legend(
        handles=[bars_h, bars_m],
        labels=["HiDDeN ep177 (baseline)", "MoE R0 ep20 (this work)"],
        frameon=True, edgecolor="black", fontsize=7.5, loc="upper right",
    )
    axes[0].set_title("(a) Absolute bit accuracy per attack", fontweight="bold", pad=10)
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].tick_params(axis="x", pad=2)

    delta_colors = [
        MOE_GREEN if d > 0.5 else (MOE_LOSS if d < -0.5 else NEAR_TIE)
        for d in delta
    ]
    axes[1].barh(x, delta, color=delta_colors, edgecolor="black", linewidth=0.8, height=0.62)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(df["attack"].str.upper(), fontsize=8)
    axes[1].set_xlabel("MoE − HiDDeN bit accuracy (pp)")
    axes[1].set_title("(b) Signed advantage (green = MoE better)", fontweight="bold", pad=10)
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", linestyle=":", alpha=0.6)
    pad = 0.45
    x_lo = min(float(delta.min()) - 1.2, -1.0)
    x_hi = float(delta.max()) + 1.2
    axes[1].set_xlim(x_lo, x_hi)
    for i, d in enumerate(delta):
        if d >= 0:
            axes[1].text(d + pad, i, f"{d:+.1f}", va="center", ha="left", fontsize=7.5)
        else:
            axes[1].text(
                max(d - pad, x_lo + 0.25), i, f"{d:+.1f}",
                va="center", ha="right", fontsize=7.5,
            )

    legend_patches = [
        mpatches.Patch(facecolor=MOE_GREEN, edgecolor="black", linewidth=0.6, label="MoE win"),
        mpatches.Patch(facecolor=NEAR_TIE, edgecolor="black", linewidth=0.6, label="Near tie"),
        mpatches.Patch(facecolor=MOE_LOSS, edgecolor="black", linewidth=0.6, label="HiDDeN win"),
    ]
    weakness = "JPEG" if n_loss == 1 else f"{n_loss} attacks"
    fig.suptitle(
        f"RQ2: MoE wins on {n_win}/8 attacks; {weakness} is the main weakness",
        fontweight="bold",
        y=0.99,
    )
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=3,
        fontsize=7.5,
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.91),
        columnspacing=1.2,
        handlelength=1.2,
    )
    fig.text(
        0.5, 0.01,
        f"Panel (a): side-by-side model comparison.  Panel (b): same data as signed gaps.  "
        f"Wins (Δ>0.5 pp): {n_win}  ·  Losses: {n_loss}  ·  Near tie: {n_tie}",
        ha="center", fontsize=7.5, style="italic",
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.38, left=0.07, right=0.98)
    save_fig(fig, "RQ2_attack_robustness.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Cross-paradigm comparison: DCT, SSL, Tree-Ring, ROBIN
# ─────────────────────────────────────────────────────────────────────────────

WAVES_ROOT = ROOT.parent.parent / "waves"   # D:\waves

# Only attacks present in ALL three post-hoc datasets
CROSS_ATTACKS = ["identity", "jpeg_q50", "crop", "gaussian", "combined"]
ATTACK_LABELS  = ["Identity", "JPEG Q50", "Crop 85%", "Gaussian\nnoise", "Combined"]

# Generation-based smoke data (2 images) – for AUROC comparison only
_TREERING_AUROC_SMOKE = {
    "identity": 1.0, "jpeg_q50": 1.0, "crop": 1.0, "gaussian": 1.0, "combined": 1.0,
}
_ROBIN_AUROC_SMOKE = {
    "identity": 1.0, "jpeg_q50": 1.0, "crop": 1.0, "gaussian": 1.0, "combined": 0.25,
}


def _load_csv_indexed(path: Path, key_col: str = "attack") -> dict:
    """Return {attack: row_dict} for a results CSV."""
    import csv as _csv
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            out[row[key_col]] = row
    return out


def plot_cross_paradigm_comparison() -> None:
    """
    Color-coded attack table: all 14 WAVES attacks × 2 post-hoc methods.
    Columns: Attack | DCT BER (%) | SSL BER (%)
    DCT has results only for 5 of the 14 attacks; remaining cells show '—'.
    """
    import matplotlib.colors as mcolors

    dct_csv = WAVES_ROOT / "benchmark_simple_output" / "dct" / "results.csv"
    ssl_csv = WAVES_ROOT / "ssl"  / "outputs_benchmark" / "results.csv"

    for p in (dct_csv, ssl_csv):
        if not p.is_file():
            print(f"  SKIP cross-paradigm: {p} not found"); return

    dct  = _load_csv_indexed(dct_csv)
    ssl  = _load_csv_indexed(ssl_csv)

    # all 14 WAVES attacks in display order
    ATTACKS_14 = [
        ("identity",        "Identity (no attack)"),
        ("rotation",        "Rotation 22.5°"),
        ("resized_crop",    "Resized crop ×0.75"),
        ("erasing",         "Random erasing 12.5%"),
        ("brightness",      "Brightness ×1.5"),
        ("contrast",        "Contrast ×1.5"),
        ("blur",            "Gaussian blur r=4"),
        ("resize_90",       "Resize 90% → back"),
        ("jpeg_q50",        "JPEG Q50"),
        ("crop",            "Crop 85% → back"),
        ("gaussian",        "Gaussian noise σ=25"),
        ("combo_geometric", "Combo: geo"),
        ("combo_photometric","Combo: photo"),
        ("combined",        "Combined (crop+JPEG+noise)"),
    ]

    # ── build cell data ───────────────────────────────────────────────────
    # value = float or None (→ "—")
    rows = []
    for atk, desc in ATTACKS_14:
        d_ber = (1.0 - float(dct[atk]["bit_accuracy"])) * 100 if atk in dct else None
        s_ber = (1.0 - float(ssl[atk]["bit_accuracy"])) * 100
        rows.append((desc, d_ber, s_ber))

    # ── layout ────────────────────────────────────────────────────────────
    COL_LABELS  = ["Attack", "BER (%)", "BER (%)"]
    GROUP_HDRS  = [
        ("Attack",               1, "#4A4A4A"),
        ("DCT (Cox)\nn=100",     1, "#B85C1A"),
        ("SSL Watermark\nn=100",  1, "#2C5F8A"),
    ]
    N_ROWS = len(rows)
    N_COLS = len(COL_LABELS)

    col_widths = [0.46, 0.27, 0.27]
    total_w    = sum(col_widths)
    col_widths = [w / total_w for w in col_widths]

    FW        = 8.5
    ROW_H     = 0.34
    GRP_H     = 0.44
    HDR_H     = 0.36
    BODY_H    = N_ROWS * ROW_H
    FH        = GRP_H + HDR_H + BODY_H + 0.20

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(FW, FH))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")

    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w * FW)
    col_w = [w * FW for w in col_widths]

    body_top  = FH - GRP_H - HDR_H
    body_bot  = body_top - BODY_H

    ber_cmap = mcolors.LinearSegmentedColormap.from_list("ber", ["#1a9641", "#fee08b", "#d73027"])

    def ber_color(v):
        if v is None:
            return "#f0f0f0"
        return ber_cmap(min(1.0, max(0.0, v / 50.0)))

    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.1f}"

    # ── group header row ──────────────────────────────────────────────────
    grp_top = FH - GRP_H
    ci = 0
    for lbl, span, bg in GROUP_HDRS:
        x0 = col_x[ci]
        x1 = col_x[ci + span - 1] + col_w[ci + span - 1]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, grp_top), x1 - x0, GRP_H,
            boxstyle="square,pad=0", linewidth=0, facecolor=bg, edgecolor="none"))
        ax.text(x0 + (x1 - x0)/2, grp_top + GRP_H/2, lbl,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white", linespacing=1.3)
        ci += span

    # ── sub-header row ────────────────────────────────────────────────────
    hdr_top = grp_top - HDR_H
    for ci_, (lbl, cx, cw) in enumerate(zip(COL_LABELS, col_x, col_w)):
        bg = "#d8e4ec" if ci_ == 0 else "#e8e8e8"
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, hdr_top), cw, HDR_H,
            boxstyle="square,pad=0", linewidth=0, facecolor=bg, edgecolor="none"))
        ax.text(cx + cw/2, hdr_top + HDR_H/2, lbl,
                ha="center", va="center", fontsize=7.5, fontweight="bold")

    # ── data rows ─────────────────────────────────────────────────────────
    for ri, (desc, d_ber, s_ber) in enumerate(rows):
        ry   = body_top - (ri + 1) * ROW_H
        stripe = "#f7f7f7" if ri % 2 == 0 else "#ffffff"

        cells = [
            (desc, stripe, False),
            (fmt(d_ber), ber_color(d_ber), True),
            (fmt(s_ber), ber_color(s_ber), True),
        ]
        for ci_, (txt, bg, _numeric) in enumerate(cells):
            ax.add_patch(mpatches.FancyBboxPatch(
                (col_x[ci_], ry), col_w[ci_], ROW_H,
                boxstyle="square,pad=0", linewidth=0, facecolor=bg, edgecolor="none"))
            fw_ = "normal" if ci_ == 0 else "normal"
            ha_ = "left"   if ci_ == 0 else "center"
            tx  = col_x[ci_] + 0.06 if ci_ == 0 else col_x[ci_] + col_w[ci_]/2
            ax.text(tx, ry + ROW_H/2, txt, ha=ha_, va="center", fontsize=7.5, fontweight=fw_)

    # ── grid lines ────────────────────────────────────────────────────────
    grid_top = FH - GRP_H
    grid_bot = body_bot
    for ri in range(N_ROWS + 1):
        y  = body_top - ri * ROW_H
        lw = 1.0 if ri == 0 else 0.3
        ax.plot([0, FW], [y, y], color="#999999", linewidth=lw, zorder=2)
    ax.plot([0, FW], [grid_top,  grid_top],  color="#333333", linewidth=1.2, zorder=2)
    ax.plot([0, FW], [hdr_top,   hdr_top],   color="#777777", linewidth=0.8, zorder=2)
    ax.plot([0, FW], [body_bot,  body_bot],  color="#333333", linewidth=1.2, zorder=2)
    for ci_ in range(N_COLS + 1):
        x = col_x[ci_] if ci_ < N_COLS else FW
        lw = 1.2 if ci_ in (0, N_COLS) else 0.4
        ax.plot([x, x], [grid_bot, FH], color="#777777", linewidth=lw, zorder=2)

    fig.tight_layout(pad=0)
    save_fig(fig, "09_cross_paradigm_comparison.png")
    print("  Saved: 09_cross_paradigm_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Collapse / routing health summary  (figure 07) — grouped header layout
# ─────────────────────────────────────────────────────────────────────────────

COLLAPSE_ROWS = [
    # model, BER, max_use, load_l1, collapse, wins, row_bg
    ("R0 ep20",             "0.32", "0.306", "0.082", "None",    "3/8", "#c8e6c9"),
    ("Frozen top k=1 ep20", "2.72", "0.552", "0.604", "Partial", "---", "#ffcdd2"),
    ("Top-k=2 ep30",        "2.66", "0.324", "0.142", "None",    "3/9", "#ffcdd2"),
    ("Dense k=4 ep20",      "2.62", "0.250", "0.000", "None",    "6/9", "#ffcdd2"),
    ("ep38 (continued)",    "0.27", "0.322", "0.166", "None",    "2/9", "#fff9c4"),
    ("ep52 (continued)",    "0.12", "0.475", "0.451", "Partial", "4/9", "#ffe0b2"),
    ("8exp collapse diag",  "0.48", "0.500", "0.840", "Partial", "4/9", "#ffe0b2"),
    ("Routing isolation",   "0.98", "0.998", "1.496", "Full",    "1/9", "#ffcdd2"),
    ("ep30 attack-trained", "0.18", "0.359", "0.253", "None",    "6/9", "#e1bee7"),
]

# (label, start_col_index, end_col_index_exclusive) — data cols only (after model col)
COLLAPSE_GROUPS = [
    ("Clean channel", 0, 1),
    ("Routing health", 1, 4),
    ("Attack eval", 4, 5),
]

COLLAPSE_METRICS = [
    "BER\n(%)", "max_use", "load_l1", "Collapse", "Wins\nvs HiDDeN",
]


def plot_collapse_analysis_table() -> None:
    """Grouped-header figure (not a LaTeX table) — styled like publication result grids."""
    n_rows = len(COLLAPSE_ROWS)
    n_metrics = len(COLLAPSE_METRICS)

    col_widths = [0.28, 0.13, 0.13, 0.13, 0.165, 0.165]
    assert abs(sum(col_widths) - 1.0) < 1e-3

    fw = 8.8
    group_h = 0.42
    hdr_h = 0.72
    row_h = 0.36
    note_h = 0.24
    fh = group_h + hdr_h + n_rows * row_h + note_h + 0.10

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")

    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w * fw)
    col_w = [w * fw for w in col_widths]

    body_top = fh - group_h - hdr_h
    body_bot = body_top - n_rows * row_h
    group_top = fh - group_h

    hdr_bg = "#2c5f7a"
    group_bg = "#1a1a2e"

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, body_bot - note_h), fw, fh - (body_bot - note_h),
        boxstyle="square,pad=0", linewidth=0.9,
        edgecolor="#222222", facecolor="none", zorder=3,
    ))

    # Top-left "Model" spanning group + metric header rows
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, group_top), col_w[0], group_h + hdr_h,
        boxstyle="square,pad=0", linewidth=0, facecolor=group_bg, edgecolor="none",
    ))
    ax.text(
        col_w[0] / 2, group_top + (group_h + hdr_h) / 2,
        "Checkpoint", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white",
    )

    # Group header row (columns 1..5)
    data_x0 = col_x[1]
    for label, c0, c1 in COLLAPSE_GROUPS:
        x0 = col_x[1 + c0]
        x1 = col_x[1 + c1] if (1 + c1) < len(col_x) else fw
        w = x1 - x0
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, group_top), w, group_h,
            boxstyle="square,pad=0", linewidth=0, facecolor=group_bg, edgecolor="none",
        ))
        ax.text(
            x0 + w / 2, group_top + group_h / 2, label,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="white",
        )

    # Metric sub-headers (rotated for narrow cols)
    for mi, (metric, cx, cw) in enumerate(zip(COLLAPSE_METRICS, col_x[1:], col_w[1:])):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, body_top), cw, hdr_h,
            boxstyle="square,pad=0", linewidth=0, facecolor=hdr_bg, edgecolor="none",
        ))
        rot = 90 if mi > 0 or "BER" in metric else 0
        ax.text(
            cx + cw / 2, body_top + hdr_h / 2, metric,
            ha="center", va="center", rotation=rot,
            fontsize=7.8, fontweight="bold", color="white", linespacing=1.15,
        )

    # Data rows
    for ri, row in enumerate(COLLAPSE_ROWS):
        name, ber, mu, ll, coll, wins, bg = row
        ry = body_top - (ri + 1) * row_h
        vals = [name, ber, mu, ll, coll, wins]
        for ci, (val, cx, cw) in enumerate(zip(vals, col_x, col_w)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, ry), cw, row_h,
                boxstyle="square,pad=0", linewidth=0, facecolor=bg, edgecolor="none",
            ))
            fw_txt = "bold" if ci == 0 else "normal"
            ax.text(
                cx + cw / 2, ry + row_h / 2, val,
                ha="center", va="center", fontsize=8.5, fontweight=fw_txt,
            )

    # Grid lines
    for ri in range(n_rows + 1):
        y = body_top - ri * row_h
        lw = 1.0 if ri == 0 else 0.35
        ax.plot([0, fw], [y, y], color="#555555", linewidth=lw, zorder=2)
    y_group = group_top
    ax.plot([0, fw], [y_group, y_group], color="#555555", linewidth=1.0, zorder=2)
    ax.plot([0, fw], [fh, fh], color="#555555", linewidth=0.9, zorder=2)

    for ci in range(len(col_x) + 1):
        x = col_x[ci] if ci < len(col_x) else fw
        lw = 0.9 if ci in (0, 1, len(col_x)) else 0.35
        ax.plot([x, x], [body_bot, fh], color="#555555", linewidth=lw, zorder=2)
    # thicker between clean | routing | attack groups
    for boundary in (col_x[2], col_x[5]):
        ax.plot([boundary, boundary], [body_bot, group_top + group_h], color="#555555", linewidth=0.7, zorder=2)

    footer_y = body_bot - note_h / 2
    ax.text(
        fw / 2, footer_y + 0.07,
        "Routing health summary across evaluated checkpoints",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#222222",
    )
    leg_items = [
        ("#c8e6c9", "Healthy (R0)"),
        ("#fff9c4", "Continued — drift"),
        ("#ffe0b2", "Low BER, routing gap"),
        ("#ffcdd2", "Full collapse"),
        ("#e1bee7", "Attack-trained"),
    ]
    lx = 0.08
    for bg, lbl in leg_items:
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx, footer_y - 0.12), 0.18, 0.09,
            boxstyle="round,pad=0.02", linewidth=0.4,
            facecolor=bg, edgecolor="#555555",
        ))
        ax.text(lx + 0.22, footer_y - 0.075, lbl, ha="left", va="center", fontsize=6.6)
        lx += 1.55

    save_fig(fig, "07_collapse_analysis.png")


if __name__ == "__main__":
    print("Generating thesis table figures...")
    plot_clean_channel_table()
    plot_attack_robustness_table()
    plot_rq2_bar_chart()
    plot_attack_ber_delta_table()
    plot_backbone_table()
    plot_collapse_analysis_table()
    plot_cross_paradigm_comparison()
    print(f"\nAll figures saved to:\n  {OUT}\n  {OUT_NEW}")
