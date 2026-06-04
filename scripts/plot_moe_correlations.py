#!/usr/bin/env python3
"""
Correlation plots: how config choices relate to routing stability and BER.

Stability (lower = better):
  - expert_max_use  (collapse indicator)
  - train_val_load_l1  (train/val routing mismatch)
  - routing_stability_score = 0.5*max_use + 0.3*load_l1 + 0.2*(1 - eff_exp/num_experts)

Usage:
  python scripts/plot_moe_correlations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_moe_experiment_results import (  # noqa: E402
    EXP_ROOT,
    OUT_DIRS,
    load_all_runs,
    save,
)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})


DATASET_SIZE_K = {"coco20k": 20.0, "coco100k": 100.0}


def add_stability_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_size_k"] = df["dataset"].map(DATASET_SIZE_K)
    df["load_l1_filled"] = df["val_load_l1"].fillna(df["val_max_use"] * 1.5)
    ne = df["num_experts"].replace(0, np.nan)
    eff = df["val_eff_exp"].fillna(ne)
    df["expert_util_gap"] = (ne - eff) / ne.clip(lower=1)  # 0 = all experts used
    # Balanced routing: max_use ≈ 1/N → ratio 1; collapse → ratio ≈ N
    df["max_use_over_uniform"] = df["val_max_use"] * ne
    df["expert_util_frac"] = (eff / ne).clip(upper=1.0)
    df["routing_stability"] = (
        0.5 * df["val_max_use"]
        + 0.3 * df["load_l1_filled"].clip(upper=2.0)
        + 0.2 * df["expert_util_gap"].fillna(0)
    )
    df["ber_metric"] = df["val_noisy_ber"].fillna(df["val_ber"])
    df["frozen_int"] = df["frozen"].astype(int)
    return df


def pearson_str(x, y) -> tuple[float, str]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, "n<3"
    r, p = stats.pearsonr(x[mask], y[mask])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    direction = "↑ more stable" if r < 0 else "↓ less stable"
    if "ber" in str(y) or (hasattr(y, 'name') and 'ber' in str(y.name)):
        direction = "↓ lower BER" if r < 0 else "↑ higher BER"
    return r, f"r={r:+.2f}{sig}\n({direction})"


def scatter_reg(
    ax,
    x,
    y,
    xlabel,
    ylabel,
    title,
    color=None,
    annotate_points=False,
    labels=None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        ax.set_title(title + " (insufficient data)")
        return np.nan

    c = color if color is not None else "#4c72b0"
    ax.scatter(x, y, s=90, c=c, edgecolors="k", linewidths=0.5, alpha=0.85, zorder=3)

    if len(x) >= 2:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 50)
        ax.plot(xline, slope * xline + intercept, "r--", lw=2, alpha=0.8, label="linear fit")

    r, rtext = pearson_str(x, y)
    ax.text(
        0.05,
        0.95,
        rtext,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    if annotate_points and labels is not None:
        for xi, yi, lab in zip(x[mask] if hasattr(mask, '__len__') else x, y, labels):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(str(lab)[:14], (xi, yi), fontsize=5, alpha=0.75, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return r


def scatter_reg_grouped(ax, sub, xcol, ycol, xlabel, ylabel, title):
    """Scatter + fit per frozen/unfrozen group + pooled r."""
    for frozen, col, mk in [(False, "#55a868", "o"), (True, "#c44e52", "s")]:
        m = sub[sub["frozen"] == frozen]
        if m.empty:
            continue
        x, y = m[xcol].values.astype(float), m[ycol].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        ax.scatter(x, y, s=90, c=col, marker=mk, edgecolors="k", linewidths=0.5, alpha=0.85, zorder=3)
        if len(x) >= 2:
            sl, ic, _, _, _ = stats.linregress(x, y)
            xl = np.linspace(x.min(), x.max(), 30)
            ax.plot(xl, sl * xl + ic, color=col, ls="--", lw=1.5, alpha=0.7)
    mask = sub[xcol].notna() & sub[ycol].notna()
    r, p = stats.pearsonr(sub.loc[mask, xcol], sub.loc[mask, ycol])
    sig = "*" if p < 0.05 else ""
    ax.text(0.05, 0.95, f"pooled r={r:+.2f}{sig}", transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_batch_size_stability(df: pd.DataFrame):
    """Main story: larger batch → more stable routing."""
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        ("val_max_use", "expert_max_use (lower = more stable)"),
        ("load_l1_filled", "train_val_load_l1 (lower = better)"),
        ("routing_stability", "Routing stability score (lower = better)"),
        ("ber_metric", "BER (lower = better)"),
    ]
    for ax, (ycol, ylab) in zip(axes.flat, panels):
        scatter_reg_grouped(ax, sub, "batch_size", ycol, "Batch size", ylab, ylab)
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#55a868", label="Unfrozen"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#c44e52", label="Frozen"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.suptitle(
        "Batch size vs routing stability & BER (COCO-100k)\n"
        "Negative r on stability metrics ⇒ larger batches tend to reduce collapse",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "corr_01_batch_size_stability.png")


def plot_config_vs_stability_grid(df: pd.DataFrame):
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()

    pairs = [
        ("batch_size", "routing_stability", "Batch size", "Routing stability score"),
        ("balance_w", "routing_stability", "Balance loss weight", "Routing stability score"),
        ("jitter", "routing_stability", "Router jitter", "Routing stability score"),
        ("load_pen", "routing_stability", "Load penalty weight", "Routing stability score"),
        ("num_experts", "routing_stability", "Number of experts", "Routing stability score"),
        ("top_k", "routing_stability", "Top-k routing", "Routing stability score"),
        ("frozen_int", "routing_stability", "Frozen backbone (0=no, 1=yes)", "Routing stability score"),
        ("balance_w", "val_max_use", "Balance loss weight", "expert_max_use"),
        ("jitter", "val_max_use", "Router jitter", "expert_max_use"),
        ("batch_size", "ber_metric", "Batch size", "BER (noisy or clean val)"),
        ("routing_stability", "ber_metric", "Routing stability score", "BER"),
        ("frozen_int", "ber_metric", "Frozen backbone", "BER"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    rs = []
    for ax, (xcol, ycol, xl, yl) in zip(axes.flat, pairs):
        r = scatter_reg(ax, sub[xcol].values, sub[ycol].values, xl, yl, f"{xl} → {yl.split()[0]}")
        rs.append((xl, yl, r))
    for ax in axes.flat[len(pairs) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Config hyperparameters vs stability & BER (Pearson r in each panel)\n"
        "COCO-100k runs only; * p<0.05, ** p<0.01, *** p<0.001",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "corr_02_config_stability_grid.png")


def plot_correlation_coefficient_bars(df: pd.DataFrame):
    """Bar chart: which configs most affect stability."""
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()

    configs = ["batch_size", "balance_w", "jitter", "load_pen", "num_experts", "top_k", "frozen_int"]
    targets = [
        ("routing_stability", "Routing stability score"),
        ("val_max_use", "expert_max_use"),
        ("ber_metric", "BER"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels_nice = {
        "batch_size": "Batch size",
        "balance_w": "Balance loss w.",
        "jitter": "Router jitter",
        "load_pen": "Load penalty",
        "num_experts": "# Experts",
        "top_k": "Top-k",
        "frozen_int": "Frozen backbone",
    }

    for ax, (tcol, ttitle) in zip(axes, targets):
        rs = []
        for cfg in configs:
            mask = sub[cfg].notna() & sub[tcol].notna()
            if mask.sum() >= 3:
                r, _ = stats.pearsonr(sub.loc[mask, cfg], sub.loc[mask, tcol])
            else:
                r = np.nan
            rs.append(r)
        colors = ["#55a868" if r < 0 else "#c44e52" for r in rs]
        ypos = np.arange(len(configs))
        ax.barh(ypos, rs, color=colors, edgecolor="gray")
        ax.set_yticks(ypos)
        ax.set_yticklabels([labels_nice[c] for c in configs])
        ax.axvline(0, color="black", lw=1)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Pearson r")
        ax.set_title(f"Correlation with {ttitle}")
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "How much each config moves the metric\n"
        "Green: increasing config tends to improve stability / lower BER  |  Red: tends to hurt",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "corr_03_correlation_coefficient_bars.png")


def plot_enhanced_heatmap(df: pd.DataFrame):
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()
    cfg_cols = ["batch_size", "balance_w", "jitter", "load_pen", "num_experts", "top_k", "frozen_int"]
    out_cols = ["routing_stability", "val_max_use", "load_l1_filled", "ber_metric", "val_eff_exp"]
    all_cols = cfg_cols + out_cols
    corr = sub[all_cols].astype(float).corr()

    # Sub-matrix: configs vs outcomes
    sub_corr = corr.loc[cfg_cols, out_cols]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub_corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(out_cols)))
    ax.set_yticks(range(len(cfg_cols)))
    ax.set_xticklabels(
        ["Stability\nscore", "max_use", "load_l1", "BER", "Eff.\nexpert"],
        fontsize=9,
    )
    ax.set_yticklabels(
        ["Batch", "Balance w", "Jitter", "Load pen", "#Experts", "Top-k", "Frozen"],
        fontsize=9,
    )
    for i in range(len(cfg_cols)):
        for j in range(len(out_cols)):
            v = sub_corr.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Config → outcome correlations (COCO-100k)\nGreen = config increase improves outcome")
    plt.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    save(fig, "corr_04_config_outcome_heatmap.png")


def plot_frozen_effect(df: pd.DataFrame):
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()
    metrics = ["routing_stability", "val_max_use", "ber_metric"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, metrics):
        u = sub[~sub["frozen"]][m].dropna()
        f = sub[sub["frozen"]][m].dropna()
        bp = ax.boxplot([u, f], tick_labels=["Unfrozen", "Frozen"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#55a868")
        bp["boxes"][1].set_facecolor("#c44e52")
        if len(u) >= 2 and len(f) >= 2:
            t, p = stats.ttest_ind(u, f, equal_var=False)
            ax.set_title(f"{m}\np(unpaired)={p:.3f}")
        else:
            ax.set_title(m)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Freezing HiDDeN backbone: stability & BER (COCO-100k)", fontsize=12)
    fig.tight_layout()
    save(fig, "corr_05_frozen_backbone_effect.png")


def plot_balance_jitter_detail(df: pd.DataFrame):
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, xcol, title in zip(
        axes,
        ["balance_w", "jitter"],
        ["Balance loss weight", "Router jitter noise"],
    ):
        for _, row in sub.iterrows():
            if np.isfinite(row[xcol]) and np.isfinite(row["val_max_use"]):
                c = "#55a868" if not row["frozen"] else "#c44e52"
                ax.scatter(row[xcol], row["val_max_use"], s=80, c=c, edgecolors="k", linewidths=0.4)
                ax.annotate(row["name"][:12], (row[xcol], row["val_max_use"]), fontsize=5, alpha=0.7)
        scatter_reg(ax, sub[xcol].values, sub["val_max_use"].values, title, "expert_max_use", title)
    fig.suptitle("Balance weight & jitter vs expert collapse", fontsize=12)
    fig.tight_layout()
    save(fig, "corr_06_balance_jitter_vs_collapse.png")


def plot_narrative_summary(df: pd.DataFrame):
    """Single figure with key takeaways as correlation numbers."""
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()
    stories = [
        ("batch_size", "routing_stability", "Larger batch → more stable routing"),
        ("batch_size", "val_max_use", "Larger batch → lower expert_max_use"),
        ("frozen_int", "routing_stability", "Unfrozen → more stable than frozen"),
        ("frozen_int", "val_max_use", "Unfrozen → less collapse"),
        ("jitter", "val_max_use", "More router jitter → more collapse"),
        ("balance_w", "val_max_use", "Balance weight vs collapse"),
        ("num_experts", "routing_stability", "More experts → stability?"),
        ("routing_stability", "ber_metric", "Stable routing ↔ lower BER"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    rs, texts, colors = [], [], []
    for xcol, ycol, desc in stories:
        mask = sub[xcol].notna() & sub[ycol].notna()
        if mask.sum() >= 3:
            r, p = stats.pearsonr(sub.loc[mask, xcol], sub.loc[mask, ycol])
            rs.append(r)
            texts.append(desc)
            colors.append("#55a868" if (r < 0 and "stability" in ycol or "max_use" in ycol or "BER" in desc) or (r > 0 and "BER" in desc and r < 0) else "#c44e52")
            # simplify color: green if relationship is "good for stability"
            if ycol in ("routing_stability", "val_max_use") and r < 0:
                colors[-1] = "#55a868"
            elif ycol == "ber_metric" and r > 0:
                colors[-1] = "#c44e52"
            elif ycol == "ber_metric" and r < 0:
                colors[-1] = "#55a868"
            else:
                colors[-1] = "#c44e52" if abs(r) > 0.3 else "#888888"

    y = np.arange(len(texts))
    ax.barh(y, rs, color=colors, edgecolor="k")
    ax.set_yticks(y)
    ax.set_yticklabels(texts, fontsize=10)
    ax.axvline(0, color="black")
    ax.set_xlabel("Pearson correlation coefficient")
    ax.set_xlim(-1, 1)
    ax.set_title("Key relationships (COCO-100k experiments, n={})".format(len(sub)))
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    save(fig, "corr_07_key_relationships_summary.png")


def _annotate_dataset_corr(ax, frame: pd.DataFrame, ycol: str):
    mask = frame["dataset_size_k"].notna() & frame[ycol].notna()
    n = int(mask.sum())
    if n >= 3:
        x = frame.loc[mask, "dataset_size_k"].values
        y = frame.loc[mask, ycol].values
        if np.std(x) > 0 and np.std(y) > 0:
            r, p = stats.pearsonr(x, y)
            ax.text(
                0.05, 0.95, f"r={r:+.2f} p={p:.3f} n={n}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            )
        else:
            ax.text(0.05, 0.95, f"n={n} (x or y constant)", transform=ax.transAxes, va="top", fontsize=8)
    else:
        ax.text(0.05, 0.95, f"n={n} (too few)", transform=ax.transAxes, va="top", fontsize=8)


def plot_dataset_size_stability(df: pd.DataFrame):
    """20k vs 100k, **stratified by num_experts** (4 vs 8); expert-normalized routing in bottom row."""
    sub = df[df["dataset"].isin(["coco20k", "coco100k"])].copy()
    if sub.empty:
        return

    colors = {"coco20k": "#8172b2", "coco100k": "#4c72b0"}
    rng = np.random.default_rng(7)
    expert_rows = [4, 8]
    panels = [
        ("val_max_use", "expert_max_use (raw)"),
        ("max_use_over_uniform", "max_use × N (1 = balanced, N = collapsed)"),
        ("expert_util_frac", "effective_experts / N"),
        ("ber_metric", "BER"),
    ]

    fig, axes = plt.subplots(len(expert_rows), len(panels), figsize=(15, 8), sharex=True)
    if len(expert_rows) == 1:
        axes = np.array([axes])

    for row_i, n_exp in enumerate(expert_rows):
        chunk = sub[sub["num_experts"] == n_exp]
        for col_i, (ycol, ylab) in enumerate(panels):
            ax = axes[row_i, col_i]
            for ds in ("coco20k", "coco100k"):
                m = chunk[chunk["dataset"] == ds]
                x = m["dataset_size_k"].values.astype(float)
                y = m[ycol].values.astype(float)
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if len(x) == 0:
                    continue
                jitter = rng.uniform(-3, 3, size=len(x))
                ax.scatter(
                    x + jitter, y, s=75, c=colors[ds],
                    edgecolors="k", linewidths=0.4, alpha=0.85,
                )
            _annotate_dataset_corr(ax, chunk, ycol)
            if row_i == len(expert_rows) - 1:
                ax.set_xlabel("Training images (thousands)")
            if col_i == 0:
                ax.set_ylabel(f"{n_exp} experts\n{ylab}")
            else:
                ax.set_ylabel(ylab)
            ax.set_xticks([20, 100])
            ax.set_xlim(5, 115)
            ax.grid(True, alpha=0.3)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["coco20k"], label="COCO 20k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["coco100k"], label="COCO 100k"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(
        "Dataset size vs routing / BER — matched by expert count\n"
        "Row 1: 4 experts only · Row 2: 8 experts only · Col 2 scales max_use by N",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "corr_08_dataset_size_stability.png")


def plot_dataset_size_topk_matched(df: pd.DataFrame):
    """Optional tighter view: same num_experts AND top_k (sparse k=1 vs dense k=2)."""
    sub = df[df["dataset"].isin(["coco20k", "coco100k"])].copy()
    if sub.empty:
        return

    colors = {"coco20k": "#8172b2", "coco100k": "#4c72b0"}
    rng = np.random.default_rng(11)
    groups = [(4, 1), (4, 2), (8, 1), (8, 2)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (n_exp, top_k) in zip(axes.flat, groups):
        chunk = sub[(sub["num_experts"] == n_exp) & (sub["top_k"] == top_k)]
        for ds in ("coco20k", "coco100k"):
            m = chunk[chunk["dataset"] == ds]
            x, y = m["dataset_size_k"].values, m["max_use_over_uniform"].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() == 0:
                continue
            jitter = rng.uniform(-3, 3, mask.sum())
            ax.scatter(x[mask] + jitter, y[mask], s=80, c=colors[ds], edgecolors="k", linewidths=0.4)
        _annotate_dataset_corr(ax, chunk, "max_use_over_uniform")
        ax.set_title(f"{n_exp} exp, top-{top_k} (n={len(chunk)})")
        ax.set_xlabel("Images (k)")
        ax.set_ylabel("max_use × N")
        ax.set_xticks([20, 100])
        ax.axhline(1.0, color="gray", ls=":", lw=1, label="balanced")
        ax.grid(True, alpha=0.3)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["coco20k"], label="20k"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["coco100k"], label="100k"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8)
    fig.suptitle("Dataset size vs expert-normalized collapse (matched experts + top-k)", fontsize=11)
    fig.tight_layout()
    save(fig, "corr_10_dataset_size_experts_topk.png")


def plot_dataset_size_summary_table(df: pd.DataFrame):
    """Medians by (dataset × num_experts) + per-run list."""
    sub = df[df["dataset"].isin(["coco20k", "coco100k"])].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(sub) + 3)))
    ax.axis("off")

    rows = []
    for ds in ("coco20k", "coco100k"):
        for n_exp in sorted(sub["num_experts"].dropna().unique()):
            g = sub[(sub["dataset"] == ds) & (sub["num_experts"] == n_exp)]
            if g.empty:
                continue
            rows.append([
                ds,
                str(int(n_exp)),
                str(len(g)),
                f"{g['val_max_use'].median():.3f}",
                f"{g['max_use_over_uniform'].median():.2f}",
                f"{g['val_ber'].median()*100:.2f}%",
            ])
    tbl = ax.table(
        cellText=rows,
        colLabels=["Dataset", "Experts", "n", "median max_use", "median max_use×N", "median BER"],
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.15, 1.35)

    detail = "\n".join(
        f"{r['dataset']:8} {int(r['num_experts'])}exp k{int(r['top_k'])} "
        f"max={r['val_max_use']:.3f}×N={r['max_use_over_uniform']:.2f} "
        f"BER={r['val_ber']*100:.1f}% {r['name'][:28]}"
        for _, r in sub.sort_values(["num_experts", "dataset", "val_max_use"]).iterrows()
        if np.isfinite(r.get("val_max_use", np.nan))
    )
    ax.text(0.02, 0.38, detail, transform=ax.transAxes, fontsize=6.5, family="monospace", va="top")
    ax.set_title("Dataset size comparison — grouped by expert count", fontsize=11, pad=20)
    fig.tight_layout()
    save(fig, "corr_09_dataset_size_run_table.png")


def write_correlation_report(df: pd.DataFrame):
    sub = df[(df["dataset"] == "coco100k") & (df["batch_size"] > 0)].copy()
    all_ds = df[df["dataset"].isin(["coco20k", "coco100k"])].copy()
    lines = ["# MoE config ↔ stability correlation report\n", f"COCO-100k runs (batch>0): {len(sub)}\n\n"]
    pairs = [
        ("batch_size", "routing_stability"),
        ("batch_size", "val_max_use"),
        ("batch_size", "ber_metric"),
        ("balance_w", "val_max_use"),
        ("jitter", "val_max_use"),
        ("load_pen", "val_max_use"),
        ("frozen_int", "routing_stability"),
        ("frozen_int", "val_max_use"),
        ("num_experts", "val_max_use"),
        ("top_k", "val_max_use"),
    ]
    for x, y in pairs:
        mask = sub[x].notna() & sub[y].notna()
        if mask.sum() >= 3:
            r, p = stats.pearsonr(sub.loc[mask, x], sub.loc[mask, y])
            lines.append(f"- **{x}** vs **{y}**: r={r:+.3f}, p={p:.4f}, n={mask.sum()}\n")

    lines.append("\n## Dataset size (20k vs 100k)\n\n")
    lines.append(
        "_Compare within the same expert count. `max_use×N` = 1 if perfectly balanced, "
        "≈ N if one expert takes all traffic._\n\n"
    )

    def _corr_block(frame: pd.DataFrame, label: str):
        lines.append(f"### {label} (n={len(frame)})\n\n")
        for x, y in [
            ("dataset_size_k", "val_max_use"),
            ("dataset_size_k", "max_use_over_uniform"),
            ("dataset_size_k", "ber_metric"),
        ]:
            mask = frame[x].notna() & frame[y].notna()
            if mask.sum() >= 3:
                r, p = stats.pearsonr(frame.loc[mask, x], frame.loc[mask, y])
                lines.append(f"- **{x}** vs **{y}**: r={r:+.3f}, p={p:.4f}, n={mask.sum()}\n")
        lines.append("\n")

    _corr_block(all_ds, "All runs (pooled — confounded by expert count)")
    for n_exp in sorted(all_ds["num_experts"].dropna().unique()):
        _corr_block(all_ds[all_ds["num_experts"] == n_exp], f"{int(n_exp)} experts only")

    for ds in ("coco20k", "coco100k"):
        for n_exp in sorted(all_ds["num_experts"].dropna().unique()):
            g = all_ds[(all_ds["dataset"] == ds) & (all_ds["num_experts"] == n_exp)]
            if g.empty:
                continue
            lines.append(f"#### {ds}, {int(n_exp)} experts (n={len(g)})\n\n")
            for _, r in g.sort_values("val_max_use").iterrows():
                ber = r["val_ber"] * 100 if np.isfinite(r["val_ber"]) else float("nan")
                lines.append(
                    f"- `{r['name'][:36]}`: max_use={r['val_max_use']:.3f}, "
                    f"×N={r['max_use_over_uniform']:.2f}, BER={ber:.2f}%, "
                    f"batch={int(r['batch_size'])}, top-{int(r['top_k'])}\n"
                )
            lines.append("\n")

    out = ROOT / "results" / "plots" / "CORRELATION_REPORT.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main():
    df = add_stability_columns(load_all_runs())
    plot_batch_size_stability(df)
    plot_config_vs_stability_grid(df)
    plot_correlation_coefficient_bars(df)
    plot_enhanced_heatmap(df)
    plot_frozen_effect(df)
    plot_balance_jitter_detail(df)
    plot_narrative_summary(df)
    plot_dataset_size_stability(df)
    plot_dataset_size_topk_matched(df)
    plot_dataset_size_summary_table(df)
    write_correlation_report(df)
    print("Correlation plots saved (corr_01 .. corr_10, 19_) in results/plots/ and thesis/figures/")


if __name__ == "__main__":
    main()
