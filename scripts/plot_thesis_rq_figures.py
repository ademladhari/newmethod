#!/usr/bin/env python3
"""
Thesis figures for Research Questions RQ1–RQ5 (grayscale, evidence-based).

Usage:
  py -3 scripts/plot_thesis_rq_figures.py

Outputs: thesis/figures/rq/*.png and *.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_moe_experiment_results import (  # noqa: E402
    EXP_ROOT,
    load_all_runs,
    load_epoch_series,
)

OUT = ROOT / "thesis" / "figures" / "rq"
ATTACK_MAIN = ROOT / "results" / "comparison_hidden_vs_moe" / "attack_summary_main_ep20.csv"
R0_VAL = (
    EXP_ROOT
    / "unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/validation.csv"
)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "serif",
        "font.size": 10,
        "axes.edgecolor": "black",
        "text.color": "black",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def r0_ep20() -> dict:
    df = pd.read_csv(R0_VAL)
    row = df[df["epoch"] == 20].iloc[0]
    return {
        "ber_pct": float(row["bitwise-error"]) * 100,
        "max_use": float(row["expert_max_use"]),
        "load_l1": float(row["train_val_load_l1"]),
        "eff_exp": float(row["effective_experts"]),
    }


# ---------------------------------------------------------------------------
# RQ1 — Clean watermark quality
# ---------------------------------------------------------------------------
def fig_rq1_clean_quality():
    """MoE routing diagnostics only (clean BER is in Figure 5.1 table)."""
    r0 = r0_ep20()

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    labels_m = ["Effective\nexperts", "max_use", "load_l1"]
    r0_vals = [r0["eff_exp"], r0["max_use"], r0["load_l1"]]
    x = np.arange(3)
    bars = ax.bar(x, r0_vals, color="#e8f4ec", edgecolor="black", linewidth=1.2)
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1, label="uniform max_use (4e)")
    ax.axhline(0.25, color="0.45", linestyle="-.", linewidth=1, label="ideal max_use (4e, $k{=}1$)")
    ax.axhline(0.20, color="0.35", linestyle="--", linewidth=1, label="healthy load_l1 ($<0.20$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m, fontsize=9)
    ax.set_ylabel("Routing metric value")
    ax.set_title("MoE R0 routing stability at epoch 20", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True, edgecolor="black")
    for b, v in zip(bars, r0_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(1.05, max(r0_vals) * 1.15))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    save(fig, "RQ1_clean_quality")

    # copy to new_figures for thesis path consistency
    new_dir = ROOT / "thesis" / "figures" / "new_figures"
    new_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        src = OUT / f"RQ1_clean_quality.{ext}"
        if src.exists():
            import shutil
            shutil.copy(src, new_dir / f"RQ1_routing_stability.{ext}")


# ---------------------------------------------------------------------------
# RQ2 — Attack robustness
# ---------------------------------------------------------------------------
def fig_rq2_attacks():
    df = pd.read_csv(ATTACK_MAIN)
    df = df[df["attack"] != "combined"].copy()
    order = ["identity", "quant", "dropout", "cropout", "crop", "resize", "gaussian", "jpeg"]
    df["attack"] = pd.Categorical(df["attack"], categories=order, ordered=True)
    df = df.sort_values("attack")

    h_acc = df["hidden_bit_acc"].values * 100
    m_acc = df["moe_bit_acc"].values * 100
    delta = (df["moe_minus_hidden_acc"].values) * 100
    n_win = int((delta > 0.5).sum())
    n_loss = int((delta < -0.5).sum())
    n_tie = len(delta) - n_win - n_loss

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(len(df))
    w = 0.36
    axes[0].bar(x - w / 2, h_acc, w, label="HiDDeN-177", color="0.85", edgecolor="black", linewidth=0.9)
    axes[0].bar(x + w / 2, m_acc, w, label="MoE R0 (ep20)", color="white", edgecolor="black",
                linewidth=0.9, hatch="///")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["attack"].str.upper(), rotation=35, ha="right")
    axes[0].set_ylabel("Bit accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(50, color="0.5", linestyle=":", linewidth=1)
    axes[0].legend(frameon=True, edgecolor="black", fontsize=8)
    axes[0].set_title("(a) Per-attack bit accuracy (n=800 images each)", fontweight="bold")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)

    colors = ["0.25" if d > 0.5 else ("0.75" if d < -0.5 else "0.55") for d in delta]
    axes[1].barh(x, delta, color=colors, edgecolor="black", linewidth=0.8, height=0.65)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(df["attack"].str.upper())
    axes[1].set_xlabel("MoE − HiDDeN bit accuracy (pp)")
    axes[1].set_title("(b) MoE advantage by attack", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", linestyle=":", alpha=0.6)
    for i, d in enumerate(delta):
        axes[1].text(d + (0.8 if d >= 0 else -0.8), i, f"{d:+.1f}", va="center",
                     ha="left" if d >= 0 else "right", fontsize=8)

    weakness = "JPEG" if n_loss == 1 else f"{n_loss} attacks"
    fig.suptitle(
        f"RQ2: MoE wins on {n_win}/8 attacks; {weakness} is the main weakness",
        fontweight="bold", y=1.06,
    )
    fig.text(0.5, -0.04,
             f"Wins (Δ>0.5 pp): {n_win} · Losses: {n_loss} · Near tie: {n_tie} · Source: attack_summary_main_ep20.csv",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    save(fig, "RQ2_attack_robustness")


# ---------------------------------------------------------------------------
# RQ3 — Four necessary conditions (ablation)
# ---------------------------------------------------------------------------
def fig_rq3_ablation():
    """Isolated comparisons for backbone, top-k, num experts."""
    rows = [
        ("Unfrozen\n(R0)", 0.32, 0.3055, 0.0824, True),
        ("Frozen\n(bal08)", 2.72, 0.552, 0.6042, False),
        ("top-k=1\n(R0)", 0.32, 0.3055, 0.0824, True),
        ("top-k=2", 3.10, 0.3662, 0.1232, False),
        ("top-k=4 dense", 2.62, 0.25, 0.0, False),
        ("4 experts\n(R0)", 0.32, 0.3055, 0.0824, True),
        ("8 experts", 1.70, 0.2965, 0.5263, False),
    ]
    panels = [
        ("Condition 1: Backbone", [0, 1], "Frozen limits adaptation"),
        ("Condition 2: top-k routing", [2, 3, 4], "Sparse top-1 only"),
        ("Condition 3: Expert count", [5, 6], "4e on COCO-100k"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (title, idxs, note) in zip(axes, panels):
        sub = [rows[i] for i in idxs]
        labels = [r[0] for r in sub]
        bers = [r[1] for r in sub]
        ok = [r[4] for r in sub]
        x = np.arange(len(sub))
        colors = ["white" if o else "0.8" for o in ok]
        hatches = ["///" if o else "" for o in ok]
        bars = ax.bar(x, bers, color=colors, edgecolor="black", linewidth=1.1)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Clean BER @ ep20 (%)")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        for b, v in zip(bars, bers):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", fontsize=8)
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", fontsize=7, style="italic")

    fig.suptitle("RQ3: Unfrozen backbone, top-k=1, and 4 experts are jointly necessary on COCO-100k",
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "RQ3_ablation_conditions")


def fig_rq3_routing_symmetry():
    """Condition 4: train/val gap — R0 vs routing isolation (jitter)."""
    paths = [
        ("unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
         "R0 (symmetric)", "-"),
        ("frozen_moe/coco100k/4exp_k1/batch32/routing_isolation_jitter005_adv0_ep40_2026-05-31",
         "Jitter (isolation)", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, ylab in zip(axes, ["expert_max_use", "train_val_load_l1"], ["expert_max_use", "train_val_load_l1"]):
        for path, label, ls in paths:
            val = load_epoch_series(EXP_ROOT / path, "validation.csv")
            if val is None:
                continue
            ax.plot(val["epoch"], val[col], ls, color="black", linewidth=1.5, label=label)
        if col == "expert_max_use":
            ax.axhline(0.25, color="0.5", linestyle=":", linewidth=1)
        if col == "train_val_load_l1":
            ax.axhline(0.20, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, frameon=True, edgecolor="black")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_title("(a) Expert max use", fontweight="bold")
    axes[1].set_title("(b) Train–validation load L1", fontweight="bold")
    fig.suptitle("RQ3 (condition 4): Train-only jitter inflates val routing gap", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "RQ3_routing_symmetry")


# ---------------------------------------------------------------------------
# RQ4 — Inference cost (data-driven from benchmark_inference.py output)
# ---------------------------------------------------------------------------
def fig_rq4_inference_cost():
    bench_csv = ROOT / "results" / "rq4_inference_benchmark.csv"
    param_csv = ROOT / "results" / "rq4_param_breakdown.csv"

    if not bench_csv.exists():
        print(f"  [RQ4] Benchmark CSV not found: {bench_csv}")
        print("  Run: python scripts/benchmark_inference.py")
        _fig_rq4_placeholder()
        return

    df = pd.read_csv(bench_csv)
    hidden_df = df[df["model"] == "HiDDeN"].set_index("batch_size")
    moe_df = df[df["model"] == "MoE"].set_index("batch_size")
    batch_sizes = sorted(df["batch_size"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- Panel 1: Full-pipeline latency per image ---
    ax = axes[0]
    x = np.arange(len(batch_sizes))
    w = 0.35
    h_vals = [hidden_df.loc[bs, "full_ms_per_image"] for bs in batch_sizes]
    m_vals = [moe_df.loc[bs, "full_ms_per_image"] for bs in batch_sizes]
    h_err = [hidden_df.loc[bs, "full_ms_std"] / bs for bs in batch_sizes]
    m_err = [moe_df.loc[bs, "full_ms_std"] / bs for bs in batch_sizes]
    ax.bar(x - w / 2, h_vals, w, yerr=h_err, label="HiDDeN", color="white",
           edgecolor="black", capsize=4, linewidth=1.2)
    ax.bar(x + w / 2, m_vals, w, yerr=m_err, label="MoE (top-1)", color="0.6",
           edgecolor="black", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"batch={b}" for b in batch_sizes])
    ax.set_ylabel("ms / image")
    ax.set_title("Full pipeline latency\n(encode + decode)", fontweight="bold")
    ax.legend(frameon=False)

    # --- Panel 2: Decode-only latency per image ---
    ax = axes[1]
    h_dec = [hidden_df.loc[bs, "decode_ms_per_image"] for bs in batch_sizes]
    m_dec = [moe_df.loc[bs, "decode_ms_per_image"] for bs in batch_sizes]
    h_dec_err = [hidden_df.loc[bs, "decode_ms_std"] / bs for bs in batch_sizes]
    m_dec_err = [moe_df.loc[bs, "decode_ms_std"] / bs for bs in batch_sizes]
    ax.bar(x - w / 2, h_dec, w, yerr=h_dec_err, label="HiDDeN", color="white",
           edgecolor="black", capsize=4, linewidth=1.2)
    ax.bar(x + w / 2, m_dec, w, yerr=m_dec_err, label="MoE (top-1)", color="0.6",
           edgecolor="black", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"batch={b}" for b in batch_sizes])
    ax.set_ylabel("ms / image")
    ax.set_title("Decode-only latency\n(router + 1 expert)", fontweight="bold")
    ax.legend(frameon=False)

    # Overhead ratio annotations
    for i, bs in enumerate(batch_sizes):
        ratio = m_dec[i] / max(h_dec[i], 1e-9)
        ax.text(i + w / 2, m_dec[i] + m_dec_err[i] + 0.002, f"{ratio:.2f}×",
                ha="center", va="bottom", fontsize=8)

    # --- Panel 3: Parameter counts (bar chart) ---
    ax = axes[2]
    if param_csv.exists():
        pdf = pd.read_csv(param_csv)
        components = ["encoder", "decoder", "moe_experts_all", "moe_router",
                      "moe_shared_extractor", "decoder_feature_layers"]
        labels, h_counts, m_counts = [], [], []
        for comp in components:
            h_row = pdf[(pdf["model"] == "HiDDeN") & (pdf["component"] == comp)]
            m_row = pdf[(pdf["model"] == "MoE") & (pdf["component"] == comp)]
            if h_row.empty and m_row.empty:
                continue
            labels.append(comp.replace("_", "\n"))
            h_counts.append(int(h_row["parameters"].values[0]) if not h_row.empty else 0)
            m_counts.append(int(m_row["parameters"].values[0]) if not m_row.empty else 0)
        xp = np.arange(len(labels))
        ax.bar(xp - w / 2, [v / 1e6 for v in h_counts], w, label="HiDDeN",
               color="white", edgecolor="black", linewidth=1.2)
        ax.bar(xp + w / 2, [v / 1e6 for v in m_counts], w, label="MoE",
               color="0.6", edgecolor="black", linewidth=1.2)
        ax.set_xticks(xp)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Parameters (M)")
        ax.set_title("Model parameters\nby component", fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Run benchmark_inference.py\nfor parameter counts",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    device_label = df.get("device", pd.Series(["unknown"]))[0] if "device" in df.columns else "measured"
    fig.suptitle(
        f"RQ4: MoE inference cost vs HiDDeN baseline ({device_label})\n"
        "Sparse top-k=1: only one expert decoded per image regardless of expert count.",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "RQ4_inference_cost")


def _fig_rq4_placeholder():
    """Fallback conceptual diagram when benchmark CSV is missing."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    stages = [
        ("Encoder\n(HiDDeN)", 1.0, ""),
        ("Router\n(2-layer MLP)", 0.08, "///"),
        ("Decoder\n(1 of 4 experts)", 1.0, ""),
    ]
    x0, w = 0.05, 0.26
    for i, (name, rel, hatch) in enumerate(stages):
        rect = plt.Rectangle((x0 + i * 0.32, 0.35), w, 0.45, fill=True, facecolor="white",
                              edgecolor="black", linewidth=1.5, hatch=hatch)
        ax.add_patch(rect)
        ax.text(x0 + i * 0.32 + w / 2, 0.57, name, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x0 + i * 0.32 + w / 2, 0.42,
                f"≈ {rel:.2f}×" if rel < 1 else "1.00× (baseline)",
                ha="center", va="center", fontsize=9)
        if i < 2:
            ax.annotate("", xy=(x0 + (i + 1) * 0.32 - 0.02, 0.57),
                        xytext=(x0 + i * 0.32 + w + 0.02, 0.57),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(0.5, 0.08,
            "⚠ Conceptual diagram — run benchmark_inference.py for real measurements.",
            ha="center", transform=ax.transAxes, fontsize=8, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("RQ4: Inference cost (conceptual — awaiting benchmark)", fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "RQ4_inference_cost")


# ---------------------------------------------------------------------------
# RQ5 — Collapse characterization
# ---------------------------------------------------------------------------
def fig_rq5_collapse_correlation():
    df = load_all_runs()
    sub = df[df["dataset"] == "coco100k"].copy()
    sub["collapse"] = sub["val_max_use"] * sub["num_experts"]
    sub["ber_pct"] = sub["val_ber"] * 100
    sub["config"] = sub.apply(
        lambda r: f"{int(r['num_experts'])}e·k{int(r['top_k'])}·b{int(r['batch_size'])}",
        axis=1,
    )
    d = sub[sub["ber_pct"].notna() & sub["collapse"].notna()].copy()
    x, y = d["collapse"].values, d["ber_pct"].values
    r, p = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for frozen, marker, hatch in [(False, "o", "///"), (True, "s", "")]:
        m = d[d["frozen"] == frozen]
        ax.scatter(m["collapse"], m["ber_pct"], s=90, marker=marker, facecolors="white",
                   edgecolors="black", linewidths=1, hatch=hatch, label="Unfrozen" if not frozen else "Frozen", zorder=3)
    xl = np.linspace(max(0.8, x.min() - 0.15), x.max() + 0.25, 50)
    ax.plot(xl, slope * xl + intercept, "k--", lw=1.8, alpha=0.8, label="Linear fit")
    ax.axvline(1.0, color="0.45", linestyle=":", linewidth=1)
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "balanced\n(score=1)", fontsize=8, va="top")
    for _, row in d.iterrows():
        if row["ber_pct"] < 2 or row["collapse"] > 1.8:
            ax.annotate(row["config"], (row["collapse"], row["ber_pct"]), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Collapse score = expert_max_use × N")
    ax.set_ylabel("Clean validation BER (%)")
    ax.set_title(
        f"RQ5: Routing collapse predicts BER (COCO-100k, n={len(d)})\n"
        f"Pearson r = {r:+.2f}, p = {p:.4f}",
        fontweight="bold",
    )
    ax.legend(frameon=True, edgecolor="black", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    save(fig, "RQ5_collapse_vs_ber")
    return r, p, len(d)


def fig_rq5_failure_modes():
    """Three signature panels: init collapse, jitter flat entropy, train/val gap."""
    examples = [
        (
            "legacy_early/coco100k/4exp_k2/batch16/moe4_balance_v2_100k_ep24_2026-05-30",
            "Mode 1: max_use ≈ 0.5 early\n(balance v2, top-2)",
        ),
        (
            "legacy_early/coco100k/4exp_k1/batch32/moe4_balance_v3_100k_ep26_2026-05-30",
            "Mode 2: high jitter\n(balance v3)",
        ),
        (
            "frozen_moe/coco100k/4exp_k1/batch32/routing_isolation_jitter005_adv0_ep40_2026-05-31",
            "Mode 3: train/val gap\n(routing isolation)",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (rel, title) in zip(axes, examples):
        val = load_epoch_series(EXP_ROOT / rel, "validation.csv")
        if val is None:
            ax.set_visible(False)
            continue
        ax2 = ax.twinx()
        l1, = ax.plot(val["epoch"], val["expert_max_use"], "k-", linewidth=1.5, label="max_use")
        l2, = ax2.plot(val["epoch"], val["bitwise_error"] * 100, "k--", linewidth=1.2, label="BER %")
        ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("max_use", color="black")
        ax2.set_ylabel("BER (%)", color="0.35")
        ax.set_title(title, fontweight="bold", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("RQ5: Three routing failure signatures in failed runs", fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "RQ5_failure_mode_signatures")


def fig_rq5_r0_vs_collapsed_trajectory():
    paths = [
        ("unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
         "R0 (success)", "-"),
        ("frozen_moe/coco100k/4exp_k1/batch32/routing_isolation_jitter005_adv0_ep40_2026-05-31",
         "Collapsed (jitter)", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric, ylab, scale in zip(
        axes,
        ["expert_max_use", "bitwise_error"],
        ["Collapse score (max_use×4)", "Validation BER (%)"],
        [4, 100],
    ):
        for path, label, ls in paths:
            val = load_epoch_series(EXP_ROOT / path, "validation.csv")
            if val is None:
                continue
            y = val[metric] * scale if metric == "bitwise_error" else val[metric] * 4
            ax.plot(val["epoch"], y, ls, color="black", linewidth=1.5, label=label)
        if metric == "expert_max_use":
            ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, frameon=True, edgecolor="black")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_title("(a) Routing health", fontweight="bold")
    axes[1].set_title("(b) Watermark BER", fontweight="bold")
    fig.suptitle("RQ5: Successful vs collapsed training dynamics (4 experts, top-1 family)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "RQ5_training_dynamics")


def write_readme(r: float, p: float, n: int) -> None:
    text = f"""# Research-question figures (Chapter 4)

Grayscale figures supporting RQ1–RQ5. Regenerate:

```bash
py -3 scripts/plot_thesis_rq_figures.py
```

| File | Research question | Use in thesis |
|------|-------------------|---------------|
| `RQ1_clean_quality` | RQ1 | §4.2 clean BER + routing @ ep20 |
| `RQ2_attack_robustness` | RQ2 | §4.3 main attack table figure |
| `RQ3_ablation_conditions` | RQ3 | §4.4 backbone / top-k / experts |
| `RQ3_routing_symmetry` | RQ3 | §4.4 train/val symmetry |
| `RQ4_inference_cost` | RQ4 | §4.6 inference schematic |
| `RQ5_collapse_vs_ber` | RQ5 | §4.5 correlation (r={r:+.2f}, p={p:.4f}, n={n}) |
| `RQ5_failure_mode_signatures` | RQ5 | §4.5 three failure modes |
| `RQ5_training_dynamics` | RQ5 | §4.5 R0 vs collapsed trajectories |

**Note:** RQ2 counts wins/losses from `attack_summary_main_ep20.csv` — verify JPEG/resize wording in text matches the figure.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    fig_rq1_clean_quality()
    fig_rq2_attacks()
    fig_rq3_ablation()
    fig_rq3_routing_symmetry()
    fig_rq4_inference_cost()
    r, p, n = fig_rq5_collapse_correlation()
    fig_rq5_failure_modes()
    fig_rq5_r0_vs_collapsed_trajectory()
    write_readme(r, p, n)
    print(f"Wrote RQ figures to {OUT} ({len(list(OUT.glob('RQ*.png')))} PNGs)")


if __name__ == "__main__":
    main()
