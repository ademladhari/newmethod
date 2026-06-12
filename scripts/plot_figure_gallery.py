#!/usr/bin/env python3
"""
Generate a gallery of example figure types for thesis experiment explanation.
Each PNG is self-contained with a title describing the figure TYPE and use case.

Usage (repo root):
    .venv\\Scripts\\python.exe scripts/plot_figure_gallery.py

Output: thesis/figures/gallery/*.png + README.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis" / "figures" / "gallery"
ATTACK = ROOT / "results/comparison_hidden_vs_moe/attack_summary_main_ep20_soft_router.csv"
R0_VAL = (
    ROOT
    / "results/experiments/unfrozen_moe/coco100k/4exp_k1/batch128"
    / "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/validation.csv"
)
SUMMARY = ROOT / "results/comparison_hidden_vs_moe/summary.csv"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 150})

INDEX: list[tuple[str, str, str]] = []  # (filename, type_name, when_to_use)


def _save(fig: plt.Figure, fname: str, type_name: str, when: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / fname
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=160)
    plt.close(fig)
    INDEX.append((fname, type_name, when))
    print(f"  {fname}")


def _banner(ax, title: str, subtitle: str) -> None:
    ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, style="italic", color="#444444")


# ── 01 Pipeline flow ──────────────────────────────────────────────────────────
def fig_01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.0, "COCO\nimage", "#e8f4fc"),
        (1.5, 1.0, "Encoder\n$E_\\theta$", "#d0e8f8"),
        (2.8, 1.0, "Watermarked\n$X_w$", "#fff3cd"),
        (4.1, 1.0, "Attack\n$\\mathcal{A}$", "#f8d7da"),
        (5.4, 1.0, "Shared trunk\n$F_\\phi$", "#d0e8f8"),
        (6.7, 1.0, "Router\n$g$", "#e2d5f0"),
        (8.0, 1.0, "Expert(s)\n$e_i$", "#d6edd5"),
        (9.2, 1.0, "Bits\n$\\hat{m}$", "#fff3cd"),
    ]
    for x, y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.95, 0.9, boxstyle="round,pad=0.04",
                                    facecolor=col, edgecolor="#333", linewidth=1))
        ax.text(x + 0.48, y + 0.45, txt, ha="center", va="center", fontsize=8)
    for x0, x1 in [(1.15, 1.5), (2.45, 2.8), (3.75, 4.1), (5.05, 5.4), (6.35, 6.7), (7.65, 8.0), (8.95, 9.2)]:
        ax.annotate("", xy=(x1, 1.45), xytext=(x0, 1.45),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.text(4.1, 0.15, "RQ1: $\\mathcal{A}$ = identity  |  RQ2: JPEG, crop, …", ha="center", fontsize=8, color="#555")
    _banner(ax, "TYPE: Experimental pipeline diagram",
            "Explains what one eval run does — embed → attack → route → decode")
    _save(fig, "01_pipeline_flow.png", "Pipeline diagram", "Methodology / start of Results")


# ── 02 Warm-start schematic ───────────────────────────────────────────────────
def fig_02_warmstart() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 1.5), 2.2, 1.8, boxstyle="round", facecolor="#eef2f6", edgecolor="#333"))
    ax.text(1.4, 2.7, "HiDDeN ep177", ha="center", fontweight="bold")
    ax.text(1.4, 2.2, "Encoder + 1 decoder head", ha="center", fontsize=8)
    ax.annotate("", xy=(3.2, 2.4), xytext=(2.6, 2.4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(2.9, 2.7, "copy weights", ha="center", fontsize=7, style="italic")
    ax.add_patch(FancyBboxPatch((3.3, 1.2), 3.4, 2.4, boxstyle="round", facecolor="#e8f4ec", edgecolor="#2E8B57", linewidth=1.5))
    ax.text(5.0, 3.2, "MoE R0 init", ha="center", fontweight="bold", color="#2E8B57")
    ax.text(5.0, 2.6, "Same encoder + shared trunk", ha="center", fontsize=8)
    ax.text(5.0, 2.1, "+ router (new)", ha="center", fontsize=8)
    ax.text(5.0, 1.6, "+ 4 expert heads (split from decoder)", ha="center", fontsize=8)
    ax.annotate("", xy=(7.4, 2.4), xytext=(6.8, 2.4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(7.1, 2.7, "20 epochs\nfine-tune", ha="center", fontsize=7)
    ax.add_patch(FancyBboxPatch((7.5, 1.5), 2.0, 1.8, boxstyle="round", facecolor="#d6edd5", edgecolor="#333"))
    ax.text(8.5, 2.4, "R0 ep20\n(report)", ha="center", fontweight="bold")
    _banner(ax, "TYPE: Warm-start / architecture schematic",
            "Explains why RQ1 expects parity — MoE starts as a near-copy of HiDDeN")
    _save(fig, "02_warmstart_schematic.png", "Warm-start schematic", "Methodology / RQ1 setup")


# ── 03 Ablation lattice ───────────────────────────────────────────────────────
def fig_03_ablation_lattice() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    cols = ["k=1", "k=2", "k=4"]
    rows = ["unfrozen", "frozen"]
    colors = {"ok": "#c8e6c9", "bad": "#ffcdd2", "r0": "#2E8B57"}
    data = [
        ["R0 ✓", "2.66%", "2.62%"],
        ["2.72%", "—", "—"],
    ]
    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            val = data[ri][ci]
            bg = colors["r0"] if val.startswith("R0") else (colors["ok"] if ri == 0 and ci == 0 else
                  colors["bad"] if "%" in val and not val.startswith("R0") else "#f5f5f5")
            fc = "white" if bg == colors["r0"] else "black"
            ax.add_patch(FancyBboxPatch((ci, 1.5 - ri), 0.9, 0.7, boxstyle="square,pad=0",
                                        facecolor=bg, edgecolor="#333"))
            ax.text(ci + 0.45, 1.85 - ri, val, ha="center", va="center", fontsize=9,
                    fontweight="bold" if "R0" in val else "normal", color=fc)
    for ci, c in enumerate(cols):
        ax.text(ci + 0.45, 2.35, c, ha="center", fontweight="bold")
    for ri, r in enumerate(rows):
        ax.text(-0.35, 1.85 - ri, r, ha="right", va="center", fontweight="bold", rotation=90)
    ax.text(1.35, -0.25, "Numbers = clean BER (%) at batch 128, N=4", ha="center", fontsize=8, color="#555")
    _banner(ax, "TYPE: Ablation lattice (design grid)",
            "Shows which factor combinations you ran — not time series")
    _save(fig, "03_ablation_lattice.png", "Ablation lattice", "RQ3 experimental design")


# ── 04 Training timeline ──────────────────────────────────────────────────────
def fig_04_timeline() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.plot([1, 20], [1.5, 1.5], "k-", lw=2)
    for ep in [1, 10, 20]:
        ax.plot(ep, 1.5, "ko", ms=8)
        ax.text(ep, 1.2, f"ep {ep}", ha="center", fontsize=8)
    ax.axvspan(1, 10, alpha=0.15, color="#4C72B0", label="balance warm-up")
    ax.axvspan(10, 20, alpha=0.15, color="#2E8B57", label="full balance weight")
    ax.annotate("τ anneal", xy=(15, 1.5), xytext=(15, 2.2), ha="center",
                arrowprops=dict(arrowstyle="->", color="#888"))
    ax.annotate("R0 checkpoint", xy=(20, 1.5), xytext=(20, 2.5), ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2E8B57"))
    ax.axvspan(21, 60, alpha=0.08, color="#C0392B", hatch="///")
    ax.text(40, 1.5, "continuation (not primary)", ha="center", fontsize=8, color="#888")
    ax.legend(loc="lower center", ncol=2, fontsize=7, frameon=True)
    _banner(ax, "TYPE: Training schedule timeline",
            "Explains when you stop and what schedules change — not performance")
    _save(fig, "04_training_timeline.png", "Training timeline", "Methodology / RQ3 duration")


# ── 05 Signed advantage bars ────────────────────────────────────────────────
def fig_05_signed_advantage() -> None:
    df = pd.read_csv(ATTACK)
    df = df[df.attack != "combined"].sort_values("moe_minus_hidden_acc", ascending=True)
    delta = df["moe_minus_hidden_acc"].values * 100
    labels = df["attack"].str.upper().values
    colors = ["#2E8B57" if d > 0.5 else "#C0392B" if d < -0.5 else "#B0B0B0" for d in delta]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(labels)), delta, color=colors, edgecolor="black", height=0.65)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("MoE − HiDDeN bit accuracy (pp)")
    for i, d in enumerate(delta):
        ax.text(d + (0.3 if d >= 0 else -0.3), i, f"{d:+.1f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8)
    _banner(ax, "TYPE: Signed advantage bar chart",
            "Shows direction + magnitude per attack — better than a 3-column table for RQ2")
    _save(fig, "05_signed_advantage_bars.png", "Signed Δ bars", "RQ2 main result")


# ── 06 Attack-family grouped bars ───────────────────────────────────────────
def fig_06_attack_families() -> None:
    df = pd.read_csv(ATTACK)
    families = {
        "Clean": ["identity", "quant"],
        "Spatial": ["crop", "cropout", "resize", "dropout"],
        "Noise": ["gaussian"],
        "Freq.": ["jpeg"],
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    xpos, ticks, ticklabels = 0, [], []
    w = 0.18
    for fam, atks in families.items():
        sub = df[df.attack.isin(atks)]
        h = sub["hidden_bit_acc"].values * 100
        m = sub["moe_bit_acc"].values * 100
        xx = np.arange(len(atks)) * 0.5 + xpos
        ax.bar(xx - w / 2, h, w, color="#5B7FA5", label="HiDDeN" if xpos == 0 else "")
        ax.bar(xx + w / 2, m, w, color="#2E8B57", label="MoE R0" if xpos == 0 else "")
        ticks.extend(xx)
        ticklabels.extend([a.upper()[:4] for a in atks])
        ax.text(np.mean(xx), 102, fam, ha="center", fontsize=8, fontweight="bold")
        xpos += len(atks) * 0.5 + 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylabel("Bit accuracy (%)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 108)
    _banner(ax, "TYPE: Attack-family grouped bars",
            "Groups attacks by mechanism — supports narrative (spatial vs JPEG)")
    _save(fig, "06_attack_family_grouped.png", "Attack-family bars", "RQ2 discussion")


# ── 07 Compact heatmap ────────────────────────────────────────────────────────
def fig_07_heatmap() -> None:
    df = pd.read_csv(ATTACK)
    df = df[df.attack != "combined"]
    attacks = df["attack"].str.upper().tolist()
    data = np.array([df["hidden_bit_acc"].values, df["moe_bit_acc"].values]) * 100
    fig, ax = plt.subplots(figsize=(8, 2.2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=45, vmax=100)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HiDDeN", "MoE R0"])
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=45, ha="right", fontsize=8)
    for i in range(2):
        for j in range(len(attacks)):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] < 60 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Bit acc. %")
    _banner(ax, "TYPE: Compact heatmap (few models)",
            "Pattern at a glance — scale fixed 45–100% so JPEG losses pop")
    _save(fig, "07_compact_heatmap.png", "Compact heatmap", "RQ2 or appendix")


# ── 08 Trade-off scatter ──────────────────────────────────────────────────────
def fig_08_tradeoff_scatter() -> None:
    runs = [
        ("R0 ep20", 0.32, 0.082, "#2E8B57"),
        ("Frozen ep30", 7.04, 0.910, "#C0392B"),
        ("k=2 ep30", 2.66, 0.140, "#E67E22"),
        ("8exp-A ep20", 1.70, 0.526, "#9B59B6"),
        ("ep52 cont.", 0.12, 0.451, "#F39C12"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, ber, l1, c in runs:
        ax.scatter(l1, ber, s=120, c=c, edgecolors="black", zorder=3)
        ax.annotate(name, (l1, ber), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axhline(2.0, color="gray", ls=":", lw=1)
    ax.axvline(0.20, color="gray", ls="--", lw=1)
    ax.set_xlabel("train–val load L1 (routing gap)")
    ax.set_ylabel("Clean BER (%)")
    ax.text(0.05, 1.8, "healthy routing\nzone", fontsize=7, color="gray")
    _banner(ax, "TYPE: Trade-off / diagnostic scatter",
            "Links clean BER to routing health — explains RQ3/RQ5 mechanism")
    _save(fig, "08_tradeoff_scatter.png", "Trade-off scatter", "RQ3 / RQ5 collapse")


# ── 09 Expert load stacked (ep20) ───────────────────────────────────────────
def _val_col(df: pd.DataFrame, key: str) -> str:
    key = key.strip()
    for c in df.columns:
        if c.strip() == key:
            return c
    raise KeyError(key)


def fig_09_expert_load_stacked() -> None:
    v = pd.read_csv(R0_VAL)
    r = v[v.epoch == 20].iloc[0]
    loads = [float(r[_val_col(v, f"expert_load_{i}")]) for i in range(4)]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["Val load\nep20"], [0], color="none", edgecolor="none")
    bottom = 0
    cols = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (ld, c) in enumerate(zip(loads, cols)):
        ax.bar([0], ld, bottom=bottom, color=c, edgecolor="white", width=0.5, label=f"Expert {i}: {ld:.2f}")
        bottom += ld
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of images")
    ax.axhline(0.25, color="gray", ls="--", label="uniform (0.25)")
    ax.legend(fontsize=7, loc="upper right")
    _banner(ax, "TYPE: Stacked expert-load bar",
            "Shows which expert owns which share — explains max_use visually")
    _save(fig, "09_expert_load_stacked.png", "Expert load stack", "RQ1 / RQ5 routing")


# ── 10 Routing metrics over epochs ────────────────────────────────────────────
def fig_10_routing_over_epochs() -> None:
    v = pd.read_csv(R0_VAL)
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    ep = v["epoch"]
    axes[0].plot(ep, v["bitwise-error"] * 100, "o-", color="#2E8B57", ms=4)
    axes[0].set_ylabel("Val BER (%)")
    axes[0].axhline(0.32, color="gray", ls=":", alpha=0.7)
    axes[1].plot(ep, v["expert_max_use"], "o-", color="#5B7FA5", ms=4)
    axes[1].axhline(0.25, color="gray", ls="--", alpha=0.7)
    axes[1].set_ylabel("max_use")
    axes[2].plot(ep, v["train_val_load_l1"], "o-", color="#C0392B", ms=4)
    axes[2].axhline(0.20, color="gray", ls="--", alpha=0.7)
    axes[2].set_ylabel("load_l1")
    axes[2].set_xlabel("Epoch")
    fig.suptitle("TYPE: Training trajectory (small multiples)", fontweight="bold", y=1.01)
    fig.text(0.5, -0.02, "Three linked views — BER can improve while routing drifts (see ep19–20)",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    _save(fig, "10_training_trajectory.png", "Training trajectory", "RQ3 duration / RQ5")


# ── 11 Win-loss summary ───────────────────────────────────────────────────────
def fig_11_win_loss() -> None:
    df = pd.read_csv(ATTACK)
    df = df[df.attack != "combined"]
    d = df["moe_minus_hidden_acc"].values * 100
    wins = int((d > 0.5).sum())
    losses = int((d < -0.5).sum())
    ties = len(d) - wins - losses
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["MoE wins", "Near tie", "HiDDeN wins"], [wins, ties, losses],
           color=["#2E8B57", "#B0B0B0", "#C0392B"], edgecolor="black")
    ax.set_ylabel("Count (of 8 attacks)")
    for i, v in enumerate([wins, ties, losses]):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    _banner(ax, "TYPE: Win–loss summary bar",
            "One-number story for abstract / conclusion — detail in per-attack chart")
    _save(fig, "11_win_loss_summary.png", "Win–loss summary", "Abstract / RQ2 intro")


# ── 12 Failure-mode quadrant ──────────────────────────────────────────────────
def fig_12_failure_quadrant() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    runs = [
        ("R0", 0.31, 0.08, "#2E8B57"),
        ("ep52", 0.48, 0.45, "#F39C12"),
        ("Frozen", 0.64, 0.91, "#C0392B"),
        ("20k run", 0.35, 0.15, "#9B59B6"),
        ("Routing iso.", 1.00, 0.20, "#E74C3C"),
    ]
    for name, mu, l1, c in runs:
        ax.scatter(mu, l1, s=150, c=c, edgecolors="black", zorder=3)
        ax.annotate(name, (mu, l1), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvline(0.40, color="gray", ls="--")
    ax.axhline(0.20, color="gray", ls="--")
    ax.text(0.15, 0.05, "healthy", fontsize=9, color="#2E8B57")
    ax.text(0.75, 0.05, "collapsed\nrouter", fontsize=9, color="#C0392B", ha="center")
    ax.text(0.15, 0.55, "partial\ndrift", fontsize=9, color="#F39C12")
    ax.text(0.75, 0.55, "full\ncollapse", fontsize=9, color="#C0392B", ha="center")
    ax.set_xlabel("max_use (expert dominance)")
    ax.set_ylabel("load_l1 (train–val gap)")
    _banner(ax, "TYPE: Failure-mode quadrant",
            "Classifies runs by mechanism — not just 'bad BER'")
    _save(fig, "12_failure_mode_quadrant.png", "Failure quadrant", "RQ5 collapse")


# ── 13 Checkpoint selection flow ──────────────────────────────────────────────
def fig_13_checkpoint_flow() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    steps = [
        (5, 5.2, "All saved epochs"),
        (5, 4.0, "Filter: load_l1 < threshold"),
        (5, 2.8, "Rank by clean BER"),
        (5, 1.6, "Composite-best checkpoint"),
        (5, 0.4, "Attack / WAVES benchmark"),
    ]
    for i, (x, y, txt) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x - 1.8, y - 0.35), 3.6, 0.7, boxstyle="round,pad=0.03",
                                    facecolor="#e8f4fc" if i < 4 else "#d6edd5", edgecolor="#333"))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x, steps[i + 1][1] + 0.35), xytext=(x, y - 0.35),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(8.2, 4.0, "drop\ncollapsed", fontsize=7, color="#C0392B", style="italic")
    _banner(ax, "TYPE: Checkpoint-selection flowchart",
            "Explains evaluation protocol — defends against cherry-picking")
    _save(fig, "13_checkpoint_selection_flow.png", "Checkpoint flow", "Methodology / Evaluation")


# ── 14 Attack-suite taxonomy ──────────────────────────────────────────────────
def fig_14_attack_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 3.2), 9.4, 1.2, boxstyle="round", facecolor="#eef2f6", edgecolor="#333"))
    ax.text(5, 3.8, "Evaluation attack suites", ha="center", fontweight="bold", fontsize=10)
    ax.add_patch(FancyBboxPatch((0.5, 1.2), 4.2, 1.6, boxstyle="round", facecolor="#d0e8f8", edgecolor="#333"))
    ax.text(2.6, 2.5, "hidden (RQ2)", ha="center", fontweight="bold")
    ax.text(2.6, 2.0, "9 HiDDeN noise layers\n+ 5 PIL (subset)", ha="center", fontsize=8)
    ax.text(2.6, 1.45, "800 img / attack", ha="center", fontsize=7, color="#555")
    ax.add_patch(FancyBboxPatch((5.3, 1.2), 4.2, 1.6, boxstyle="round", facecolor="#e8f4ec", edgecolor="#2E8B57"))
    ax.text(7.4, 2.5, "waves (cross-method)", ha="center", fontweight="bold")
    ax.text(7.4, 2.0, "14 WAVES PIL attacks", ha="center", fontsize=8)
    ax.text(7.4, 1.45, "1000 img / attack", ha="center", fontsize=7, color="#555")
    ax.annotate("", xy=(2.6, 3.2), xytext=(2.6, 2.85), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.4, 3.2), xytext=(7.4, 2.85), arrowprops=dict(arrowstyle="->"))
    _banner(ax, "TYPE: Attack-suite taxonomy",
            "Explains why you have two benchmarks — not duplicate results")
    _save(fig, "14_attack_suite_taxonomy.png", "Attack taxonomy", "Methodology / Evaluation")


# ── 15 Protocol comparison (SSL vs yours) ─────────────────────────────────────
def fig_15_protocol_box() -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    rows = ["Resolution", "Payload", "Images", "Train data", "Comparable?"]
    h = ["128×128", "30-bit", "1000", "~10k", "—"]
    m = ["128×128", "30-bit", "1000", "~118k", "✓ primary"]
    s = ["512×512", "48-bit", "100", "varies", "context only"]
    col_x = [0.02, 0.28, 0.52, 0.76]
    headers = ["", "HiDDeN", "MoE R0", "SSL"]
    for j, (cx, hdr) in enumerate(zip(col_x, headers)):
        ax.text(cx, 0.92, hdr, fontweight="bold", fontsize=9)
    for i, row in enumerate(rows):
        y = 0.75 - i * 0.15
        ax.text(col_x[0], y, row, fontsize=8, fontweight="bold")
        ax.text(col_x[1], y, h[i], fontsize=8)
        ax.text(col_x[2], y, m[i], fontsize=8)
        ax.text(col_x[3], y, s[i], fontsize=8, color="#888" if i == 4 else "black")
    _banner(ax, "TYPE: Protocol comparison table (figure)",
            "Explains why SSL sits in heatmap as context — not head-to-head")
    _save(fig, "15_protocol_comparison.png", "Protocol comparison", "WAVES / cross-method")


# ── 16 PSNR distribution (violin/box) ─────────────────────────────────────────
def fig_16_psnr_distribution() -> None:
    s = pd.read_csv(SUMMARY)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    data = [s["hidden_psnr_full"], s["moe_psnr_full"]]
    bp = ax.boxplot(data, tick_labels=["HiDDeN", "MoE R0"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#5B7FA5", "#2E8B57"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("PSNR (dB) — 200 images")
    ax.axhline(35, color="gray", ls=":", label="35 dB reference")
    ax.legend(fontsize=7)
    _banner(ax, "TYPE: Distribution plot (box/violin)",
            "Shows spread not just mean — explains '≈2 dB gap' caveat in RQ1")
    _save(fig, "16_psnr_distribution.png", "PSNR distribution", "RQ1 imperceptibility")


# ── 17 Factor effect forest plot ──────────────────────────────────────────────
def fig_17_forest_effects() -> None:
    factors = ["Backbone\n(frozen→unf.)", "k: 1→2", "k: 1→4", "N: 4→8", "Continue\nto ep60"]
    delta_ber = [+6.7, +2.3, +2.3, +1.4, +0.4]  # illustrative pp shifts vs R0
    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = np.arange(len(factors))
    colors = ["#C0392B" if d > 1 else "#F39C12" if d > 0 else "#2E8B57" for d in delta_ber]
    ax.barh(y, delta_ber, color=colors, edgecolor="black", height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(factors, fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Δ clean BER (pp) vs R0 baseline")
    _banner(ax, "TYPE: Factor-effect / forest plot",
            "Ranks which ablation knobs hurt most — RQ3 summary without 11-row table")
    _save(fig, "17_factor_effect_forest.png", "Factor forest plot", "RQ3 summary")


# ── 18 Parallel coordinates (runs) ────────────────────────────────────────────
def fig_18_parallel_coords() -> None:
    runs = ["R0", "Frozen", "k=2", "k=4", "8exp-A"]
    metrics = np.array([
        [0.32, 0.31, 0.08, 5.0],   # BER, max_use, load_l1, attack_wins (of 8)
        [7.04, 0.64, 0.91, 1.0],
        [2.66, 0.32, 0.14, 3.0],
        [2.62, 0.25, 0.00, 2.0],
        [1.70, 0.30, 0.53, 4.0],
    ])
    # normalize 0-1 per column for display
    mn, mx = metrics.min(0), metrics.max(0)
    norm = (metrics - mn) / (mx - mn + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(4)
    labels = ["BER ↓", "max_use ↓", "load_l1 ↓", "atk wins ↑"]
    for i, name in enumerate(runs):
        ax.plot(xs, norm[i], "o-", label=name, ms=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("normalized (0=worst, 1=best in sample)")
    ax.legend(fontsize=7, loc="upper right")
    _banner(ax, "TYPE: Parallel coordinates",
            "Compare many metrics across runs — good for ablation overview")
    _save(fig, "18_parallel_coordinates.png", "Parallel coordinates", "RQ3 overview")


# ── 19 Slope chart (epoch continuation) ─────────────────────────────────────
def fig_19_slope_epochs() -> None:
    epochs = ["ep20\n(R0)", "ep38", "ep52", "ep60"]
    ber = [0.32, 0.27, 0.12, 0.73]
    l1 = [0.08, 0.17, 0.45, 0.40]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(epochs, ber, "o-", color="#2E8B57", lw=2, label="BER (%)")
    ax2.plot(epochs, l1, "s--", color="#C0392B", lw=2, label="load_l1")
    ax1.set_ylabel("Clean BER (%)", color="#2E8B57")
    ax2.set_ylabel("load_l1", color="#C0392B")
    ax1.annotate("best BER\nworst routing", xy=(2, 0.12), xytext=(2.3, 0.5),
                 arrowprops=dict(arrowstyle="->", color="#888"), fontsize=7)
    _banner(ax1, "TYPE: Slope / dual-axis epoch chart",
            "Shows continuation trap — BER and routing diverge after ep20")
    _save(fig, "19_continuation_slope.png", "Continuation slope", "RQ3 duration")


# ── 20 Radar / spider (single run profile) ────────────────────────────────────
def fig_20_radar_profile() -> None:
    cats = ["Clean\nBER", "Gaussian", "JPEG", "Crop", "Routing\nhealth"]
    # invert BER so higher = better on all axes
    r0 = np.array([99.7, 91.4, 54.9, 59.2, 92.0])  # last = synthetic routing score
    hidden = np.array([99.8, 84.3, 59.2, 53.1, 100.0])
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    r0 = np.concatenate([r0, [r0[0]]])
    hidden = np.concatenate([hidden, [hidden[0]]])
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.plot(angles, hidden, "o-", color="#5B7FA5", label="HiDDeN")
    ax.fill(angles, hidden, alpha=0.1, color="#5B7FA5")
    ax.plot(angles, r0, "o-", color="#2E8B57", label="MoE R0")
    ax.fill(angles, r0, alpha=0.1, color="#2E8B57")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(40, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("TYPE: Radar / spider chart\n(single-run profile)", fontsize=10, fontweight="bold", pad=20)
    fig.text(0.5, 0.02, "Shape = strength profile — good for one-slide model summary", ha="center", fontsize=7.5, style="italic")
    _save(fig, "20_radar_profile.png", "Radar profile", "Summary slide / intro to Results")


# ── 21 Dumbbell chart (paired comparison) ─────────────────────────────────────
def fig_21_dumbbell() -> None:
    df = pd.read_csv(ATTACK)
    df = df[df.attack != "combined"].head(6)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(df))
    h = df["hidden_bit_acc"].values * 100
    m = df["moe_bit_acc"].values * 100
    for i in range(len(df)):
        ax.plot([h[i], m[i]], [i, i], "k-", lw=1.5, zorder=1)
        ax.plot(h[i], i, "o", color="#5B7FA5", ms=8, zorder=2)
        ax.plot(m[i], i, "o", color="#2E8B57", ms=8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["attack"].str.upper())
    ax.set_xlabel("Bit accuracy (%)")
    ax.invert_yaxis()
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#5B7FA5", markersize=8, label="HiDDeN"),
                       plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E8B57", markersize=8, label="MoE")],
              fontsize=8)
    _banner(ax, "TYPE: Dumbbell / connected dot chart",
            "Emphasises paired shift per attack — cleaner than grouped bars for 6–8 items")
    _save(fig, "21_dumbbell_chart.png", "Dumbbell chart", "RQ2 alternative")


# ── README ────────────────────────────────────────────────────────────────────
def write_readme() -> None:
    lines = [
        "# Figure type gallery\n",
        "Generated examples using your MoE thesis data. Pick a **type** for each section.\n",
        "| File | Type | When to use |\n",
        "|------|------|-------------|\n",
    ]
    for fname, typ, when in INDEX:
        lines.append(f"| `{fname}` | {typ} | {when} |\n")
    lines.append("\n## Categories\n\n")
    lines.append("- **Explain setup:** 01, 02, 03, 04, 13, 14, 15\n")
    lines.append("- **Show results:** 05, 06, 07, 11, 21\n")
    lines.append("- **Explain mechanism:** 08, 09, 10, 12, 19\n")
    lines.append("- **Summarise many runs:** 17, 18, 20\n")
    lines.append("- **Distributions:** 16\n")
    (OUT / "README.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Generating figure gallery → {OUT}\n")
    fig_01_pipeline()
    fig_02_warmstart()
    fig_03_ablation_lattice()
    fig_04_timeline()
    fig_05_signed_advantage()
    fig_06_attack_families()
    fig_07_heatmap()
    fig_08_tradeoff_scatter()
    fig_09_expert_load_stacked()
    fig_10_routing_over_epochs()
    fig_11_win_loss()
    fig_12_failure_quadrant()
    fig_13_checkpoint_flow()
    fig_14_attack_taxonomy()
    fig_15_protocol_box()
    fig_16_psnr_distribution()
    fig_17_forest_effects()
    fig_18_parallel_coords()
    fig_19_slope_epochs()
    fig_20_radar_profile()
    fig_21_dumbbell()
    write_readme()
    print(f"\nDone: {len(INDEX)} examples + README.md")


if __name__ == "__main__":
    main()
