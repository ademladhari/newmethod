"""
Generate publication-quality table figures grouped by experimental dimension.
Each group gets its own PNG saved to thesis/figures/tables/.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis" / "figures" / "tables"
OUT_NEW = ROOT / "thesis" / "figures" / "new_figures"
OUT.mkdir(parents=True, exist_ok=True)
OUT_NEW.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str, dpi: int = 200) -> None:
    """Save to tables/ and new_figures/ (thesis uses new_figures/)."""
    for dest in (OUT, OUT_NEW):
        path = dest / filename
        fig.savefig(path, dpi=dpi, facecolor="white")
        print(f"Saved: {path}")

ATK_DIR = Path("results/ber_500_attacks")
ATTACK_SOFT = Path("results/comparison_hidden_vs_moe/attack_summary_main_ep20_soft_router.csv")
WIN_PP = 1.5

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "#2ca02c"
RED    = "#d62728"
ORANGE = "#ff7f0e"
BLUE   = "#1f77b4"
GREY   = "#888888"
LGREY  = "#dddddd"

def cell_color(val, lo, hi, reverse=False):
    """Return background color on a white→green gradient."""
    if val is None or val == "":
        return "white"
    try:
        v = float(str(val).replace("%","").replace("−","-").replace("–","-"))
    except Exception:
        return "white"
    t = (v - lo) / (hi - lo + 1e-9)
    t = max(0, min(1, t))
    if reverse:
        t = 1 - t
    r = 1.0
    g = 0.55 + 0.45 * t
    b = 0.55 + 0.45 * t if t < 0.5 else 0.55 + 0.45 * (1 - t)
    return (r * (1-t) + 0.18 * t, g * (1-t) + 0.8 * t, b * (1-t) + 0.18 * t)

def make_table_fig(title, col_headers, rows, row_colors=None,
                   col_widths=None, highlight_row=0, filename="table.png",
                   figsize=None, fontsize=9):
    n_rows = len(rows)
    n_cols = len(col_headers)
    if figsize is None:
        figsize = (max(10, n_cols * 1.6), 0.55 * (n_rows + 2) + 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    cell_text = [list(r) for r in rows]
    cell_colors_arr = []
    for i, row in enumerate(rows):
        row_c = []
        for j, val in enumerate(row):
            if row_colors and i < len(row_colors) and j < len(row_colors[i]):
                row_c.append(row_colors[i][j])
            elif i == highlight_row:
                row_c.append("#fff9c4")   # light yellow for primary row
            else:
                row_c.append("white")
        cell_colors_arr.append(row_c)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors_arr,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(n_rows + 1):
                tbl[i, j].set_width(w)
    # header style
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.tight_layout(rect=[0, 0.10, 1, 0.98])
    fig.text(
        0.5, 0.02, title,
        ha="center", va="bottom", fontsize=10, fontweight="bold", wrap=True,
    )
    for dest in (OUT, OUT_NEW):
        dest_path = dest / filename
        fig.savefig(dest_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {dest_path}")
    plt.close(fig)
    return OUT / filename

# ══════════════════════════════════════════════════════════════════════════════
# 1. CLEAN-CHANNEL FIDELITY — R0 vs HiDDeN
# ══════════════════════════════════════════════════════════════════════════════
make_table_fig(
    title="Table 1  ·  Clean-Channel Fidelity: HiDDeN vs MoE R0 (identity channel, 500 images)",
    col_headers=["Model", "BER (%)", "Bit Accuracy (%)", "Enc. MSE", "PSNR (dB)", "Backbone", "Epochs"],
    rows=[
        ["HiDDeN ep177",    "0.31", "99.69", "0.0026", "~35", "frozen (HiDDeN)", "177"],
        ["MoE R0 (reported)", "0.32", "99.68", "0.0026", "~33", "unfrozen", "20"],
    ],
    highlight_row=1,
    filename="01_clean_channel_fidelity.png",
    figsize=(12, 2.8),
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. ATTACK ROBUSTNESS — R0 vs HiDDeN (main result)
# ══════════════════════════════════════════════════════════════════════════════
attacks = ["identity","jpeg","quant","crop","cropout","dropout","resize","gaussian","combined"]
hidden_ba = [99.65, 57.95, 99.75, 53.04, 85.59, 87.30, 75.76, 76.24, 50.01]
r0_ba     = [99.71, 51.99, 99.62, 56.15, 83.75, 92.03, 73.16, 91.75, 49.98]
delta     = [r - h for r, h in zip(r0_ba, hidden_ba)]

atk_rows = []
atk_colors = []
for i, atk in enumerate(attacks):
    d = delta[i]
    d_str = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
    d_color = "#c8e6c9" if d > 0.5 else ("#ffcdd2" if d < -0.5 else "#fff9c4")
    atk_rows.append([atk.capitalize(), f"{hidden_ba[i]:.2f}", f"{r0_ba[i]:.2f}", d_str])
    atk_colors.append(["white", "white", "white", d_color])

make_table_fig(
    title="Table 2  ·  Attack Robustness: HiDDeN ep177 vs MoE R0 (bit accuracy %, 512 images/attack, clean-trained)",
    col_headers=["Attack", "HiDDeN BA (%)", "MoE R0 BA (%)", "Δ (MoE−HiDDeN)"],
    rows=atk_rows,
    row_colors=atk_colors,
    highlight_row=-1,
    filename="02_attack_robustness_r0_vs_hidden.png",
    figsize=(10, 5.5),
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKBONE COMPARISON (batch 128, N=4, k=1)
# ══════════════════════════════════════════════════════════════════════════════
make_table_fig(
    title="Table 3  ·  Backbone Freezing: batch=128, N=4, k=1, COCO-100k",
    col_headers=["Run", "Backbone", "BER ep20 (%)", "BER final (%)", "Final ep", "max_use", "load_l1"],
    rows=[
        ["R0 (reported)",      "unfrozen", "0.32", "0.32", "20", "0.306", "0.082"],
        ["Frozen sym_bal08",   "frozen",   "2.72", "7.04", "30", "0.638", "0.910"],
    ],
    highlight_row=0,
    filename="03_backbone_comparison_b128.png",
    figsize=(13, 2.8),
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. ROUTING SPARSITY k — bar-chart figure (unfrozen, N=4, batch 128)
# ══════════════════════════════════════════════════════════════════════════════
ROUTING_K_ROWS = [
    ("top-1 (R0, ep20)",  0.32, 0.306, 0.082, "#2ca02c", "Selected"),
    ("top-2 (k=2, ep30)", 2.66, 0.324, 0.140, "#d62728", "Failed"),
    ("dense (k=4, ep20)", 2.62, 0.250, 0.000, "#d62728", "Uniform routing"),
]


def plot_routing_sparsity_figure(filename="04_routing_sparsity_k.png"):
    """Horizontal BER + routing health for k ∈ {1, 2, 4}."""
    names = [r[0] for r in ROUTING_K_ROWS]
    bers = [r[1] for r in ROUTING_K_ROWS]
    mus = [r[2] for r in ROUTING_K_ROWS]
    lls = [r[3] for r in ROUTING_K_ROWS]
    colors = [r[4] for r in ROUTING_K_ROWS]
    outcomes = [r[5] for r in ROUTING_K_ROWS]
    n = len(names)
    y = np.arange(n)

    fig = plt.figure(figsize=(10.0, 3.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.26)
    ax_ber = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])

    bars = ax_ber.barh(y, bers, color=colors, height=0.55, edgecolor="#333333", linewidth=0.6)
    ax_ber.set_yticks(y)
    ax_ber.set_yticklabels(names, fontsize=11)
    ax_ber.invert_yaxis()
    ax_ber.set_xlabel("Clean-channel BER (%)", fontsize=11, fontweight="bold")
    ax_ber.set_xlim(0, max(bers) * 1.25)
    for bar, ber, out in zip(bars, bers, outcomes):
        ax_ber.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{ber:.2f}%  ·  {out}",
            va="center", ha="left", fontsize=9.5, color="#222222",
        )
    ax_ber.set_title("Validation BER", fontsize=12, fontweight="bold", pad=8)
    ax_ber.grid(axis="x", alpha=0.25)
    ax_ber.spines["top"].set_visible(False)
    ax_ber.spines["right"].set_visible(False)
    ax_ber.tick_params(axis="x", labelsize=10)

    w = 0.34
    ax_rt.barh(y - w / 2, mus, height=w, color="#5b9bd5", label="max_use", edgecolor="#333", linewidth=0.4)
    ax_rt.barh(y + w / 2, lls, height=w, color="#ed7d31", label="load_l1", edgecolor="#333", linewidth=0.4)
    ax_rt.set_yticks(y)
    ax_rt.set_yticklabels([""] * n)
    ax_rt.invert_yaxis()
    ax_rt.set_xlabel("Routing metric value", fontsize=11, fontweight="bold")
    ax_rt.set_xlim(0, 1.05)
    ax_rt.set_title("Routing health", fontsize=12, fontweight="bold", pad=8)
    ax_rt.grid(axis="x", alpha=0.25)
    ax_rt.spines["top"].set_visible(False)
    ax_rt.spines["right"].set_visible(False)
    ax_rt.tick_params(axis="x", labelsize=10)
    leg = ax_rt.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    for t in leg.get_texts():
        t.set_fontsize(9)

    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.14)

    save_figure(fig, filename)
    plt.close(fig)


plot_routing_sparsity_figure()

# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPERT COUNT N — bar-chart figure (unfrozen, k=1, batch 128)
# ══════════════════════════════════════════════════════════════════════════════
EXPERT_N_ROWS = [
    ("R0 · N=4 · 100k (ep20)",      0.32, 0.306, 0.082, "#2ca02c", "Selected"),
    ("8exp A · 100k (ep20)",        1.70, 0.297, 0.526, "#d62728", "Routing gap"),
    ("8exp B · ~240k (ep20)",       1.36, 0.509, 1.114, "#d62728", "Large routing gap"),
]


def plot_expert_count_figure(filename="05_expert_count_N.png"):
    """Horizontal BER + routing health for N ∈ {4, 8} (two 8-exp data scales)."""
    names = [r[0] for r in EXPERT_N_ROWS]
    bers = [r[1] for r in EXPERT_N_ROWS]
    mus = [r[2] for r in EXPERT_N_ROWS]
    lls = [r[3] for r in EXPERT_N_ROWS]
    colors = [r[4] for r in EXPERT_N_ROWS]
    outcomes = [r[5] for r in EXPERT_N_ROWS]
    n = len(names)
    y = np.arange(n)

    fig = plt.figure(figsize=(10.0, 3.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.26)
    ax_ber = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])

    bars = ax_ber.barh(y, bers, color=colors, height=0.55, edgecolor="#333333", linewidth=0.6)
    ax_ber.set_yticks(y)
    ax_ber.set_yticklabels(names, fontsize=11)
    ax_ber.invert_yaxis()
    ax_ber.set_xlabel("Clean-channel BER (%)", fontsize=11, fontweight="bold")
    ax_ber.set_xlim(0, max(bers) * 1.25)
    for bar, ber, out in zip(bars, bers, outcomes):
        ax_ber.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{ber:.2f}%  ·  {out}",
            va="center", ha="left", fontsize=9.5, color="#222222",
        )
    ax_ber.set_title("Validation BER", fontsize=12, fontweight="bold", pad=8)
    ax_ber.grid(axis="x", alpha=0.25)
    ax_ber.spines["top"].set_visible(False)
    ax_ber.spines["right"].set_visible(False)
    ax_ber.tick_params(axis="x", labelsize=10)

    w = 0.34
    ax_rt.barh(y - w / 2, mus, height=w, color="#5b9bd5", label="max_use", edgecolor="#333", linewidth=0.4)
    ax_rt.barh(y + w / 2, lls, height=w, color="#ed7d31", label="load_l1", edgecolor="#333", linewidth=0.4)
    ax_rt.set_yticks(y)
    ax_rt.set_yticklabels([""] * n)
    ax_rt.invert_yaxis()
    ax_rt.set_xlabel("Routing metric value", fontsize=11, fontweight="bold")
    ax_rt.set_xlim(0, 1.2)
    ax_rt.set_title("Routing health", fontsize=12, fontweight="bold", pad=8)
    ax_rt.grid(axis="x", alpha=0.25)
    ax_rt.spines["top"].set_visible(False)
    ax_rt.spines["right"].set_visible(False)
    ax_rt.tick_params(axis="x", labelsize=10)
    leg = ax_rt.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    for t in leg.get_texts():
        t.set_fontsize(9)

    fig.subplots_adjust(left=0.24, right=0.98, top=0.88, bottom=0.14)

    save_figure(fig, filename)
    plt.close(fig)


plot_expert_count_figure()

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING DURATION — bar-chart checkpoints (R0 continuation)
# ══════════════════════════════════════════════════════════════════════════════
VAL_R0 = (
    ROOT / "results/experiments/unfrozen_moe/coco100k/4exp_k1/batch128/"
    "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/validation.csv"
)
VAL_CONT = (
    ROOT / "results/experiments/unfrozen_moe/coco100k/4exp_k1/batch128/"
    "sym_t14_unfrozen_continue_ep60_2026-06-01/validation.csv"
)


def _val_at_epoch(path: Path, epoch: int) -> tuple[float, float, float]:
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if int(row["epoch"]) == epoch:
            return (
                float(row["bitwise-error"]) * 100,
                float(row["expert_max_use"]),
                float(row["train_val_load_l1"]),
            )
    raise KeyError(f"epoch {epoch} not in {path}")


def _duration_outcome(ep: int, ber: float, ll: float, baseline_ber: float) -> tuple[str, str]:
    if ep == 20:
        return "#2ca02c", "Reported stop"
    if ber >= 5.0:
        return "#d62728", "Collapse spike"
    if ber < baseline_ber and ll <= 0.20:
        return "#ff7f0e", "Lower BER, gap rising"
    if ber < baseline_ber:
        return "#ff7f0e", "Lower BER, routing gap high"
    return "#d62728", "Worse than ep20"


DURATION_CHECKPOINTS = (20, 30, 31, 38, 52, 56, 60)


def plot_training_duration_figure(filename="06_training_duration.png"):
    """Horizontal BER + routing health at key continuation checkpoints."""
    baseline_ber, _, _ = _val_at_epoch(VAL_R0, 20)
    rows = []
    for ep in DURATION_CHECKPOINTS:
        src = VAL_R0 if ep == 20 else VAL_CONT
        ber, mu, ll = _val_at_epoch(src, ep)
        color, outcome = _duration_outcome(ep, ber, ll, baseline_ber)
        label = f"ep{ep} · R0 reported" if ep == 20 else f"ep{ep}"
        rows.append((label, ber, mu, ll, color, outcome))

    names = [r[0] for r in rows]
    bers = [r[1] for r in rows]
    mus = [r[2] for r in rows]
    lls = [r[3] for r in rows]
    colors = [r[4] for r in rows]
    outcomes = [r[5] for r in rows]
    n = len(names)
    y = np.arange(n)

    fig = plt.figure(figsize=(10.0, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.26)
    ax_ber = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])

    bars = ax_ber.barh(y, bers, color=colors, height=0.55, edgecolor="#333333", linewidth=0.6)
    ax_ber.set_yticks(y)
    ax_ber.set_yticklabels(names, fontsize=11)
    ax_ber.invert_yaxis()
    ax_ber.set_xlabel("Clean-channel BER (%)", fontsize=11, fontweight="bold")
    ax_ber.set_xlim(0, max(bers) * 1.18)
    for bar, ber, out in zip(bars, bers, outcomes):
        ax_ber.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{ber:.2f}%  ·  {out}",
            va="center", ha="left", fontsize=9, color="#222222",
        )
    ax_ber.set_title("Validation BER", fontsize=12, fontweight="bold", pad=8)
    ax_ber.grid(axis="x", alpha=0.25)
    ax_ber.spines["top"].set_visible(False)
    ax_ber.spines["right"].set_visible(False)

    w = 0.34
    ax_rt.barh(y - w / 2, mus, height=w, color="#5b9bd5", label="max_use", edgecolor="#333", linewidth=0.4)
    ax_rt.barh(y + w / 2, lls, height=w, color="#ed7d31", label="load_l1", edgecolor="#333", linewidth=0.4)
    ax_rt.set_yticks(y)
    ax_rt.set_yticklabels([""] * n)
    ax_rt.invert_yaxis()
    ax_rt.set_xlabel("Routing metric value", fontsize=11, fontweight="bold")
    ax_rt.set_xlim(0, 0.85)
    ax_rt.set_title("Routing health", fontsize=12, fontweight="bold", pad=8)
    ax_rt.grid(axis="x", alpha=0.25)
    ax_rt.spines["top"].set_visible(False)
    ax_rt.spines["right"].set_visible(False)
    ax_rt.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.12)
    save_figure(fig, filename)
    plt.close(fig)


plot_training_duration_figure()

# 7. COLLAPSE ANALYSIS — generated by scripts/plot_thesis_tables_new.py (grouped-header figure)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FULL ATTACK COMPARISON — 14 attacks, BER + Δ BitAcc (HiDDeN vs MoE R0)
# ══════════════════════════════════════════════════════════════════════════════
BENCHMARK_14 = ROOT / "results" / "benchmark_hidden" / "results_combined.csv"
ATTACKS_14 = [
    "identity", "jpeg", "quant", "crop", "cropout", "dropout", "resize",
    "gaussian", "combined", "rotation", "resized_crop", "erasing",
    "brightness", "contrast",
]
ATTACK_LABELS = {
    "identity": "Identity", "jpeg": "JPEG", "quant": "Quant", "crop": "Crop",
    "cropout": "Cropout", "dropout": "Dropout", "resize": "Resize",
    "gaussian": "Gaussian", "combined": "Combined", "rotation": "Rotation",
    "resized_crop": "Resized crop", "erasing": "Erasing",
    "brightness": "Brightness", "contrast": "Contrast",
}

full_rows = []
full_colors = []
if BENCHMARK_14.is_file():
    hidden_ba, moe_ba = {}, {}
    with open(BENCHMARK_14, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            atk = row["attack"]
            ba = float(row["bit_accuracy"])
            if row["method"] == "HiDDeN-ep177":
                hidden_ba[atk] = ba
            elif row["method"] == "MoE-R0-soft":
                moe_ba[atk] = ba
    for atk in ATTACKS_14:
        if atk not in hidden_ba or atk not in moe_ba:
            continue
        h_ber = (1.0 - hidden_ba[atk]) * 100.0
        m_ber = (1.0 - moe_ba[atk]) * 100.0
        delta = moe_ba[atk] - hidden_ba[atk]
        d_str = f"{delta:+.3f}"
        if delta > 0.015:
            d_bg = "#c8e6c9"
        elif delta < -0.015:
            d_bg = "#ffcdd2"
        else:
            d_bg = "#fff9c4"
        full_rows.append([
            ATTACK_LABELS.get(atk, atk),
            f"{h_ber:.2f}",
            f"{m_ber:.2f}",
            d_str,
        ])
        full_colors.append(["white", "white", "white", d_bg])

make_table_fig(
    title="Table 8  ·  HiDDeN vs MoE R0: BER (%) and Δ bit accuracy (14 attacks, 1{,}000 images each)",
    col_headers=["Attack", "HiDDeN BER (%)", "MoE BER (%)", "Δ BitAcc"],
    rows=full_rows,
    row_colors=full_colors,
    highlight_row=-1,
    filename="08_full_attack_comparison.png",
    figsize=(10, 7.5),
    fontsize=9,
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. BATCH 128 ABLATION SUMMARY — readable bar-chart figure
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_ROWS = [
    # name, ber%, max_use, load_l1, color, outcome
    ("R0 primary (selected)",           0.32, 0.306, 0.082, "#2ca02c", "Selected"),
    ("Attack-trained (ep30)",           0.18, 0.359, 0.253, "#7b68ee", "6/9 attack wins vs HiDDeN"),
    ("R0 continued — ep52 (same run)",  0.12, 0.475, 0.451, "#f0ad4e", "Low BER, routing drift"),
    ("top-2 routing (k=2)",             2.66, 0.324, 0.140, "#d62728", "Failed"),
    ("dense routing (k=4)",             2.62, 0.250, 0.000, "#d62728", "Failed"),
    ("8 experts, top-1",                1.70, 0.297, 0.526, "#d62728", "Failed"),
    ("frozen top k=1",                  2.72, 0.552, 0.604, "#d62728", "Failed"),
]


def plot_ablation_summary_figure(filename="09_batch128_ablation_summary.png"):
    """Horizontal BER bars + routing panel — thesis-readable at full textwidth."""
    names = [r[0] for r in ABLATION_ROWS]
    bers = [r[1] for r in ABLATION_ROWS]
    mus = [r[2] for r in ABLATION_ROWS]
    lls = [r[3] for r in ABLATION_ROWS]
    colors = [r[4] for r in ABLATION_ROWS]
    outcomes = [r[5] for r in ABLATION_ROWS]
    n = len(names)
    y = np.arange(n)

    fig = plt.figure(figsize=(10.5, 5.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.26)
    ax_ber = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])

    # ── Left: clean-channel BER ───────────────────────────────────────────────
    bars = ax_ber.barh(y, bers, color=colors, height=0.62, edgecolor="#333333", linewidth=0.6)
    ax_ber.set_yticks(y)
    ax_ber.set_yticklabels(names, fontsize=11)
    ax_ber.invert_yaxis()
    ax_ber.set_xlabel("Clean-channel BER (%)", fontsize=11, fontweight="bold")
    ax_ber.set_xlim(0, max(bers) * 1.22)
    ax_ber.axvline(0.32, color="#2ca02c", ls="--", lw=1.2, alpha=0.55)
    for bar, ber, out in zip(bars, bers, outcomes):
        ax_ber.text(
            bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2,
            f"{ber:.2f}%  ·  {out}",
            va="center", ha="left", fontsize=9.5, color="#222222",
        )
    ax_ber.tick_params(axis="x", labelsize=10)
    ax_ber.set_title("Clean-channel BER by configuration", fontsize=12, fontweight="bold", pad=8)
    ax_ber.grid(axis="x", alpha=0.25, linestyle="-")
    ax_ber.spines["top"].set_visible(False)
    ax_ber.spines["right"].set_visible(False)

    # ── Right: routing health (max_use + load_l1) ─────────────────────────────
    w = 0.36
    ax_rt.barh(y - w / 2, mus, height=w, color="#5b9bd5", label="max_use", edgecolor="#333", linewidth=0.4)
    ax_rt.barh(y + w / 2, lls, height=w, color="#ed7d31", label="load_l1", edgecolor="#333", linewidth=0.4)
    ax_rt.set_yticks(y)
    ax_rt.set_yticklabels([""] * n)
    ax_rt.invert_yaxis()
    ax_rt.set_xlabel("Routing metric value", fontsize=11, fontweight="bold")
    ax_rt.set_xlim(0, 1.05)
    ax_rt.set_title("Routing health", fontsize=12, fontweight="bold", pad=8)
    ax_rt.tick_params(axis="x", labelsize=10)
    ax_rt.grid(axis="x", alpha=0.25)
    ax_rt.spines["top"].set_visible(False)
    ax_rt.spines["right"].set_visible(False)
    leg = ax_rt.legend(
        loc="upper right",
        fontsize=9,
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=1.1,
        handleheight=0.65,
        labelspacing=0.35,
        borderaxespad=0.5,
    )
    for t in leg.get_texts():
        t.set_fontsize(9)

    fig.subplots_adjust(left=0.26, right=0.98, top=0.90, bottom=0.12)

    save_figure(fig, filename)
    plt.close(fig)


plot_ablation_summary_figure()

print(f"\nAll tables generated in {OUT} and {OUT_NEW}")
