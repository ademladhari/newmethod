"""
Publication-quality figure: routing collapse score vs clean-channel BER.
3-panel layout:
  Left  — COCO-100k only (strong correlation; tells the main story)
  Centre — All 28 runs (broken by small-dataset failure mode)
  Right  — Per-failure-mode annotation strip
"""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── load data ─────────────────────────────────────────────────────────────────
exp_root = Path("results/experiments")
rows = []
for val_csv in sorted(exp_root.rglob("validation.csv")):
    parts = str(val_csv.parent).replace("\\", "/").split("results/experiments/")[-1]
    backbone = "unfrozen" if "unfrozen_moe" in parts else "frozen"
    n_match  = re.search(r"(\d+)exp", parts)
    b_match  = re.search(r"batch(\d+)", parts)
    k_match  = re.search(r"_k(\d+)", parts)
    N = int(n_match.group(1)) if n_match else 4
    B = int(b_match.group(1)) if b_match else 32
    k = int(k_match.group(1)) if k_match else 1
    dataset = "coco20k" if "coco20k" in parts else "coco100k"
    run_name = parts.split("/")[-1]
    try:
        with open(val_csv) as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        last = data[-1]
        ber     = float(last.get("bitwise-error", "nan")) * 100
        maxuse  = float(last.get("expert_max_use",
                         last.get("val_expert_max_use", "nan")))
        ep      = int(last.get("epoch", 0))
        if np.isnan(ber) or np.isnan(maxuse):
            continue
        rows.append(dict(B=B, bb=backbone, N=N, k=k, ep=ep,
                         ber=ber, maxuse=maxuse, collapse=maxuse * N,
                         dataset=dataset, run=run_name))
    except Exception:
        pass

coco100k = [r for r in rows if r["dataset"] == "coco100k"]
coco20k  = [r for r in rows if r["dataset"] == "coco20k"]

# Pearson r — 100k only
xs100 = np.array([r["collapse"] for r in coco100k])
ys100 = np.array([r["ber"]      for r in coco100k])
r100, p100 = stats.pearsonr(xs100, ys100)

# Pearson r — all runs
xsall = np.array([r["collapse"] for r in rows])
ysall = np.array([r["ber"]      for r in rows])
rall, pall = stats.pearsonr(xsall, ysall)

# find R0
r0 = next(r for r in rows
          if r["bb"] == "unfrozen" and r["N"] == 4 and r["k"] == 1
          and round(r["ber"], 2) == 0.32)

# ── colour helper ─────────────────────────────────────────────────────────────
UNFROZEN_C = "#2ca02c"
FROZEN_C   = "#d62728"
SMALL_C    = "#ff7f0e"   # 20k runs get a distinct warm orange

def dot_props(r):
    if r["dataset"] == "coco20k":
        c, m = SMALL_C, "s"
    elif r["bb"] == "unfrozen":
        c, m = UNFROZEN_C, "o"
    else:
        c, m = FROZEN_C, "o"
    sz = max(40, min(r["B"] * 1.2, 200))
    alpha = 0.85
    return c, m, sz, alpha

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.8], wspace=0.38)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

fig.suptitle("Does Routing Collapse Predict Watermark Failure?  —  All 28 Runs",
             fontsize=13, fontweight="bold", y=1.02)


# ════════════════════════════════════════════════════════════════════════════════
# Panel A — COCO-100k only (strong signal)
# ════════════════════════════════════════════════════════════════════════════════
for r in coco100k:
    c, m, s, a = dot_props(r)
    ax1.scatter(r["collapse"], r["ber"], c=c, marker=m, s=s, alpha=a,
                linewidths=0.7, edgecolors="k", zorder=3)

# OLS trend
m_fit, b_fit, *_ = stats.linregress(xs100, ys100)
x_line = np.linspace(xs100.min() - 0.05, xs100.max() + 0.05, 300)
ax1.plot(x_line, m_fit * x_line + b_fit, "--", color="#555", lw=1.8,
         label=f"OLS  ($r={r100:.2f}$, $p={p100:.3f}$)", zorder=2)

# regime shading
ax1.axvspan(0.9, 1.5,  alpha=0.08, color="#2ca02c")
ax1.axvspan(1.5, 2.8,  alpha=0.07, color="orange")
ax1.axvspan(3.5, xs100.max() + 0.4, alpha=0.08, color="#d62728")
ax1.axvline(1.0, color="#2ca02c", lw=0.9, ls=":", alpha=0.6)

# R0 annotation
ax1.annotate("R0\n(best run)", xy=(r0["collapse"], r0["ber"]),
             xytext=(r0["collapse"] + 0.35, r0["ber"] + 1.8),
             fontsize=8.5, fontweight="bold", color=UNFROZEN_C,
             arrowprops=dict(arrowstyle="->", color=UNFROZEN_C, lw=1.2))

# worst run annotation
worst100 = max(coco100k, key=lambda r: r["ber"])
ax1.annotate(f"BER={worst100['ber']:.0f}%\n(8-exp, frozen)", 
             xy=(worst100["collapse"], worst100["ber"]),
             xytext=(worst100["collapse"] - 1.2, worst100["ber"] - 3),
             fontsize=7.5, color=FROZEN_C,
             arrowprops=dict(arrowstyle="->", color=FROZEN_C, lw=0.9))

ax1.set_xlabel("Collapse score  (max\_use × N)", fontsize=10.5)
ax1.set_ylabel("Clean-channel BER (%)", fontsize=10.5)
ax1.set_title(f"(a) COCO-100k runs only  ($n={len(coco100k)}$)\n"
              f"Pearson $r={r100:.2f}$,  $p={p100:.3f}$",
              fontsize=10, pad=8)
ax1.legend(fontsize=8.5, loc="upper left")
ax1.grid(True, alpha=0.28)
ax1.set_xlim(0.7, xs100.max() + 0.3)
ax1.set_ylim(-0.5, max(ys100) * 1.1)

# ════════════════════════════════════════════════════════════════════════════════
# Panel B — all 28 runs (correlation breaks down)
# ════════════════════════════════════════════════════════════════════════════════
for r in rows:
    c, m, s, a = dot_props(r)
    ax2.scatter(r["collapse"], r["ber"], c=c, marker=m, s=s, alpha=a,
                linewidths=0.7, edgecolors="k", zorder=3)

m_fit2, b_fit2, *_ = stats.linregress(xsall, ysall)
x_line2 = np.linspace(xsall.min() - 0.05, xsall.max() + 0.05, 300)
ax2.plot(x_line2, m_fit2 * x_line2 + b_fit2, "--", color="#555", lw=1.8,
         label=f"OLS all runs ($r={rall:.2f}$, $p={pall:.3f}$)", zorder=2)

# circle the coco20k cluster
from matplotlib.patches import Ellipse
small_xs = np.array([r["collapse"] for r in coco20k])
small_ys = np.array([r["ber"] for r in coco20k])
if len(small_xs):
    cx, cy = small_xs.mean(), small_ys.mean()
    w = small_xs.ptp() + 0.6
    h = small_ys.ptp() + 5
    ell = Ellipse((cx, cy), width=w, height=h, angle=0,
                  edgecolor=SMALL_C, facecolor="none", lw=2, ls="--", zorder=4)
    ax2.add_patch(ell)
    ax2.annotate("COCO-20k runs:\nlow collapse,\nhigh BER\n(dataset too small)",
                 xy=(cx + w / 2, cy),
                 xytext=(cx + w / 2 + 0.5, cy - 6),
                 fontsize=7.5, color=SMALL_C,
                 arrowprops=dict(arrowstyle="->", color=SMALL_C, lw=0.9))

ax2.set_xlabel("Collapse score  (max\_use × N)", fontsize=10.5)
ax2.set_ylabel("Clean-channel BER (%)", fontsize=10.5)
ax2.set_title(f"(b) All 28 runs\n"
              f"Pearson $r={rall:.2f}$,  $p={pall:.3f}$  (signal diluted by 20k runs)",
              fontsize=10, pad=8)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.grid(True, alpha=0.28)
ax2.set_xlim(0.5, xsall.max() + 0.3)
ax2.set_ylim(-0.5, max(ysall) * 1.1)

# ════════════════════════════════════════════════════════════════════════════════
# Panel C — BER by failure-mode category (box / strip)
# ════════════════════════════════════════════════════════════════════════════════
def failure_mode(r):
    if r["dataset"] == "coco20k":
        return "Small\ndataset"
    if r["collapse"] > 3.5:
        return "Full\ncollapse"
    if r["collapse"] > 1.8:
        return "Partial\ncollapse"
    if r["bb"] == "unfrozen":
        return "Healthy\n(unfrozen)"
    return "Healthy\n(frozen)"

cats  = ["Healthy\n(unfrozen)", "Healthy\n(frozen)", "Partial\ncollapse",
         "Full\ncollapse", "Small\ndataset"]
cat_colors = {
    "Healthy\n(unfrozen)": UNFROZEN_C,
    "Healthy\n(frozen)":   "#8B4513",
    "Partial\ncollapse":   "orange",
    "Full\ncollapse":      FROZEN_C,
    "Small\ndataset":      SMALL_C,
}
from collections import defaultdict
cat_data = defaultdict(list)
for r in rows:
    cat_data[failure_mode(r)].append(r["ber"])

# jitter strip + box
np.random.seed(42)
for i, cat in enumerate(cats):
    vals = np.array(cat_data[cat])
    if not len(vals):
        continue
    jitter = np.random.normal(0, 0.12, len(vals))
    ax3.scatter(np.full(len(vals), i) + jitter, vals,
                c=cat_colors[cat], alpha=0.75, s=70, zorder=4,
                linewidths=0.5, edgecolors="k")
    # median line
    ax3.hlines(np.median(vals), i - 0.35, i + 0.35,
               colors=cat_colors[cat], lw=2.5, zorder=5)
    # IQR box
    q25, q75 = np.percentile(vals, [25, 75])
    ax3.fill_betweenx([q25, q75], i - 0.3, i + 0.3,
                      color=cat_colors[cat], alpha=0.2, zorder=3)

ax3.set_xticks(range(len(cats)))
ax3.set_xticklabels(cats, fontsize=8.5)
ax3.set_ylabel("Clean-channel BER (%)", fontsize=10.5)
ax3.set_title("(c) BER by failure-mode\ncategory (median + IQR)", fontsize=10, pad=8)
ax3.grid(True, alpha=0.28, axis="y")
ax3.set_xlim(-0.7, len(cats) - 0.3)
ax3.set_ylim(-0.5, max(ysall) * 1.1)

# add median values
for i, cat in enumerate(cats):
    vals = np.array(cat_data[cat])
    if not len(vals):
        continue
    med = np.median(vals)
    ax3.text(i, med + 0.6, f"{med:.1f}%", ha="center", va="bottom",
             fontsize=7.5, color=cat_colors[cat], fontweight="bold")

# ── shared legend ─────────────────────────────────────────────────────────────
leg_handles = [
    mpatches.Patch(color=UNFROZEN_C, label="Unfrozen backbone (COCO-100k)"),
    mpatches.Patch(color=FROZEN_C,   label="Frozen backbone (COCO-100k)"),
    mpatches.Patch(color=SMALL_C,    label="COCO-20k exploratory runs"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
               markersize=13, label="Batch 128"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
               markersize=8,  label="Batch ≤ 32"),
]
fig.legend(handles=leg_handles, loc="lower center", ncol=5,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.07), frameon=True)

fig.tight_layout()
out = Path("thesis/figures/rq/collapse_vs_ber_all_runs.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}  ({out.stat().st_size // 1024} KB)")
