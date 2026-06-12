"""Per-run training curves (validation BER) + combined overlay.

Usage (repo root):
  py -3.10 scripts/plot_training_curves.py

Outputs:
  thesis/figures/training_curves/individual/{slug}.png
  thesis/figures/training_curves/combined_thesis_ber.png
  thesis/figures/training_curves/combined_all_ber.png
  thesis/figures/training_curves/combined_thesis_max_use.png
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "thesis" / "figures" / "training_curves"
OUT_NEW = ROOT / "thesis" / "figures" / "new_figures"
OUT_IND = OUT / "individual"
OUT_NEW.mkdir(parents=True, exist_ok=True)

KEY_BER = "bitwise-error"
KEY_MU = "expert_max_use"
KEY_LL = "train_val_load_l1"
KEY_EFF = "effective_experts"

THESIS_LABELS = [
    "R0 PRIMARY",
    "R0 continuation",
    "Attack-trained",
    "200k data scale",
    "k=2 ablation",
    "k=4 dense",
    "8exp run A",
    "8exp run B",
    "Frozen bal08 ep20",
    "Frozen bal08 ep30",
    "Routing isolation",
    "8exp collapse diag",
    "Frozen t14 temp0.9",
]

# Reported thesis experiments only (Table 9 batch-128 ablations + data-scale).
# Excludes legacy sweeps and batch-32 diagnostics (routing isolation, collapse diag, temp0.9).
THESIS_REPORTED = [
  # group "r0": reported model + continuation (same lineage)
    ("sym_t14_unfrozen_bal004warm10", "R0 primary (ep 1–20)", {"color": "#111111", "lw": 2.6, "ls": "-", "zorder": 10, "group": "r0"}),
    ("sym_t14_unfrozen_continue", "R0 continued (ep 1–60)", {"color": "#e65100", "lw": 2.0, "ls": "-", "zorder": 9, "group": "r0"}),
  # group "4exp": other N=4 ablations (no R0 / continue)
    ("sym_t14_unfrozen_attack", "Attack-trained", {"color": "#6a1b9a", "lw": 2.4, "ls": "-", "group": "4exp"}),
    ("unfrozen_4exp_top2", "top-2 (k=2)", {"color": "#1565c0", "lw": 2.2, "ls": "-", "group": "4exp"}),
    ("unfrozen_4exp_dense_k4", "dense (k=4)", {"color": "#0288d1", "lw": 2.2, "ls": "-", "group": "4exp"}),
    ("sym_bal08_jitter0_bal08warm5_temp14to10_ep20", "Frozen top-1 (20 ep)", {"color": "#c62828", "lw": 2.2, "ls": "--", "group": "4exp"}),
    ("sym_bal08_jitter0_bal08warm5_temp14to10_ep30", "Frozen top-1 (30 ep)", {"color": "#ef6c00", "lw": 2.2, "ls": "-.", "group": "4exp"}),
    ("sym_t14_unfrozen_200k", "200k data scale", {"color": "#2e7d32", "lw": 2.2, "ls": (0, (3, 1, 1, 1)), "group": "4exp"}),
  # group "8exp": N=8 experts (RQ3 expert-count ablation)
    ("unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02", "N=8 run A", {"color": "#b71c1c", "lw": 2.0, "ls": "-", "group": "8exp"}),
    ("unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03", "N=8 run B", {"color": "#ef5350", "lw": 2.0, "ls": "-", "group": "8exp"}),
]

# Figure used in thesis §backbone freezing (fig:4exp-curves): R0 vs frozen only.
FIG_4EXP_BACKBONE = [
    ("sym_t14_unfrozen_bal004warm10", "R0 (unfrozen)", {"color": "#2ca02c", "lw": 2.8, "ls": "-"}),
    ("sym_bal08_jitter0_bal08warm5_temp14to10_ep20", "Frozen top-1", {"color": "#c62828", "lw": 2.4, "ls": "--"}),
]

THESIS_GROUPS = {
    "r0": {
        "filename": "combined_r0_continue_ber_loadl1",
        "title": "Training curves — R0 primary and continuation (BER + load L1)",
        "subtitle": "Reported R0 (epochs 1–20) and the same model continued to epoch 60.",
    },
    "4exp": {
        "filename": "combined_4exp_ber_loadl1",
        "new_figure": "04_backbone_r0_vs_frozen.png",
        "title": "Training curves — R0 vs frozen backbone",
        "subtitle": "",
    },
    "8exp": {
        "filename": "combined_8exp_ber_loadl1",
        "title": "Training curves — N=8 experts (BER + load L1)",
        "subtitle": "Two independent 8-expert runs (unfrozen, k=1, batch 128, COCO-100k).",
    },
}
OUT_THESIS = OUT / "thesis_experiments"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "axes.edgecolor": "black",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
    }
)


def ffloat(v):
    try:
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def n_exp(parts):
    for p in parts:
        if m := re.match(r"(\d+)exp", p):
            return int(m.group(1))
    return 4


def stability(mu, ll, eff, n):
    if mu is None:
        return None
    c = max(0.0, (mu - 1 / n) / (1 - 1 / n + 1e-9))
    s = 1.0 - 0.5 * min(1.0, c)
    if ll is not None:
        s -= 0.3 * min(1.0, ll / 1.5)
    if eff is not None:
        s += 0.2 * min(1.0, eff / n)
    return max(0.0, min(1.0, s))


def composite(ber_pct, stab):
    if ber_pct is None or stab is None:
        return None
    return 0.6 * max(0, 1 - ber_pct / 25) + 0.4 * stab


def label(run, branch):
    m = [
        ("sym_t14_unfrozen_bal004warm10", "R0 PRIMARY"),
        ("sym_t14_unfrozen_continue", "R0 continuation"),
        ("sym_t14_unfrozen_attack", "Attack-trained"),
        ("sym_t14_unfrozen_200k", "200k data scale"),
        ("unfrozen_4exp_top2", "k=2 ablation"),
        ("unfrozen_4exp_dense_k4", "k=4 dense"),
        ("unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02", "8exp run A"),
        ("unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03", "8exp run B"),
        ("sym_bal08_jitter0_bal08warm5_temp14to10_ep20", "Frozen bal08 ep20"),
        ("sym_bal08_jitter0_bal08warm5_temp14to10_ep30", "Frozen bal08 ep30"),
        ("routing_isolation", "Routing isolation"),
        ("collapse_diag", "8exp collapse diag"),
        ("sym_t14_temp09", "Frozen t14 temp0.9"),
    ]
    for k, v in m:
        if k in run:
            return v
    return "Legacy" if branch == "legacy_early" else run[:36]


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", text.lower()).strip("_")
    return s[:72] or "run"


def best_epoch(rows, n, saved_only=False, saved_eps=None):
    cands = []
    for r in rows:
        ep = int(r["epoch"])
        if saved_only and ep not in saved_eps:
            continue
        ber = ffloat(r[KEY_BER])
        if ber is None:
            continue
        ber_pct = ber * 100
        mu = ffloat(r.get(KEY_MU))
        ll = ffloat(r.get(KEY_LL))
        eff = ffloat(r.get(KEY_EFF))
        stab = stability(mu, ll, eff, n)
        comp = composite(ber_pct, stab)
        cands.append((ep, ber_pct, comp or 0))
    if not cands:
        return None, None
    best = max(cands, key=lambda x: x[2])
    return best[0], best[1]


def load_runs():
    runs = []
    for val_csv in sorted(EXP.rglob("validation.csv")):
        if "images" in str(val_csv):
            continue
        run_dir = val_csv.parent
        parts = run_dir.relative_to(EXP).parts
        if len(parts) < 5:
            continue
        branch, run_name = parts[0], parts[4]
        n = n_exp(parts)
        rows = list(csv.DictReader(open(val_csv, encoding="utf-8")))
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        bers = [ffloat(r[KEY_BER]) * 100 for r in rows]
        mus = [ffloat(r.get(KEY_MU)) for r in rows]
        load_l1 = [ffloat(r.get(KEY_LL)) for r in rows]
        saved = set()
        chk = run_dir / "checkpoints"
        if chk.is_dir():
            for f in chk.glob("*.pyt"):
                if m := re.search(r"epoch-(\d+)", f.name):
                    saved.add(int(m.group(1)))
        best_ep, best_ber = best_epoch(rows, n, saved_only=False)
        saved_ep, saved_ber = best_epoch(rows, n, saved_only=True, saved_eps=saved) if saved else (None, None)
        lbl = label(run_name, branch)
        runs.append(
            {
                "label": lbl,
                "run_name": run_name,
                "branch": branch,
                "run_dir": run_dir,
                "epochs": epochs,
                "ber": bers,
                "max_use": mus,
                "load_l1": load_l1,
                "final_ep": max(epochs),
                "best_ep": best_ep,
                "best_ber": best_ber,
                "saved_ep": saved_ep,
                "saved_ber": saved_ber,
                "thesis": lbl in THESIS_LABELS,
            }
        )
    return runs


def sort_key(r):
    if r["thesis"]:
        return (0, THESIS_LABELS.index(r["label"]))
    return (1, r["best_ber"] or 999)


def plot_individual(run: dict, out_dir: Path | None = None, label: str | None = None) -> None:
    out_dir = out_dir or OUT_IND
    title = label or run["label"]
    ep = run["epochs"]
    ber = run["ber"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, height_ratios=[2, 1])

    ax = axes[0]
    ax.plot(ep, ber, color="black", linewidth=1.8, marker="o", markersize=3, label="Val BER")
    if run["best_ep"]:
        ax.scatter([run["best_ep"]], [run["best_ber"]], color="0.35", s=50, zorder=5, label=f"Best val ep{run['best_ep']}")
        ax.axvline(run["best_ep"], color="0.55", linestyle=":", linewidth=1)
    if run["saved_ep"] and run["saved_ep"] != run["best_ep"]:
        ax.scatter(
            [run["saved_ep"]],
            [run["saved_ber"]],
            color="0.65",
            s=50,
            marker="s",
            zorder=5,
            label=f"Best saved ep{run['saved_ep']}",
        )
        ax.axvline(run["saved_ep"], color="0.75", linestyle="--", linewidth=1)
    elif run["saved_ep"]:
        ax.axvline(run["saved_ep"], color="0.75", linestyle="--", linewidth=1, label=f"Saved ep{run['saved_ep']}")
    ax.set_ylabel("BER (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, edgecolor="black")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_xlim(0.5, run["final_ep"] + 0.5)

    ax2 = axes[1]
    if any(m is not None for m in run["max_use"]):
        ax2.plot(ep, run["max_use"], color="0.35", linewidth=1.5, marker=".", markersize=3)
        ax2.axhline(0.5, color="0.6", linestyle="--", linewidth=0.8, label="50% collapse")
        ax2.legend(loc="upper right", fontsize=6, frameon=True, edgecolor="black")
    ax2.set_ylabel("expert_max_use")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, linestyle=":", alpha=0.7)

    fig.text(
        0.5,
        0.01,
        f"{run['run_name']}  ·  epochs 1–{run['final_ep']}",
        ha="center",
        fontsize=7,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    stem = slugify(f"{title}_{run['run_name'][:24]}")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_thesis_reported(runs: list[dict], group: str | None = None) -> list[tuple[dict, str, dict]]:
    """Match runs to THESIS_REPORTED order; each item is (run, display_label, style)."""
    out = []
    for pattern, display, style in THESIS_REPORTED:
        if group and style.get("group") != group:
            continue
        for run in runs:
            if pattern in run["run_name"]:
                out.append((run, display, style))
                break
    return out


def select_pattern_runs(runs: list[dict], patterns: list[tuple]) -> list[tuple[dict, str, dict]]:
    """Match runs to an explicit (pattern, label, style) list."""
    out = []
    for pattern, display, style in patterns:
        for run in runs:
            if pattern in run["run_name"]:
                out.append((run, display, style))
                break
    return out


def _smooth(vals: list, window: int = 3) -> list:
    """Light rolling mean for display (reduces per-epoch noise)."""
    arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    if len(arr) < window:
        return arr.tolist()
    kernel = np.ones(window) / window
    sm = np.convolve(arr, kernel, mode="same")
    half = window // 2
    sm[:half] = arr[:half]
    sm[-half:] = arr[-half:]
    return sm.tolist()


def _plot_thesis_lines(ax, reported: list[tuple[dict, str, dict]], ykey: str, mark_r0_best: bool = False) -> list:
    handles = []
    for run, display, style in reported:
        y = run[ykey]
        if not any(v is not None for v in y):
            continue
        (line,) = ax.plot(
            run["epochs"],
            y,
            color=style.get("color", "black"),
            linewidth=style.get("lw", 1.6),
            linestyle=style.get("ls", "-"),
            zorder=style.get("zorder", 5),
            label=display,
        )
        handles.append(line)
        if mark_r0_best and display.startswith("R0 primary") and run["best_ep"]:
            ax.scatter([run["best_ep"]], [run["best_ber"]], color=style["color"], s=40, zorder=11)
    return handles


def plot_thesis_experiments_ber_loadl1(
    reported: list[tuple[dict, str, dict]],
    filename: str,
    title: str,
    subtitle: str = "",
    out_subdir: Path | None = None,
) -> None:
    if not reported:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 9.0), sharex=True, height_ratios=[1.05, 1.0])
    fig.subplots_adjust(left=0.09, right=0.70, top=0.96, bottom=0.10, hspace=0.38)
    ax_ber, ax_ll = axes

    handles = _plot_thesis_lines(ax_ber, reported, "ber", mark_r0_best=True)
    _plot_thesis_lines(ax_ll, reported, "load_l1")

    ax_ber.set_yscale("log")
    ax_ber.set_ylim(0.1, 12.0)
    ax_ber.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax_ber.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax_ber.axhline(1.0, color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=0)
    ax_ll.axhline(0.20, color="#888888", linestyle="--", linewidth=1.2, zorder=0)

    ax_ber.set_ylabel("Validation BER (%) — log scale", fontsize=13, fontweight="bold")
    ax_ber.set_title("(a) Validation BER", fontweight="bold", loc="left", fontsize=13, pad=10)
    ax_ber.tick_params(labelsize=11)
    ax_ll.set_ylabel("train_val_load_l1", fontsize=13, fontweight="bold")
    ax_ll.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax_ll.set_title("(b) Train–validation routing gap (load L1)", fontweight="bold", loc="left", fontsize=13, pad=10)
    ax_ll.set_ylim(-0.02, 1.08)
    ax_ll.tick_params(labelsize=11)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.45, linewidth=0.8)
        ax.set_xlim(left=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    legend_handles = list(handles) + [
        Line2D([0], [0], color="#888888", linestyle="--", linewidth=1.2, label="Healthy threshold (0.20)"),
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.72, 0.5),
            ncol=1,
            fontsize=12,
            frameon=True,
            edgecolor="#666666",
            handlelength=2.6,
            handletextpad=0.7,
            labelspacing=0.55,
        )

    save_dirs = [OUT_THESIS, OUT]
    if out_subdir:
        save_dirs.insert(0, out_subdir)
    for d in save_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{filename}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def plot_4exp_backbone_faceted(
    reported: list[tuple[dict, str, dict]],
    filename: str,
    out_subdir: Path | None = None,
    new_figure: str | None = "04_backbone_r0_vs_frozen.png",
) -> None:
    """R0 vs frozen at 20 epochs: epoch-20 BER bars + load_l1 trajectories."""
    if not reported:
        return

    xmax = 20
    fig, (ax_ber, ax_ll) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.16, wspace=0.32)

    labels, ep20_bers, colors = [], [], []
    for run, display, style in reported:
        ep = [e for e in run["epochs"] if e <= xmax]
        ll = _smooth([v for e, v in zip(run["epochs"], run["load_l1"]) if e <= xmax], window=3)
        c = style.get("color", "#333333")
        lw = style.get("lw", 2.4)
        ls = style.get("ls", "-")
        ax_ll.plot(ep, ll, color=c, linewidth=lw, linestyle=ls, label=display, solid_capstyle="round")
        ber20 = next((b for e, b in zip(run["epochs"], run["ber"]) if e == xmax), None)
        if ber20 is not None:
            labels.append(display)
            ep20_bers.append(ber20)
            colors.append(c)

    x = np.arange(len(labels))
    bars = ax_ber.bar(x, ep20_bers, color=colors, width=0.55, edgecolor="#333333", linewidth=0.7)
    ax_ber.set_xticks(x)
    ax_ber.set_xticklabels(labels, fontsize=10)
    ax_ber.set_ylabel("Validation BER at ep20 (%)", fontsize=12, fontweight="bold")
    ax_ber.set_title("BER at matched budget", fontsize=12, fontweight="bold", pad=8)
    ax_ber.set_ylim(0, max(ep20_bers) * 1.35 if ep20_bers else 3)
    for bar, val in zip(bars, ep20_bers):
        ax_ber.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06, f"{val:.2f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_ber.grid(axis="y", alpha=0.35)
    ax_ber.spines["top"].set_visible(False)
    ax_ber.spines["right"].set_visible(False)
    ax_ber.tick_params(labelsize=10)

    ax_ll.axhline(0.20, color="#aaaaaa", linestyle="--", linewidth=0.9, zorder=0)
    ax_ll.set_ylabel("train_val_load_l1", fontsize=12, fontweight="bold")
    ax_ll.set_xlabel("Epoch", fontsize=11)
    ax_ll.set_title("Routing gap over training", fontsize=12, fontweight="bold", pad=8)
    ax_ll.set_xlim(0.5, xmax + 0.5)
    ax_ll.set_ylim(0, 0.70)
    ax_ll.grid(True, linestyle=":", alpha=0.4)
    ax_ll.tick_params(labelsize=10)
    ax_ll.spines["top"].set_visible(False)
    ax_ll.spines["right"].set_visible(False)
    ax_ll.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95, edgecolor="#cccccc")

    save_dirs = [OUT_THESIS, OUT]
    if out_subdir:
        save_dirs.insert(0, out_subdir)
    for d in save_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(d / f"{filename}.{ext}", dpi=220, facecolor="white")
    if new_figure:
        fig.savefig(OUT_NEW / new_figure, dpi=220, facecolor="white")
        print(f"Saved: {OUT_NEW / new_figure}")
    plt.close(fig)


def plot_thesis_experiments(reported: list[tuple[dict, str, dict]], ykey: str, ylab: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for run, display, style in reported:
        y = run[ykey]
        if ykey == "max_use" and not any(v is not None for v in y):
            continue
        ax.plot(
            run["epochs"],
            y,
            color=style.get("color", "black"),
            linewidth=style.get("lw", 1.6),
            linestyle=style.get("ls", "-"),
            zorder=style.get("zorder", 5),
            label=display,
        )
        if display.startswith("R0 primary"):
            ax.scatter([run["best_ep"]], [run["best_ber"]], color=style["color"], s=40, zorder=11)

    ax.axhline(1.0, color="#cccccc", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.set_xlim(left=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=8,
        frameon=True,
        edgecolor="black",
    )
    fig.text(
        0.5,
        -0.22,
        "Reported MoE experiments (COCO-100k batch-128 ablations + 200k scale). "
        "Diagnostics and legacy sweeps excluded.",
        ha="center",
        fontsize=7,
        style="italic",
    )
    fig.tight_layout()
    OUT_THESIS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_THESIS / f"{filename}.{ext}", bbox_inches="tight", facecolor="white")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined(runs: list[dict], filename: str, title: str, ykey: str = "ber", ylab: str = "Validation BER (%)") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab20
    n = len(runs)
    for i, run in enumerate(runs):
        y = run[ykey]
        if ykey == "max_use" and not any(v is not None for v in y):
            continue
        color = cmap(i % 20)
        ls = "-" if run["thesis"] else "--"
        lw = 1.8 if run["thesis"] else 1.0
        ax.plot(
            run["epochs"],
            y,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=0.9 if run["thesis"] else 0.65,
            label=run["label"],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ncol = 2 if n <= 14 else 3
    ax.legend(loc="upper right", fontsize=6, frameon=True, edgecolor="black", ncol=ncol)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_index(runs: list[dict]) -> None:
    lines = [
        "# Training curves",
        "",
        "Regenerate: `py -3.10 scripts/plot_training_curves.py`",
        "",
        "## Combined",
        "",
        "- `combined_thesis_experiments_ber_loadl1.png` — **BER + load_l1** (all reported experiments)",
        "- `thesis_experiments/r0/combined_r0_continue_ber_loadl1.png` — R0 primary + continuation",
        "- `thesis_experiments/4exp/combined_4exp_ber_loadl1.png` — other N=4 ablations",
        "- `thesis_experiments/8exp/combined_8exp_ber_loadl1.png` — N=8 expert group",
        "- `combined_thesis_experiments_ber.png` — reported thesis experiments only (Table 9 + 200k)",
        "- `combined_thesis_experiments_max_use.png` — same runs, routing metric",
        "- `combined_thesis_ber.png` — all thesis-labelled runs incl. diagnostics (13)",
        "- `combined_all_ber.png` — all runs with validation logs (29)",
        "- `combined_thesis_max_use.png` — routing overlay (13 thesis-labelled)",
        "",
        "## Thesis experiments (`thesis_experiments/`)",
        "",
        "Individual curves for the 10 reported runs only.",
        "",
        "",
        "## Individual (`individual/`)",
        "",
    ]
    for r in sorted(runs, key=sort_key):
        stem = slugify(f"{r['label']}_{r['run_name'][:24]}")
        lines.append(f"- `{stem}.png` — **{r['label']}** (ep 1–{r['final_ep']})")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    runs = sorted(load_runs(), key=sort_key)
    if not runs:
        raise SystemExit("No validation.csv found under results/experiments/")

    OUT_IND.mkdir(parents=True, exist_ok=True)

    for run in runs:
        plot_individual(run)

    reported = select_thesis_reported(runs)
    if reported:
        plot_thesis_experiments_ber_loadl1(
            reported,
            "combined_thesis_experiments_ber_loadl1",
            "Training curves — reported MoE experiments (BER + load L1)",
            "All batch-128 thesis runs: N=4 and N=8 groups combined.",
        )
        for group_id, meta in THESIS_GROUPS.items():
            if group_id == "4exp":
                subset = select_pattern_runs(runs, FIG_4EXP_BACKBONE)
                plot_4exp_backbone_faceted(
                    subset,
                    meta["filename"],
                    out_subdir=OUT_THESIS / group_id,
                    new_figure=meta.get("new_figure"),
                )
                continue
            subset = select_thesis_reported(runs, group=group_id)
            plot_thesis_experiments_ber_loadl1(
                subset,
                meta["filename"],
                meta["title"],
                meta["subtitle"],
                out_subdir=OUT_THESIS / group_id,
            )
        plot_thesis_experiments(
            reported,
            "ber",
            "Validation BER (%)",
            "combined_thesis_experiments_ber",
            "Training curves — reported MoE experiments (validation BER)",
        )
        plot_thesis_experiments(
            reported,
            "max_use",
            "expert_max_use",
            "combined_thesis_experiments_max_use",
            "Training curves — reported MoE experiments (routing collapse)",
        )
        OUT_THESIS.mkdir(parents=True, exist_ok=True)
        for run, display, _ in reported:
            plot_individual(run, out_dir=OUT_THESIS, label=display)

    thesis = [r for r in runs if r["thesis"]]
    plot_combined(
        thesis,
        "combined_thesis_ber",
        "Validation BER — thesis runs (epoch 1 to final)",
    )
    plot_combined(
        runs,
        "combined_all_ber",
        "Validation BER — all experiments (epoch 1 to final)",
    )
    plot_combined(
        thesis,
        "combined_thesis_max_use",
        "Expert max use — thesis runs (epoch 1 to final)",
        ykey="max_use",
        ylab="expert_max_use",
    )

    write_index(runs)
    print(f"Individual curves: {len(runs)} → {OUT_IND}")
    if reported:
        for gid in THESIS_GROUPS:
            n = len(select_thesis_reported(runs, group=gid))
            print(f"  {gid} group: {n} → {OUT_THESIS / gid / f'combined_{gid}_ber_loadl1.png' if gid != 'r0' else OUT_THESIS / gid / 'combined_r0_continue_ber_loadl1.png'}")
        print(f"Thesis BER+load_l1 (all): {len(reported)} → {OUT / 'combined_thesis_experiments_ber_loadl1.png'}")
        print(f"Thesis experiments only: {len(reported)} → {OUT / 'combined_thesis_experiments_ber.png'}")
    print(f"Combined (thesis+diag): {len(thesis)} runs → {OUT / 'combined_thesis_ber.png'}")
    print(f"Combined (all): {len(runs)} runs → {OUT / 'combined_all_ber.png'}")


if __name__ == "__main__":
    main()
