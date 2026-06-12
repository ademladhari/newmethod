"""Training-duration rationale: BER + routing-gap trajectories (no table panel).

Usage (repo root):
  py -3 scripts/plot_training_duration_figure.py

Output:
  thesis/figures/training_curves/duration/training_duration_rationale.png
  thesis/figures/new_figures/training_duration_rationale.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments" / "unfrozen_moe" / "coco100k" / "4exp_k1" / "batch128"
OUT_DURATION = ROOT / "thesis" / "figures" / "training_curves" / "duration"
OUT_NEW = ROOT / "thesis" / "figures" / "new_figures"

R0_VAL = EXP / "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01" / "validation.csv"
CONT_VAL = EXP / "sym_t14_unfrozen_continue_ep60_2026-06-01" / "validation.csv"

SPIKE_EPOCHS = (30, 58)
STOP_EPOCH = 20
GREEN = "#2e7d32"
GREEN_TEXT = "#1b5e20"
RED = "#c62828"
RED_TEXT = "#b71c1c"


def load_series(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return {
        "epoch": np.array([int(r["epoch"]) for r in rows]),
        "ber": np.array([float(r["bitwise-error"]) * 100 for r in rows]),
        "load_l1": np.array([float(r["train_val_load_l1"]) for r in rows]),
    }


def _shade_regions(ax, xmax: float) -> None:
    ax.axvspan(0.5, 20.5, color="#e8f5e9", alpha=0.35, zorder=0)
    ax.axvspan(20.5, xmax + 0.5, color="#ffebee", alpha=0.35, zorder=0)
    ax.axvline(20, color="#2e7d32", ls="--", lw=1.3, zorder=1)


def _mark_epoch(
    ax,
    epoch: np.ndarray,
    values: np.ndarray,
    e: int,
    *,
    color: str,
    text_color: str,
    label: str,
    text_offset: tuple[float, float],
) -> None:
    idx = np.where(epoch == e)[0]
    if not len(idx):
        return
    i = int(idx[0])
    val = float(values[i])
    ax.scatter([e], [val], color=color, s=42, zorder=6, edgecolors="white", linewidths=0.6)
    ax.annotate(
        label,
        xy=(e, val),
        xytext=(e + text_offset[0], val + text_offset[1]),
        fontsize=8.5,
        color=text_color,
        fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
    )


def _mark_stop_epoch(ax_ber, ax_ll, r0: dict) -> None:
    idx = np.where(r0["epoch"] == STOP_EPOCH)[0]
    if not len(idx):
        return
    i = int(idx[0])
    ber = float(r0["ber"][i])
    ll = float(r0["load_l1"][i])
    _mark_epoch(
        ax_ber, r0["epoch"], r0["ber"], STOP_EPOCH,
        color=GREEN, text_color=GREEN_TEXT,
        label=f"ep{STOP_EPOCH}\n{ber:.2f}%",
        text_offset=(2.5, 4.8),
    )
    _mark_epoch(
        ax_ll, r0["epoch"], r0["load_l1"], STOP_EPOCH,
        color=GREEN, text_color=GREEN_TEXT,
        label=f"ep{STOP_EPOCH}\n{ll:.2f}",
        text_offset=(2.5, 0.06),
    )


def _mark_spikes(ax, epoch: np.ndarray, ber: np.ndarray) -> None:
    for e in SPIKE_EPOCHS:
        _mark_epoch(
            ax, epoch, ber, e,
            color=RED, text_color=RED_TEXT,
            label=f"ep{e}\n{float(ber[np.where(epoch == e)[0][0]]):.1f}%",
            text_offset=(3.0, 0.8),
        )


def main() -> None:
    if not R0_VAL.exists() or not CONT_VAL.exists():
        raise SystemExit("Missing R0 or continuation validation.csv")

    r0 = load_series(R0_VAL)
    cont = load_series(CONT_VAL)
    xmax = int(cont["epoch"].max())
    ber_max = float(max(cont["ber"].max(), r0["ber"].max()))

    fig = plt.figure(figsize=(10.5, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.35], hspace=0.22)
    ax_ber = fig.add_subplot(gs[0, 0])
    ax_ll = fig.add_subplot(gs[1, 0], sharex=ax_ber)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.11)

    _shade_regions(ax_ber, xmax)
    ax_ber.plot(cont["epoch"], cont["ber"], color="#e65100", lw=2.2, label="R0 continued (epochs 1–60)", zorder=4)
    ax_ber.plot(r0["epoch"], r0["ber"], color="#111111", lw=2.0, ls="--", label="R0 primary (epochs 1–20)", zorder=5)
    ax_ber.set_ylim(0, ber_max * 1.14)
    ax_ber.set_ylabel("Validation BER (%)", fontsize=11, fontweight="bold")
    ax_ber.set_title("Clean-channel BER", fontsize=12, fontweight="bold", loc="left", pad=6)
    ax_ber.grid(axis="y", alpha=0.3)
    ax_ber.spines["top"].set_visible(False)
    ax_ber.spines["right"].set_visible(False)
    ax_ber.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    _shade_regions(ax_ll, xmax)
    ax_ll.plot(cont["epoch"], cont["load_l1"], color="#e65100", lw=2.2, zorder=4)
    ax_ll.plot(r0["epoch"], r0["load_l1"], color="#111111", lw=2.0, ls="--", zorder=5)
    ax_ll.axhline(0.20, color="#888888", ls=":", lw=1.0, label="Healthy threshold (0.20)")
    ax_ll.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_ll.set_ylabel("train_val_load_l1", fontsize=11, fontweight="bold")
    ax_ll.set_title("Routing generalisation gap", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax_ll.set_xlim(0.5, xmax + 0.5)
    ll_max = float(cont["load_l1"].max())
    ax_ll.set_ylim(0, max(0.80, ll_max * 1.18))
    ax_ll.grid(axis="y", alpha=0.3)
    ax_ll.spines["top"].set_visible(False)
    ax_ll.spines["right"].set_visible(False)
    ax_ll.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    ax_ll.tick_params(axis="x", labelsize=10)
    ax_ll.tick_params(axis="y", labelsize=10)

    _mark_stop_epoch(ax_ber, ax_ll, r0)
    _mark_spikes(ax_ber, cont["epoch"], cont["ber"])

    fig.suptitle(
        "R0 continuation: no stable improvement past epoch 20",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.02,
        "Green shading: convergence zone (epochs 1–20).  Pink shading: continuation (epochs 21–60).",
        ha="center",
        fontsize=8,
        color="#444444",
    )

    OUT_DURATION.mkdir(parents=True, exist_ok=True)
    OUT_NEW.mkdir(parents=True, exist_ok=True)
    for dest, name in ((OUT_DURATION, "training_duration_rationale"), (OUT_NEW, "training_duration_rationale")):
        for ext in ("png", "pdf"):
            fig.savefig(dest / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {dest / name}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
