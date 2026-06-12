"""
Routing collapse score vs clean-channel BER (COCO-100k only) — multiple style variants.

Usage (repo root):
  py -3 scripts/plot_collapse_vs_ber.py

Outputs:
  thesis/figures/new_figures/collapse_vs_ber_all_runs.png   (default: backbone + N marker)
  thesis/figures/rq/collapse_vs_ber_all_runs.png
  thesis/figures/new_figures/collapse_vs_ber_variants/*.png
  thesis/figures/rq/collapse_vs_ber_variants/*.png
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_RQ = ROOT / "thesis" / "figures" / "rq"
OUT_NEW = ROOT / "thesis" / "figures" / "new_figures"
VARIANT_SUBDIR = "collapse_vs_ber_variants"

UNFROZEN_C = "#2ca02c"
FROZEN_C = "#d62728"
N4_C = "#1f77b4"
N8_C = "#9467bd"
N_OTHER_C = "#7f7f7f"

N_MARKERS = {4: "o", 8: "s"}
N_MARKER_LABELS = {4: "N = 4 experts", 8: "N = 8 experts"}


def load_runs() -> list[dict]:
    exp_root = ROOT / "results" / "experiments"
    rows: list[dict] = []
    for val_csv in sorted(exp_root.rglob("validation.csv")):
        parts = str(val_csv.parent).replace("\\", "/").split("results/experiments/")[-1]
        backbone = "unfrozen" if "unfrozen_moe" in parts else "frozen"
        n_match = re.search(r"(\d+)exp", parts)
        b_match = re.search(r"batch(\d+)", parts)
        k_match = re.search(r"_k(\d+)", parts)
        n_experts = int(n_match.group(1)) if n_match else 4
        batch = int(b_match.group(1)) if b_match else 32
        top_k = int(k_match.group(1)) if k_match else 1
        dataset = "coco20k" if "coco20k" in parts else "coco100k"
        run_name = parts.split("/")[-1]
        try:
            with open(val_csv, encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            if not data:
                continue
            last = data[-1]
            ber = float(last.get("bitwise-error", "nan")) * 100
            maxuse = float(last.get("expert_max_use", last.get("val_expert_max_use", "nan")))
            ep = int(last.get("epoch", 0))
            if np.isnan(ber) or np.isnan(maxuse):
                continue
            rows.append(
                dict(
                    B=batch,
                    bb=backbone,
                    N=n_experts,
                    k=top_k,
                    ep=ep,
                    ber=ber,
                    maxuse=maxuse,
                    collapse=maxuse * n_experts,
                    dataset=dataset,
                    run=run_name,
                )
            )
        except Exception:
            pass
    return rows


def coco100k_runs(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["dataset"] == "coco100k"]


def find_r0(runs: list[dict]) -> dict:
    for r in runs:
        if r["bb"] == "unfrozen" and r["N"] == 4 and r["k"] == 1 and round(r["ber"], 2) == 0.32:
            return r
    return runs[0]


def backbone_color(r: dict) -> str:
    return UNFROZEN_C if r["bb"] == "unfrozen" else FROZEN_C


def n_marker(r: dict) -> str:
    return N_MARKERS.get(r["N"], "D")


def n_color(r: dict) -> str:
    if r["N"] == 4:
        return N4_C
    if r["N"] == 8:
        return N8_C
    return N_OTHER_C


def batch_size(r: dict) -> float:
    return max(40, min(r["B"] * 1.2, 200))


def pearson(runs: list[dict]) -> tuple[float, float]:
    xs = np.array([r["collapse"] for r in runs])
    ys = np.array([r["ber"] for r in runs])
    return stats.pearsonr(xs, ys)


def _style_axes(ax, runs: list[dict], r_val: float, p_val: float, title_extra: str = "") -> None:
    xs = np.array([r["collapse"] for r in runs])
    ys = np.array([r["ber"] for r in runs])
    m_fit, b_fit, *_ = stats.linregress(xs, ys)
    x_line = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 300)
    ax.plot(
        x_line, m_fit * x_line + b_fit, "--", color="#555", lw=1.8,
        label=f"OLS ($r={r_val:.2f}$, $p={p_val:.3f}$)", zorder=2,
    )
    ax.axvspan(0.9, 1.5, alpha=0.08, color="#2ca02c")
    ax.axvspan(1.5, 2.8, alpha=0.07, color="orange")
    ax.axvspan(3.5, xs.max() + 0.4, alpha=0.08, color="#d62728")
    ax.axvline(1.0, color="#2e7d32", ls=":", lw=0.9, alpha=0.6)
    ax.set_xlabel("Collapse score  (max_use × N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clean-channel BER (%)", fontsize=11, fontweight="bold")
    title = f"Routing collapse vs clean BER — COCO-100k ($n={len(runs)}$)"
    if title_extra:
        title += f"\n{title_extra}"
    else:
        title += f"\nPearson $r={r_val:.2f}$, $p={p_val:.3f}$"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.28)
    ax.set_xlim(0.7, xs.max() + 0.3)
    ax.set_ylim(-0.5, max(ys) * 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_r0_worst(ax, runs: list[dict], r0: dict) -> None:
    ax.annotate(
        "R0 (N=4)",
        xy=(r0["collapse"], r0["ber"]),
        xytext=(r0["collapse"] + 0.4, r0["ber"] + max(1.5, runs[-1]["ber"] * 0.08)),
        fontsize=8.5, fontweight="bold", color=UNFROZEN_C,
        arrowprops=dict(arrowstyle="->", color=UNFROZEN_C, lw=1.2),
    )
    worst = max(runs, key=lambda r: r["ber"])
    ax.annotate(
        f"worst: {worst['ber']:.0f}% BER\nN={worst['N']}, {worst['bb']}",
        xy=(worst["collapse"], worst["ber"]),
        xytext=(worst["collapse"] - 1.4, worst["ber"] - max(3, worst["ber"] * 0.12)),
        fontsize=7.5, color=FROZEN_C if worst["bb"] == "frozen" else N8_C,
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.9),
    )


def plot_v1_backbone_color_n_marker(ax, runs: list[dict]) -> None:
    """Colour = backbone; marker shape = N (circle/square)."""
    for r in runs:
        ax.scatter(
            r["collapse"], r["ber"],
            c=backbone_color(r), marker=n_marker(r), s=batch_size(r),
            alpha=0.88, linewidths=0.7, edgecolors="k", zorder=3,
        )


def legend_v1(ax) -> None:
    handles = ax.get_legend_handles_labels()[0]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=UNFROZEN_C, markersize=9, label="Unfrozen"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FROZEN_C, markersize=9, label="Frozen"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=11, markeredgecolor="k", label="N = 4 (circle)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markersize=9, markeredgecolor="k", label="N = 8 (square)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=11, label="Larger = batch 128"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")


def plot_v2_color_by_n(ax, runs: list[dict]) -> None:
    """Fill colour = N; unfrozen filled, frozen hollow (open marker)."""
    for r in runs:
        nc = n_color(r)
        mk = n_marker(r)
        sz = batch_size(r)
        if r["bb"] == "unfrozen":
            ax.scatter(
                r["collapse"], r["ber"],
                c=nc, marker=mk, s=sz,
                alpha=0.88, linewidths=0.7, edgecolors="k", zorder=3,
            )
        else:
            ax.scatter(
                r["collapse"], r["ber"],
                facecolors="none", edgecolors=nc, marker=mk, s=sz * 1.08,
                linewidths=2.0, zorder=3,
            )


def legend_v2(ax) -> None:
    handles = ax.get_legend_handles_labels()[0]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=N4_C, markersize=9, label="N = 4 (filled = unfrozen)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=N8_C, markersize=9, label="N = 8"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=N4_C,
               markersize=9, markeredgewidth=2, label="Frozen (hollow)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")


def plot_v3_facets(runs: list[dict], r_val: float, p_val: float, r0: dict) -> plt.Figure:
    """Side-by-side panels: N=4 only | N=8 only."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    for ax, n_exp, subtitle in zip(
        axes,
        (4, 8),
        ("N = 4 experts", "N = 8 experts"),
    ):
        sub = [r for r in runs if r["N"] == n_exp]
        if len(sub) < 2:
            ax.text(0.5, 0.5, f"No runs with N={n_exp}", ha="center", va="center", transform=ax.transAxes)
            continue
        xs = np.array([r["collapse"] for r in sub])
        ys = np.array([r["ber"] for r in sub])
        r_sub, p_sub = stats.pearsonr(xs, ys)
        for r in sub:
            ax.scatter(
                r["collapse"], r["ber"],
                c=backbone_color(r), marker="o", s=batch_size(r),
                alpha=0.88, linewidths=0.7, edgecolors="k", zorder=3,
            )
        m_fit, b_fit, *_ = stats.linregress(xs, ys)
        x_line = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 300)
        ax.plot(x_line, m_fit * x_line + b_fit, "--", color="#555", lw=1.6, zorder=2)
        ax.axvline(n_exp * 0.25, color="#2e7d32", ls=":", lw=0.9, alpha=0.5,
                   label=f"Uniform ideal ({n_exp}×0.25)")
        ax.set_title(f"{subtitle}  ($n={len(sub)}$, $r={r_sub:.2f}$)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Collapse score (max_use × N)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.28)
        ax.set_xlim(0.7, max(xs.max(), n_exp * 0.5) + 0.5)
        if n_exp == 4 and any(r["run"] == r0["run"] for r in sub):
            ax.scatter([r0["collapse"]], [r0["ber"]], s=120, facecolors="none",
                       edgecolors=UNFROZEN_C, linewidths=2.2, zorder=5, label="R0")
    axes[0].set_ylabel("Clean-channel BER (%)", fontsize=11, fontweight="bold")
    fig.suptitle(
        f"Collapse vs BER split by expert count  (full sample $r={r_val:.2f}$, $p={p_val:.3f}$)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=UNFROZEN_C, markersize=8, label="Unfrozen"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FROZEN_C, markersize=8, label="Frozen"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


def plot_v4_n_text_labels(ax, runs: list[dict]) -> None:
    """Backbone colour + marker N + small N=k label beside each point."""
    for r in runs:
        ax.scatter(
            r["collapse"], r["ber"],
            c=backbone_color(r), marker=n_marker(r), s=batch_size(r),
            alpha=0.88, linewidths=0.7, edgecolors="k", zorder=3,
        )
        ax.annotate(
            f"N{r['N']}",
            xy=(r["collapse"], r["ber"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6.5, fontweight="bold", color="#333333", zorder=5,
        )


def plot_v5_hollow_n(ax, runs: list[dict]) -> None:
    """Hollow markers: edge colour = N, fill = backbone (lighter)."""
    for r in runs:
        fill = backbone_color(r)
        edge = n_color(r)
        ax.scatter(
            r["collapse"], r["ber"],
            c=fill, marker=n_marker(r), s=batch_size(r) * 1.1,
            alpha=0.55, linewidths=2.2, edgecolors=edge, zorder=3,
        )


def legend_v5(ax) -> None:
    handles = ax.get_legend_handles_labels()[0]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=UNFROZEN_C, markeredgecolor=N4_C,
               markersize=10, markeredgewidth=2, label="N=4 unfrozen"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=FROZEN_C, markeredgecolor=N8_C,
               markersize=10, markeredgewidth=2, label="N=8 frozen"),
        Line2D([0], [0], color=N4_C, lw=2, label="Edge colour → N (blue=4, purple=8)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")


def build_single_panel(
    runs: list[dict],
    r0: dict,
    plot_fn,
    legend_fn,
    title_extra: str,
) -> plt.Figure:
    r_val, p_val = pearson(runs)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    plot_fn(ax, runs)
    _style_axes(ax, runs, r_val, p_val, title_extra=title_extra)
    _annotate_r0_worst(ax, runs, r0)
    legend_fn(ax)
    fig.tight_layout()
    return fig


def save_fig(fig: plt.Figure, name: str, *, primary: bool = False) -> None:
    for base in (OUT_RQ, OUT_NEW):
        base.mkdir(parents=True, exist_ok=True)
        if primary:
            path = base / f"{name}.png"
        else:
            dest = base / VARIANT_SUBDIR
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


def write_variant_index() -> None:
    lines = [
        "# Collapse vs BER figure variants (COCO-100k, n=20)",
        "",
        "| File | Encoding | Best for |",
        "|------|----------|----------|",
        "| `collapse_vs_ber_all_runs.png` | **Default** — colour=backbone, marker=N | Thesis main figure |",
        "| `v1_backbone_marker_n.png` | Same as default | Clean legend |",
        "| `v2_color_by_n.png` | Colour=N, rim style=backbone | Emphasise expert count |",
        "| `v3_facets_n4_n8.png` | Two panels N=4 / N=8 | Compare scales separately |",
        "| `v4_n_text_labels.png` | Backbone + marker + `N4`/`N8` text | No legend needed |",
        "| `v5_hollow_n_edge.png` | Fill=backbone, thick edge=N colour | Print / colour-blind friendly |",
        "",
        "Regenerate: `py -3 scripts/plot_collapse_vs_ber.py`",
    ]
    for base in (OUT_NEW, OUT_RQ):
        idx = base / VARIANT_SUBDIR / "README.md"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_rows = load_runs()
    runs = coco100k_runs(all_rows)
    if not runs:
        raise SystemExit("No COCO-100k validation runs found")
    r0 = find_r0(runs)
    r_val, p_val = pearson(runs)

    variants = [
        ("v1_backbone_marker_n", plot_v1_backbone_color_n_marker, legend_v1,
         "Colour = backbone; marker = N (○4, □8); size = batch"),
        ("v2_color_by_n", plot_v2_color_by_n, legend_v2,
         "Colour = N; rim solid=unfrozen, dashed=frozen"),
        ("v4_n_text_labels", plot_v4_n_text_labels, legend_v1,
         "Backbone colour + N label on each point"),
        ("v5_hollow_n_edge", plot_v5_hollow_n, legend_v5,
         "Fill = backbone; edge colour = N"),
    ]

    for name, plot_fn, legend_fn, extra in variants:
        fig = build_single_panel(runs, r0, plot_fn, legend_fn, extra)
        save_fig(fig, name, primary=False)

    # Default = v1 copy
    fig_default = build_single_panel(
        runs, r0, plot_v1_backbone_color_n_marker, legend_v1,
        f"Pearson $r={r_val:.2f}$, $p={p_val:.3f}$  ·  ○ $N{{=}}4$, □ $N{{=}}8$",
    )
    save_fig(fig_default, "collapse_vs_ber_all_runs", primary=True)

    fig_facets = plot_v3_facets(runs, r_val, p_val, r0)
    save_fig(fig_facets, "v3_facets_n4_n8", primary=False)

    write_variant_index()
    print(f"\n{len(variants) + 2} figures written (+ README in {VARIANT_SUBDIR}/)")


if __name__ == "__main__":
    main()
