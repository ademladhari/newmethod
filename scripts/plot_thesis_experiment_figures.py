#!/usr/bin/env python3
"""
Thesis figures: one grayscale figure per experiment (validation curves + expert loads).

Also renders grayscale summary plots and attack comparisons.

Usage (repo root):
  py -3 scripts/build_thesis_grouped_tables.py
  py -3 scripts/plot_thesis_grouped_tables.py
  py -3 scripts/plot_thesis_experiment_figures.py

Outputs:
  thesis/figures/experiments/{nn}_{slug}.png|.pdf
  thesis/figures/summary/*.png
  thesis/figures/attacks/*.png
  thesis/figures/FIGURES_INDEX.md
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
THESIS_CSV = RESULTS / "THESIS_ALL_EXPERIMENTS.csv"
EXP_ROOT = RESULTS / "experiments"
OUT_EXP = ROOT / "thesis" / "figures" / "experiments"
OUT_SUM = ROOT / "thesis" / "figures" / "summary"
OUT_ATK = ROOT / "thesis" / "figures" / "attacks"
ATTACK_DIR = RESULTS / "comparison_hidden_vs_moe"

SEARCH_ROOTS = [
    EXP_ROOT,
    RESULTS / "exp4continue_extracted",
    RESULTS / "attack_eval_runs",
    RESULTS / "extracted_bal08_ep30",
    RESULTS / "extracted_collapse_diag_new",
]

SKIP_PATH_PARTS = (
    "archive/duplicates",
    "archive/loose",
)

# Grayscale thesis style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "grid.color": "#bbbbbb",
        "grid.linewidth": 0.5,
        "text.color": "black",
    }
)

HATCHES = ["", "///", "...", "xxx", "\\\\", "+++", "ooo", "|||"]


def _col(df: pd.DataFrame, name: str) -> str | None:
    for c in df.columns:
        if c.strip() == name:
            return c
    return None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def expert_load_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if re.match(r"expert_load_\d+\s*$", c.strip()))


def slugify(text: str, max_len: int = 72) -> str:
    s = re.sub(r"[^\w\-]+", "_", text.lower()).strip("_")
    return s[:max_len] or "run"


def unique_slug(run: dict, meta: dict) -> str:
    """Unique filename from install path (disambiguates duplicate experiment_name)."""
    parts = run["rel"].replace("\\", "/").strip("/").split("/")
    leaf = run["run_dir"].name
    if len(parts) >= 2:
        tail = f"{parts[-2]}_{leaf}"
    else:
        tail = leaf
    return slugify(tail)


def infer_experiment_name(run_dir: Path) -> str:
    logs = list(run_dir.glob("*.log"))
    if logs:
        m = re.search(r"'experiment_name': '([^']+)'", logs[0].read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1)
    parent = run_dir.name.split(" 20")[0].strip()
    return parent


def infer_dataset(rel: str) -> str:
    r = rel.replace("\\", "/").lower()
    if "coco20k" in r or "/20k/" in r:
        return "coco20k"
    if "200k" in r or "240k" in r:
        return "coco~240k"
    if "coco100k" in r or "100k" in r:
        return "coco100k"
    return "unknown"


def load_thesis_meta() -> pd.DataFrame:
    if not THESIS_CSV.is_file():
        return pd.DataFrame()
    df = pd.read_csv(THESIS_CSV)
    return df[df["row_type"].isin(("metrics_on_disk", "in_progress"))].copy()


def match_meta(meta: pd.DataFrame, run_dir: Path, exp_name: str) -> dict:
    if meta.empty:
        return {}
    rel = str(run_dir).replace("\\", "/")
    best: dict | None = None
    best_score = -1
    for _, row in meta.iterrows():
        loc = str(row.get("location", "") or "").replace("\\", "/")
        folder = str(row.get("run_folder", "") or "")
        name = str(row.get("experiment_name", "") or "")
        score = 0
        if folder and folder in rel:
            score += 10
        if loc and loc in rel:
            score += 8
        if name == exp_name:
            score += 2
        if name and name in rel:
            score += 1
        if score > best_score:
            best_score = score
            best = row.to_dict()
    return best or {}


def should_skip(path: Path) -> bool:
    s = str(path).replace("\\", "/")
    return any(p in s for p in SKIP_PATH_PARTS)


def discover_runs() -> list[dict]:
    seen: set[str] = set()
    runs: list[dict] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for val_path in sorted(root.rglob("validation.csv")):
            if should_skip(val_path):
                continue
            run_dir = val_path.parent
            exp_name = infer_experiment_name(run_dir)
            rel = str(run_dir.relative_to(ROOT)).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            val = load_csv(val_path)
            if val is None or val.empty:
                continue
            runs.append(
                {
                    "run_dir": run_dir,
                    "rel": rel,
                    "experiment_name": exp_name,
                    "dataset": infer_dataset(rel),
                    "epochs": int(val[_col(val, "epoch")].max()) if _col(val, "epoch") else len(val),
                }
            )
    return runs


def epoch_series(run_dir: Path, csv_name: str) -> pd.DataFrame | None:
    df = load_csv(run_dir / csv_name)
    if df is None or df.empty:
        return None
    ep_c = _col(df, "epoch")
    if not ep_c:
        return None
    out = pd.DataFrame({"epoch": df[ep_c].astype(int)})
    for key in ["bitwise-error", "expert_max_use", "train_val_load_l1", "effective_experts"]:
        c = _col(df, key)
        if c:
            out[key.replace("-", "_")] = pd.to_numeric(df[c], errors="coerce")
    for lc in expert_load_cols(df):
        out[lc.strip()] = pd.to_numeric(df[lc], errors="coerce")
    return out


def config_subtitle(meta: dict, run: dict) -> str:
    parts = []
    if meta:
        for k, label in [
            ("dataset", "data"),
            ("num_experts", "E"),
            ("top_k", "k"),
            ("batch_size", "batch"),
            ("backbone_frozen", "frozen"),
            ("outcome", ""),
            ("thesis_use", "role"),
        ]:
            v = meta.get(k)
            if v not in (None, "", "nan", np.nan):
                if k == "backbone_frozen":
                    v = "yes" if str(v).lower() in ("true", "1") else "no"
                parts.append(f"{label}={v}" if label else str(v))
    else:
        parts.append(run["dataset"])
    return " · ".join(parts[:8])


def plot_experiment_figure(run: dict, meta: dict, idx: int, slug: str) -> None:
    run_dir: Path = run["run_dir"]
    val = epoch_series(run_dir, "validation.csv")
    valn = epoch_series(run_dir, "validation_noisy.csv")
    if val is None:
        return

    exp_name = run["experiment_name"]
    ber = val["bitwise_error"] * 100
    ep = val["epoch"]
    best_i = int(ber.idxmin())
    best_ep = int(ep.iloc[best_i])
    best_ber = float(ber.iloc[best_i])
    final_ep = int(ep.iloc[-1])

    report_ep = 20
    rep_raw = meta.get("report_epoch")
    if rep_raw is not None and str(rep_raw) not in ("", "nan"):
        try:
            report_ep = int(float(rep_raw))
        except (ValueError, TypeError):
            report_ep = 20
    if final_ep < report_ep:
        report_ep = final_ep

    fig = plt.figure(figsize=(6.8, 7.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.1], hspace=0.38, wspace=0.32)

    def style_ax(ax, title: str, ylabel: str):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("black")

    # (a) clean BER
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(ep, ber, color="black", linewidth=1.5, label="Clean val")
    ax0.axvline(report_ep, color="0.45", linestyle="--", linewidth=1, label=f"Report ep{report_ep}")
    if best_ep != report_ep:
        ax0.axvline(best_ep, color="0.65", linestyle=":", linewidth=1, label=f"Best ep{best_ep}")
    style_ax(ax0, "(a) Validation BER", "BER (%)")
    ax0.legend(loc="upper right", frameon=True, edgecolor="black")

    # (b) noisy BER
    ax1 = fig.add_subplot(gs[0, 1])
    if valn is not None and "bitwise_error" in valn.columns:
        ax1.plot(valn["epoch"], valn["bitwise_error"] * 100, color="0.35", linewidth=1.5, linestyle="--", label="Noisy val")
        ax1.axvline(report_ep, color="0.45", linestyle="--", linewidth=1)
        style_ax(ax1, "(b) Noisy validation BER", "BER (%)")
        ax1.legend(loc="upper right", frameon=True, edgecolor="black")
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No noisy validation log", ha="center", va="center", fontsize=10, transform=ax1.transAxes)

    # (c) max_use
    ax2 = fig.add_subplot(gs[1, 0])
    if "expert_max_use" in val.columns:
        n_exp = int(meta.get("num_experts") or 4)
        ax2.plot(ep, val["expert_max_use"], color="black", linewidth=1.5)
        ax2.axhline(1.0 / n_exp, color="0.55", linestyle=":", linewidth=1, label=f"Uniform (1/{n_exp})")
        ax2.axhline(0.5, color="0.75", linestyle="--", linewidth=1, label="50% on one expert")
        ax2.axvline(report_ep, color="0.45", linestyle="--", linewidth=1)
        style_ax(ax2, "(c) Expert max use", "expert_max_use")
        ax2.legend(loc="upper right", fontsize=6, frameon=True, edgecolor="black")

    # (d) load_l1
    ax3 = fig.add_subplot(gs[1, 1])
    if "train_val_load_l1" in val.columns:
        ax3.plot(ep, val["train_val_load_l1"], color="0.35", linewidth=1.5)
        ax3.axhline(0.20, color="0.55", linestyle="--", linewidth=1, label="Healthy (<0.20)")
        ax3.axvline(report_ep, color="0.45", linestyle="--", linewidth=1)
        style_ax(ax3, "(d) Train–val load L1", "L1 distance")
        ax3.legend(loc="upper right", fontsize=6, frameon=True, edgecolor="black")

    # (e) expert loads at report epoch
    ax4 = fig.add_subplot(gs[2, :])
    load_cols = [c for c in val.columns if c.startswith("expert_load_")]
    row = val[val["epoch"] == report_ep]
    if row.empty:
        row = val.iloc[[-1]]
        report_ep = int(row["epoch"].iloc[0])
    if load_cols:
        loads = [float(row[c].iloc[0]) for c in load_cols]
        x = np.arange(len(loads))
        bars = ax4.bar(
            x,
            loads,
            color="white",
            edgecolor="black",
            linewidth=1,
            hatch=[HATCHES[i % len(HATCHES)] for i in range(len(loads))],
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"Expert {i}" for i in range(len(loads))])
        ax4.set_ylim(0, max(0.5, max(loads) * 1.15))
        ax4.axhline(1.0 / len(loads), color="0.45", linestyle=":", linewidth=1)
        for b, v in zip(bars, loads):
            ax4.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    style_ax(ax4, f"(e) Validation expert loads at epoch {report_ep}", "Load share")
    ax4.set_xlabel("Expert index")

    outcome = str(meta.get("outcome", "") or "")
    role = str(meta.get("thesis_use", "") or "")
    title_line = f"{exp_name}"
    if outcome:
        title_line += f"  [{outcome}]"
    fig.suptitle(title_line, fontsize=10, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, config_subtitle(meta, run), ha="center", fontsize=8, style="italic")
    fig.text(
        0.5,
        0.02,
        f"Best: {best_ber:.2f}% @ ep{best_ep}  ·  Final: {float(ber.iloc[-1]):.2f}% @ ep{final_ep}",
        ha="center",
        fontsize=8,
    )

    OUT_EXP.mkdir(parents=True, exist_ok=True)
    stem = f"{idx:02d}_{slug}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT_EXP / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sort_runs(runs: list[dict], meta_df: pd.DataFrame) -> list[dict]:
    outcome_rank = {"SUCCESS": 0, "IN PROGRESS": 1, "FAILED": 2}
    ds_rank = {"coco100k": 0, "coco~240k": 1, "coco20k": 2, "unknown": 3}

    def key(r):
        m = match_meta(meta_df, r["run_dir"], r["experiment_name"])
        oc = str(m.get("outcome", "FAILED"))
        return (outcome_rank.get(oc, 3), ds_rank.get(r["dataset"], 9), r["experiment_name"])

    return sorted(runs, key=key)


def plot_attack_figures() -> list[str]:
    OUT_ATK.mkdir(parents=True, exist_ok=True)
    made = []
    for csv_path in sorted(ATTACK_DIR.glob("attack_summary*.csv")):
        df = pd.read_csv(csv_path)
        if "moe_bit_acc" not in df.columns or "hidden_bit_acc" not in df.columns:
            continue
        attacks = df["attack"].tolist()
        x = np.arange(len(attacks))
        w = 0.36
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(
            x - w / 2,
            df["hidden_bit_acc"] * 100,
            width=w,
            label="HiDDeN-177",
            color="white",
            edgecolor="black",
            hatch="///",
            linewidth=0.9,
        )
        ax.bar(
            x + w / 2,
            df["moe_bit_acc"] * 100,
            width=w,
            label="MoE",
            color="0.75",
            edgecolor="black",
            linewidth=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=35, ha="right")
        ax.set_ylabel("Bit accuracy (%)")
        ax.set_ylim(0, 105)
        ax.axhline(50, color="0.5", linestyle=":", linewidth=1)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        slug = csv_path.stem.replace("attack_summary_", "atk_")
        title = csv_path.stem.replace("attack_summary_", "").replace("_", " ")
        ax.set_title(f"Attack robustness: {title}", fontweight="bold")
        ax.legend(frameon=True, edgecolor="black", loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ATK / f"{slug}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        made.append(slug)
    return made


def plot_summary_frozen_vs_unfrozen(meta_df: pd.DataFrame, runs: list[dict]) -> None:
    """Matched 4e k=1 coco100k: BER and max_use at report epoch."""
    rows = []
    for r in runs:
        if r["dataset"] != "coco100k":
            continue
        m = match_meta(meta_df, r["run_dir"], r["experiment_name"])
        if not m:
            continue
        if int(float(m.get("num_experts") or 0)) != 4 or int(float(m.get("top_k") or 0)) != 1:
            continue
        rep_raw = m.get("report_epoch")
        rep = 20
        if rep_raw is not None and str(rep_raw) not in ("", "nan"):
            try:
                rep = int(float(rep_raw))
            except (ValueError, TypeError):
                rep = 20
        val = epoch_series(r["run_dir"], "validation.csv")
        if val is None:
            continue
        row = val[val["epoch"] == rep]
        if row.empty:
            continue
        frozen = str(m.get("backbone_frozen", "")).lower() in ("true", "1")
        rows.append(
            {
                "name": r["experiment_name"][:28],
                "frozen": "Frozen" if frozen else "Unfrozen",
                "ber": float(row["bitwise_error"].iloc[0]) * 100,
                "max_use": float(row["expert_max_use"].iloc[0]),
            }
        )
    if len(rows) < 2:
        return
    d = pd.DataFrame(rows).drop_duplicates(subset=["name"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, col, ylab in zip(axes, ["ber", "max_use"], ["BER @ ep20 (%)", "expert_max_use"]):
        for i, fr in enumerate(["Unfrozen", "Frozen"]):
            sub = d[d["frozen"] == fr]
            ax.bar(
                np.arange(len(sub)) + i * 0.25,
                sub[col],
                width=0.22,
                label=fr,
                color="white" if fr == "Unfrozen" else "0.7",
                edgecolor="black",
                hatch="" if fr == "Unfrozen" else "///",
            )
        ax.set_ylabel(ylab)
        ax.set_title(ylab, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(frameon=True, edgecolor="black")
    fig.suptitle("COCO-100k · 4 experts · top-1 (selected runs)", fontweight="bold")
    fig.tight_layout()
    OUT_SUM.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_SUM / f"summary_frozen_vs_unfrozen.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_index(exp_entries: list[tuple], attack_slugs: list[str]) -> None:
    lines = [
        "# Thesis figures index",
        "",
        "Grayscale, print-ready. Regenerate:",
        "",
        "```bash",
        "py -3 scripts/build_thesis_grouped_tables.py",
        "py -3 scripts/plot_thesis_grouped_tables.py",
        "py -3 scripts/plot_thesis_experiment_figures.py",
        "```",
        "",
        "## Per-experiment (`figures/experiments/`)",
        "",
        "Each file: validation BER, noisy BER (if logged), routing stability, expert loads.",
        "",
    ]
    for idx, slug, name, outcome in exp_entries:
        lines.append(f"- `{idx:02d}_{slug}.pdf` — **{name}** ({outcome})")
    lines.extend(["", "## Grouped tables (`figures/grouped/`)", "", "See `figures/grouped/README.md`.", ""])
    lines.extend(["## Attack eval (`figures/attacks/`)", ""])
    for s in attack_slugs:
        lines.append(f"- `{s}.pdf` / `.png`")
    lines.extend(["", "## Summary (`figures/summary/`)", "", "- `summary_frozen_vs_unfrozen.pdf`", ""])
    path = ROOT / "thesis" / "figures" / "FIGURES_INDEX.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for out_dir in (OUT_EXP, OUT_ATK):
        if out_dir.is_dir():
            for f in out_dir.iterdir():
                if f.is_file():
                    f.unlink()

    meta_df = load_thesis_meta()
    runs = sort_runs(discover_runs(), meta_df)
    if not runs:
        raise SystemExit("No validation.csv runs found under results/")

    exp_entries: list[tuple] = []
    for idx, run in enumerate(runs, start=1):
        meta = match_meta(meta_df, run["run_dir"], run["experiment_name"])
        slug = unique_slug(run, meta)
        plot_experiment_figure(run, meta, idx, slug)
        exp_entries.append(
            (
                idx,
                slug,
                run["experiment_name"],
                str(meta.get("outcome", "—")),
            )
        )

    attack_slugs = plot_attack_figures()
    plot_summary_frozen_vs_unfrozen(meta_df, runs)
    write_index(exp_entries, attack_slugs)

    print(f"Per-experiment figures: {len(exp_entries)} → {OUT_EXP}")
    print(f"Attack figures: {len(attack_slugs)} → {OUT_ATK}")
    print(f"Index: thesis/figures/FIGURES_INDEX.md")


if __name__ == "__main__":
    main()
