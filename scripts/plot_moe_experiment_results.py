#!/usr/bin/env python3
"""
Generate thesis-ready plots from all installed MoE experiment CSVs.

Usage (from repo root):
  python scripts/plot_moe_experiment_results.py

Outputs: results/plots/*.png and thesis/figures/*.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "results" / "experiments"
REGISTRY = ROOT / "results" / "FULL_EXPERIMENT_REGISTRY.json"
ATTACK_CSV = ROOT / "results" / "comparison_hidden_vs_moe" / "attack_summary.csv"
OUT_DIRS = [ROOT / "results" / "plots", ROOT / "thesis" / "figures"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
)


def _col(df: pd.DataFrame, name: str) -> str | None:
    name = name.strip()
    for c in df.columns:
        if c.strip() == name:
            return c
    return None


def _fval(row: pd.Series, df: pd.DataFrame, key: str, default=np.nan):
    c = _col(df, key)
    if c is None:
        return default
    try:
        return float(row[c])
    except (TypeError, ValueError):
        return default


def _infer_from_path(rel: str) -> dict:
    """Fill gaps when registry pickle/logs omit fields (common for legacy_early)."""
    r = rel.replace("\\", "/")
    out: dict = {}
    m = re.search(r"/batch(\d+)/", r)
    if m:
        out["batch_size"] = int(m.group(1))
    m = re.search(r"/(\d+)exp_k(\d+)/", r)
    if m:
        out["num_experts"] = int(m.group(1))
        out["top_k"] = int(m.group(2))
    return out


def infer_dataset(rel: str, cfg: dict) -> str:
    """Registry may miss dataset; path layout is experiments/.../coco{20,100}k/..."""
    ds = (cfg.get("dataset") or "unknown").strip()
    if ds in ("coco20k", "coco100k"):
        return ds
    r = rel.replace("\\", "/")
    if "/coco20k/" in r or r.startswith("coco20k/"):
        return "coco20k"
    if "/coco100k/" in r or r.startswith("coco100k/"):
        return "coco100k"
    return ds


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def expert_load_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if re.match(r"expert_load_\d+\s*$", c.strip()))


def load_all_runs() -> pd.DataFrame:
    rows = []
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    cfg_by_path = {e["path"]: e for e in registry.get("installed_experiments", [])}

    for train_path in sorted(EXP_ROOT.rglob("train.csv")):
        run_dir = train_path.parent
        rel = str(run_dir.relative_to(EXP_ROOT)).replace("\\", "/")
        meta = cfg_by_path.get(f"experiments/{rel}", {})
        cfg = meta.get("config", {})

        train = load_csv(train_path)
        val = load_csv(run_dir / "validation.csv")
        valn = load_csv(run_dir / "validation_noisy.csv")
        if train is None or train.empty:
            continue

        t_last = train.iloc[-1]
        v_last = val.iloc[-1] if val is not None and not val.empty else None
        vn_last = valn.iloc[-1] if valn is not None and not valn.empty else None

        def snap(row, df):
            if row is None:
                return {}
            return {
                "ber": _fval(row, df, "bitwise-error"),
                "expert_max_use": _fval(row, df, "expert_max_use"),
                "effective_experts": _fval(row, df, "effective_experts"),
                "train_val_load_l1": _fval(row, df, "train_val_load_l1"),
                "balance_loss": _fval(row, df, "balance_loss"),
                "encoder_mse": _fval(row, df, "encoder_mse"),
            }

        ts = snap(t_last, train)
        vs = snap(v_last, val) if v_last is not None else {}
        vns = snap(vn_last, valn) if vn_last is not None else {}

        short = cfg.get("experiment_name") or run_dir.name[:40]
        freeze = str(cfg.get("freeze_hidden_backbone", "")).lower() in ("true", "1")
        branch = rel.split("/")[0] if "/" in rel else "unknown"
        path_hints = _infer_from_path(rel)

        rows.append(
            {
                "path": rel,
                "branch": branch,
                "name": short,
                "dataset": infer_dataset(rel, cfg),
                "num_experts": int(float(cfg.get("num_experts") or path_hints.get("num_experts", 4) or 4)),
                "top_k": int(float(cfg.get("top_k") or path_hints.get("top_k", 1) or 1)),
                "batch_size": int(float(cfg.get("batch_size") or path_hints.get("batch_size", 0) or 0)),
                "balance_w": float(cfg.get("balance_loss_weight", np.nan) or np.nan),
                "jitter": float(cfg.get("router_jitter_noise", np.nan) or np.nan),
                "load_pen": float(cfg.get("load_penalty_weight", 0) or 0),
                "temp_end": float(cfg.get("router_temperature_end", np.nan) or np.nan),
                "frozen": freeze,
                "has_noisy_val": valn is not None and not valn.empty,
                "epochs": int(_fval(t_last, train, "epoch")),
                "train_ber": ts["ber"],
                "val_ber": vs.get("ber", np.nan),
                "val_noisy_ber": vns.get("ber", np.nan),
                "train_max_use": ts["expert_max_use"],
                "val_max_use": vs.get("expert_max_use", ts["expert_max_use"]),
                "val_load_l1": vs.get("train_val_load_l1", np.nan),
                "val_eff_exp": vs.get("effective_experts", np.nan),
                "train_eff_exp": ts["effective_experts"],
                "run_dir": str(run_dir),
            }
        )

    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_epoch_series(run_dir: Path, csv_name: str) -> pd.DataFrame | None:
    df = load_csv(run_dir / csv_name)
    if df is None or df.empty:
        return None
    ep = _col(df, "epoch")
    if ep is None:
        return None
    out = pd.DataFrame({"epoch": df[ep].astype(int)})
    for key in ["bitwise-error", "expert_max_use", "effective_experts", "train_val_load_l1"]:
        c = _col(df, key)
        if c:
            out[key.replace("-", "_")] = pd.to_numeric(df[c], errors="coerce")
    loads = expert_load_cols(df)
    for lc in loads:
        out[lc.strip()] = pd.to_numeric(df[lc], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def short_run_tag(name: str, frozen: bool) -> str:
    """Readable setup tag from experiment name."""
    n = name.lower()
    if "unfrozen" in n or "sym_t14_unfrozen" in n:
        return "unfrozen · sym_t14"
    if "routing_isolation" in n:
        return "frozen · routing isolation"
    if "collapse_diag" in n:
        return "collapse diagnostic"
    if "router_diag" in n:
        return "router diagnostic"
    if "uniform_first" in n:
        return "uniform router init"
    if "balance_v3" in n:
        return "high jitter (balance v3)"
    if "balance_v2" in n:
        return "balance v2 · top-2"
    if "sym_bal08" in n or "bal08" in n:
        return "frozen · strong balance (0.08)"
    if "sym_t14" in n and "t09" in n:
        return "frozen · sym_t14 · temp→0.9"
    if "sym_t14" in n:
        return "frozen · sym_t14"
    if "ber_route" in n:
        return "BER + routing loss"
    if "hidden178" in n or "anticollapse" in n:
        return "HiDDeN-178 frozen backbone"
    if "stabilized" in n:
        return "stabilized (legacy)"
    if "moe_hidden" in n:
        return "early MoE baseline"
    return "frozen MoE" if frozen else "unfrozen MoE"


def routing_landscape_label(row: pd.Series) -> str:
    tag = short_run_tag(str(row["name"]), bool(row["frozen"]))
    return (
        f"{tag}  ({int(row['num_experts'])} experts · top-{int(row['top_k'])} · "
        f"batch {int(row['batch_size'])})"
    )


def plot_collapse_comparison(df: pd.DataFrame):
    """Routing landscape: collapse score (max_use×N) + train–val routing gap."""
    sub = df[df["dataset"] == "coco100k"].copy()
    sub["collapse_score"] = sub["val_max_use"] * sub["num_experts"]
    sub["label"] = sub.apply(routing_landscape_label, axis=1)
    sub = sub.sort_values("collapse_score", ascending=False)
    colors = sub["frozen"].map({True: "#c44e52", False: "#55a868"})

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6.5, len(sub) * 0.42)))
    y = np.arange(len(sub))
    scores = sub["collapse_score"].values
    best_idx = int(np.nanargmin(scores))

    axes[0].barh(y, scores, color=colors, edgecolor="gray", linewidth=0.5, height=0.72)
    axes[0].barh(
        best_idx,
        scores[best_idx],
        color="#55a868",
        edgecolor="#2d6a4f",
        linewidth=2.0,
        height=0.72,
    )
    axes[0].axvline(1.0, color="#2d6a4f", ls="--", lw=1.5, label="balanced (score = 1)")
    axes[0].axvline(2.0, color="#e67e22", ls=":", lw=1.2, label="≥50% on one expert (4-exp)")
    axes[0].axvline(4.0, color="#c44e52", ls="--", lw=1.5, label="≥50% on one expert (8-exp)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["label"], fontsize=8)
    axes[0].set_xlabel("Collapse score = expert_max_use × N")
    axes[0].set_title(
        "Routing collapse (higher = worse)\n"
        "Corrects for expert count — 4-exp is not inherently worse than 8-exp."
    )
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=6.5)
    axes[0].grid(True, axis="x", alpha=0.25)

    l1 = sub["val_load_l1"]
    l1_plot = l1.fillna(sub["val_max_use"] * 0.5)
    axes[1].barh(y, l1_plot, color=colors, edgecolor="gray", linewidth=0.5, height=0.72)
    axes[1].axvline(0.20, color="#c44e52", ls="--", lw=1.5, label="healthy (< 0.20)")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(sub["label"], fontsize=8)
    axes[1].set_xlabel("L1 distance: train vs validation expert-load histograms")
    axes[1].set_title(
        "Train vs validation routing gap (higher = worse)\n"
        "Val images routed differently than train → unstable watermarking."
    )
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=6.5)
    axes[1].grid(True, axis="x", alpha=0.25)
    if l1.isna().any():
        axes[1].text(
            0.98,
            0.02,
            "† L1 missing in some legacy CSVs (bar approximated)",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            style="italic",
        )

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor="#55a868", label="Unfrozen HiDDeN + MoE"),
            Patch(facecolor="#c44e52", label="Frozen HiDDeN backbone"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(
        "Routing landscape — COCO-100k, final epoch (ranked by collapse score)",
        y=1.08,
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "01_collapse_comparison_coco100k.png")


def plot_frozen_vs_unfrozen(df: pd.DataFrame):
    sub = df[(df["dataset"] == "coco100k") & (df["num_experts"] == 4) & (df["top_k"] == 1)].copy()
    metrics = [
        ("val_ber", "Val BER (clean)", "lower"),
        ("val_noisy_ber", "Val BER (noisy)", "lower"),
        ("val_max_use", "expert_max_use", "lower"),
        ("val_load_l1", "train_val_load_l1", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (col, title, _) in zip(axes.flat, metrics):
        g = sub.groupby("frozen")[col].agg(["mean", "std", "count"])
        xs = [0, 1]
        labels = ["Unfrozen", "Frozen"]
        means = [g.loc[f, "mean"] if f in g.index else np.nan for f in [False, True]]
        stds = [g.loc[f, "std"] if f in g.index else 0 for f in [False, True]]
        ax.bar(xs, means, yerr=stds, color=["#55a868", "#c44e52"], capsize=5, edgecolor="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(
                0 if not row["frozen"] else 1,
                row[col],
                s=40,
                c="black",
                alpha=0.5,
                zorder=5,
            )
    fig.suptitle("Frozen vs unfrozen (4 experts, top-1, COCO-100k)", fontsize=12)
    fig.tight_layout()
    save(fig, "02_frozen_vs_unfrozen_4exp_k1.png")


def plot_batch_size_effects(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [
        ("val_ber", "Val BER (clean)"),
        ("val_noisy_ber", "Val BER (noisy)"),
        ("val_max_use", "expert_max_use"),
        ("val_load_l1", "train_val_load_l1"),
        ("val_eff_exp", "Effective experts"),
        ("train_ber", "Train BER"),
    ]
    batches = sorted(sub["batch_size"].unique())
    for ax, (col, title) in zip(axes.flat, metrics):
        valid = sub[sub[col].notna() & (sub[col] == sub[col])]
        if valid.empty:
            ax.set_visible(False)
            continue
        for frozen, marker, color in [(False, "o", "#55a868"), (True, "s", "#c44e52")]:
            m = valid[valid["frozen"] == frozen]
            if m.empty:
                continue
            ax.scatter(
                m["batch_size"],
                m[col],
                s=80,
                c=color,
                marker=marker,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.5,
                label="Frozen" if frozen else "Unfrozen",
            )
        ax.set_xlabel("Batch size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(batches)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle("Batch size vs routing / BER metrics (COCO-100k)", fontsize=12)
    fig.tight_layout()
    save(fig, "03_batch_size_scatter.png")

    # Boxplot by batch size (val_ber + max_use)
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5))
    for i, col in enumerate(["val_ber", "val_max_use"]):
        data = [sub[sub["batch_size"] == b][col].dropna().values for b in batches]
        bp = ax2[i].boxplot(data, tick_labels=[str(b) for b in batches], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4c72b0")
            patch.set_alpha(0.6)
        ax2[i].set_xlabel("Batch size")
        ax2[i].set_title(col.replace("_", " "))
        ax2[i].grid(True, alpha=0.3, axis="y")
    fig2.suptitle("Distribution by batch size (COCO-100k)", fontsize=12)
    fig2.tight_layout()
    save(fig2, "04_batch_size_boxplot.png")


def plot_num_experts_topk(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    sub["cfg"] = sub.apply(lambda r: f"{int(r['num_experts'])}e,k{int(r['top_k'])}", axis=1)
    cfg_order = sorted(sub["cfg"].unique())
    xmap = {c: i for i, c in enumerate(cfg_order)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, ylab in zip(
        axes,
        ["val_ber", "val_max_use", "val_load_l1"],
        ["Val BER", "expert_max_use", "train_val_load_l1"],
    ):
        for (ne, tk), grp in sub.groupby(["num_experts", "top_k"]):
            cfg = f"{int(ne)}e,k{int(tk)}"
            xs = [xmap[cfg]] * len(grp)
            jitter = (np.random.rand(len(grp)) - 0.5) * 0.25
            ax.scatter(
                np.array(xs) + jitter,
                grp[col],
                s=70,
                alpha=0.8,
                label=f"{ne} exp, k={tk}",
            )
        ax.set_ylabel(ylab)
        ax.set_xlabel("Configuration")
        ax.set_xticks(range(len(cfg_order)))
        ax.set_xticklabels(cfg_order, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=6, loc="best")
    fig.suptitle("Number of experts & top-k vs metrics (COCO-100k)", fontsize=12)
    fig.tight_layout()
    save(fig, "05_num_experts_topk.png")


def plot_hyperparameter_scatters(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    pairs = [
        ("balance_w", "val_ber", "Balance loss weight", "Val BER"),
        ("balance_w", "val_max_use", "Balance loss weight", "expert_max_use"),
        ("jitter", "val_max_use", "Router jitter", "expert_max_use"),
        ("jitter", "val_ber", "Router jitter", "Val BER"),
        ("load_pen", "val_max_use", "Load penalty", "expert_max_use"),
        ("temp_end", "val_ber", "Router temp end", "Val BER"),
    ]
    for ax, (x, y, xl, yl) in zip(axes.flat, pairs):
        v = sub[sub[x].notna() & sub[y].notna()]
        sc = ax.scatter(
            v[x],
            v[y],
            c=v["frozen"].map({True: 0, False: 1}),
            cmap="RdYlGn_r",
            s=80,
            edgecolors="k",
            linewidths=0.4,
        )
        for _, r in v.iterrows():
            ax.annotate(r["name"][:12], (r[x], r[y]), fontsize=5, alpha=0.7)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Hyperparameters vs BER / expert utilisation", fontsize=12)
    fig.tight_layout()
    save(fig, "06_hyperparameter_scatters.png")


def plot_ber_vs_max_use(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for frozen, color, label in [(False, "#55a868", "Unfrozen"), (True, "#c44e52", "Frozen")]:
        m = sub[sub["frozen"] == frozen]
        ax.scatter(
            m["val_max_use"],
            m["val_ber"] * 100,
            s=60 + m["num_experts"] * 15,
            c=color,
            alpha=0.8,
            label=label,
            edgecolors="k",
            linewidths=0.4,
        )
    ax.axvline(0.50, color="red", ls="--", alpha=0.7, label="collapse gate")
    ax.set_xlabel("Validation expert_max_use")
    ax.set_ylabel("Validation BER (%)")
    ax.set_title("BER vs expert collapse (COCO-100k, final epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "07_ber_vs_expert_max_use.png")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    v = sub[sub["val_noisy_ber"].notna()]
    ax2.scatter(v["val_max_use"], v["val_noisy_ber"] * 100, c=v["frozen"].map({True: "#c44e52", False: "#55a868"}), s=80, edgecolors="k")
    ax2.set_xlabel("expert_max_use")
    ax2.set_ylabel("Noisy val BER (%)")
    ax2.set_title("Noisy BER vs collapse")
    ax2.grid(True, alpha=0.3)
    save(fig2, "08_noisy_ber_vs_expert_max_use.png")


def plot_training_curves(df: pd.DataFrame):
    """Epoch curves for representative runs."""
    picks = [
        ("unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01", "Unfrozen sym_t14 (best)", "#55a868"),
        ("frozen_moe/coco100k/4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01", "Frozen bal08 b128", "#4c72b0"),
        ("frozen_moe/coco100k/4exp_k1/batch32/sym_t14_temp09_bal004warm10_ep20_2026-06-01", "Frozen t14 t0.9 (collapsed)", "#c44e52"),
        ("frozen_moe/coco100k/8exp_k2/batch32/collapse_diag_jitter02_loadpen01_bal06_ep34_2026-05-31", "8exp collapse diag", "#dd8452"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (rel, label, color) in zip(axes.flat, picks):
        rd = EXP_ROOT / rel
        val = load_epoch_series(rd, "validation.csv")
        if val is None:
            ax.set_title(f"{label} (no val csv)")
            continue
        ax.plot(val["epoch"], val["bitwise_error"] * 100, color=color, lw=2, label="BER")
        if "expert_max_use" in val.columns:
            ax2 = ax.twinx()
            ax2.plot(val["epoch"], val["expert_max_use"], color="gray", ls="--", lw=1.5, label="max_use")
            ax2.set_ylabel("expert_max_use", color="gray")
            ax2.axhline(0.5, color="red", ls=":", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BER (%)")
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training curves: BER and expert_max_use", fontsize=12)
    fig.tight_layout()
    save(fig, "09_training_curves_ber_maxuse.png")


def plot_expert_load_heatmap(df: pd.DataFrame):
    """Final validation expert loads for selected runs."""
    picks = [
        "unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
        "frozen_moe/coco100k/4exp_k1/batch32/sym_t14_temp09_bal004warm10_ep20_2026-06-01",
        "frozen_moe/coco100k/4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01",
        "frozen_moe/coco100k/8exp_k2/batch32/collapse_diag_jitter02_loadpen01_bal06_ep34_2026-05-31",
    ]
    labels = []
    matrices = []
    for rel in picks:
        val = load_csv(EXP_ROOT / rel / "validation.csv")
        if val is None:
            continue
        loads = expert_load_cols(val)
        if not loads:
            continue
        row = val.iloc[-1]
        vec = [float(row[c]) for c in loads]
        labels.append(rel.split("/")[-1][:35])
        matrices.append(vec)

    if not matrices:
        return
    max_e = max(len(m) for m in matrices)
    mat = np.zeros((len(matrices), max_e))
    for i, vec in enumerate(matrices):
        mat[i, : len(vec)] = vec

    fig, ax = plt.subplots(figsize=(10, 1 + len(matrices) * 0.6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, mat.max()))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(max_e))
    ax.set_xticklabels([f"E{i}" for i in range(max_e)])
    ax.set_title("Final validation expert loads (per-expert selection frequency)")
    plt.colorbar(im, ax=ax, label="Load")
    fig.tight_layout()
    save(fig, "10_expert_load_heatmap.png")


def plot_expert_load_over_epochs():
    rel_good = "unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01"
    rel_bad = "frozen_moe/coco100k/4exp_k1/batch32/sym_t14_temp09_bal004warm10_ep20_2026-06-01"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rel, title in zip(
        axes,
        [rel_good, rel_bad],
        ["Unfrozen (balanced)", "Frozen sym_t14 t0.9 (collapsed)"],
    ):
        val = load_epoch_series(EXP_ROOT / rel, "validation.csv")
        if val is None:
            continue
        loads = [c for c in val.columns if c.startswith("expert_load_")]
        for lc in loads:
            ax.plot(val["epoch"], val[lc], lw=1.5, label=lc.replace("_", " "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Expert load")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-expert load over training", fontsize=12)
    fig.tight_layout()
    save(fig, "11_expert_load_over_epochs.png")


def plot_attack_comparison():
    if not ATTACK_CSV.is_file():
        return
    atk = pd.read_csv(ATTACK_CSV)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(atk))
    w = 0.35
    ax.bar(x - w / 2, atk["hidden_bit_acc"] * 100, w, label="HiDDeN", color="#4c72b0")
    ax.bar(x + w / 2, atk["moe_bit_acc"] * 100, w, label="MoE unfrozen", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(atk["attack"], rotation=35, ha="right")
    ax.set_ylabel("Bit accuracy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", ls=":", label="chance (50%)")
    ax.legend()
    ax.set_title("Attack robustness: HiDDeN vs MoE (800 val images / attack)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "12_attack_bit_accuracy.png")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    colors = ["#55a868" if d >= 0 else "#c44e52" for d in atk["moe_minus_hidden_acc"]]
    ax2.barh(atk["attack"], atk["moe_minus_hidden_acc"] * 100, color=colors)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("MoE advantage (percentage points)")
    ax2.set_title("MoE minus HiDDeN bit accuracy")
    fig2.tight_layout()
    save(fig2, "13_attack_moe_advantage.png")


def plot_effective_experts(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    sub["ideal_eff"] = sub["num_experts"]  # perfect routing uses all experts
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        sub["num_experts"],
        sub["val_eff_exp"],
        s=sub["batch_size"] / 2,
        c=sub["val_max_use"],
        cmap="RdYlGn_r",
        vmin=0.2,
        vmax=0.7,
        edgecolors="k",
        linewidths=0.4,
    )
    for ne in sorted(sub["num_experts"].unique()):
        ax.axhline(ne, color="green", ls="--", alpha=0.3)
        ax.plot([ne - 0.2, ne + 0.2], [ne, ne], "g--", alpha=0.5)
    ax.set_xlabel("Number of experts")
    ax.set_ylabel("Effective experts (validation)")
    ax.set_title("Effective experts vs capacity (color = expert_max_use)")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("expert_max_use")
    ax.grid(True, alpha=0.3)
    save(fig, "14_effective_experts.png")


def plot_summary_dashboard(df: pd.DataFrame):
    """2x2 summary for thesis."""
    sub = df[(df["dataset"] == "coco100k") & (df["has_noisy_val"])].copy()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    for frozen, c in [(False, "#55a868"), (True, "#c44e52")]:
        m = sub[sub["frozen"] == frozen]
        ax1.scatter(m["val_max_use"], m["val_noisy_ber"] * 100, c=c, s=70, label="Unfrozen" if not frozen else "Frozen", alpha=0.85)
    ax1.axvline(0.5, color="red", ls="--", alpha=0.6)
    ax1.set_xlabel("expert_max_use")
    ax1.set_ylabel("Noisy val BER (%)")
    ax1.set_title("Collapse vs noisy BER")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    bs = sorted(sub["batch_size"].unique())
    ax2.scatter(sub["batch_size"], sub["val_max_use"], c=sub["val_ber"] * 100, cmap="viridis_r", s=80, edgecolors="k")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("expert_max_use")
    ax2.set_title("Batch size vs collapse (color=BER%)")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    if ATTACK_CSV.is_file():
        atk = pd.read_csv(ATTACK_CSV)
        x = np.arange(len(atk))
        ax3.bar(x - 0.2, atk["hidden_bit_acc"] * 100, 0.4, label="HiDDeN", color="#4c72b0")
        ax3.bar(x + 0.2, atk["moe_bit_acc"] * 100, 0.4, label="MoE", color="#55a868")
        ax3.set_xticks(x)
        ax3.set_xticklabels(atk["attack"], rotation=45, ha="right", fontsize=7)
        ax3.set_ylabel("Bit accuracy (%)")
        ax3.legend(fontsize=8)
        ax3.set_title("Attack comparison")

    ax4 = fig.add_subplot(gs[1, 1])
    top = sub.nsmallest(8, "val_noisy_ber")
    y = np.arange(len(top))
    ax4.barh(y, top["val_noisy_ber"] * 100, color=top["frozen"].map({True: "#c44e52", False: "#55a868"}))
    ax4.set_yticks(y)
    ax4.set_yticklabels(top["name"].str[:25], fontsize=7)
    ax4.set_xlabel("Noisy val BER (%)")
    ax4.set_title("Best runs (lowest noisy BER)")
    ax4.invert_yaxis()

    fig.suptitle("MoE experiment summary dashboard (COCO-100k)", fontsize=13, y=1.01)
    save(fig, "00_summary_dashboard.png")


def plot_legacy_batch_sweep(df: pd.DataFrame):
    """4exp k=2: batch 12/16/24/32 comparison."""
    sub = df[
        (df["name"].str.contains("balance_v2|uniform_first|ber_route", case=False, na=False))
        & (df["num_experts"] == 4)
    ].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col in zip(axes, ["val_ber", "train_max_use"]):
        for _, r in sub.iterrows():
            ax.scatter(r["batch_size"], r[col] * (100 if "ber" in col else 1), s=100, alpha=0.8)
            ax.annotate(r["name"][:15], (r["batch_size"], r[col] * (100 if "ber" in col else 1)), fontsize=6)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Legacy 4-expert batch-size sweep", fontsize=12)
    fig.tight_layout()
    save(fig, "15_legacy_4exp_batch_sweep.png")


def export_summary_table(df: pd.DataFrame):
    out = ROOT / "results" / "plots" / "experiment_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "name", "branch", "dataset", "num_experts", "top_k", "batch_size",
        "balance_w", "jitter", "load_pen", "frozen",
        "val_ber", "val_noisy_ber", "val_max_use", "val_load_l1", "val_eff_exp",
    ]
    df[cols].sort_values(["dataset", "val_max_use"]).to_csv(out, index=False, float_format="%.4f")
    print(f"Wrote {out}")


def plot_run_ranking(df: pd.DataFrame):
    """Combined score: noisy BER + routing penalty."""
    sub = df[(df["dataset"] == "coco100k") & (df["has_noisy_val"])].copy()
    sub["score"] = sub["val_noisy_ber"].fillna(sub["val_ber"]) + 0.5 * sub["val_max_use"].fillna(0.5) + 0.3 * sub["val_load_l1"].fillna(0.5)
    sub = sub.sort_values("score")
    fig, ax = plt.subplots(figsize=(10, max(5, len(sub) * 0.4)))
    y = np.arange(len(sub))
    colors = sub["frozen"].map({True: "#c44e52", False: "#55a868"})
    ax.barh(y, sub["score"], color=colors, edgecolor="gray", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(
        sub.apply(lambda r: f"{r['name'][:28]} (b{r['batch_size']})", axis=1),
        fontsize=7,
    )
    ax.set_xlabel("Combined score (lower = better BER + routing)")
    ax.set_title("Run ranking: noisy BER + 0.5·max_use + 0.3·load_l1")
    fig.tight_layout()
    save(fig, "16_run_ranking_combined_score.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    cols = ["batch_size", "balance_w", "jitter", "load_pen", "num_experts", "top_k", "val_ber", "val_noisy_ber", "val_max_use", "val_load_l1", "val_eff_exp"]
    mat = sub[cols].astype(float)
    corr = mat.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Metric correlation (COCO-100k runs)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, "17_correlation_heatmap.png")


def plot_dataset_20k_vs_100k(df: pd.DataFrame):
    """20k vs 100k boxplots, one row per expert count (4 and 8)."""
    sub = df[df["dataset"].isin(["coco20k", "coco100k"])].copy()
    if sub.empty:
        return
    sub["max_use_over_uniform"] = sub["val_max_use"] * sub["num_experts"]

    order = ["coco20k", "coco100k"]
    colors = {"coco20k": "#8172b2", "coco100k": "#4c72b0"}
    labels = {"coco20k": "20k", "coco100k": "100k"}
    expert_rows = [4, 8]
    panels = [
        ("max_use_over_uniform", "max_use × N (1=balanced)"),
        ("val_ber", "Val BER (%)"),
    ]

    fig, axes = plt.subplots(len(expert_rows), len(panels), figsize=(10, 7), sharex=False)
    rng = np.random.default_rng(42)

    for row_i, n_exp in enumerate(expert_rows):
        chunk = sub[sub["num_experts"] == n_exp]
        for col_i, (col, title) in enumerate(panels):
            ax = axes[row_i, col_i]
            data, tick_labels, positions = [], [], []
            for i, ds in enumerate(order):
                vals = chunk.loc[chunk["dataset"] == ds, col].dropna().values
                if col == "val_ber":
                    vals = vals * 100
                if len(vals) == 0:
                    continue
                data.append(vals)
                tick_labels.append(f"{labels[ds]}\nn={len(vals)}")
                positions.append(i + 1)
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full(len(vals), i + 1) + jitter,
                    vals,
                    s=55,
                    c=colors[ds],
                    edgecolors="k",
                    linewidths=0.4,
                    alpha=0.85,
                    zorder=3,
                )
            if data:
                bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True, showfliers=False)
                for patch, ds in zip(bp["boxes"], order[: len(data)]):
                    patch.set_facecolor(colors[ds])
                    patch.set_alpha(0.45)
            if col == "max_use_over_uniform":
                ax.axhline(1.0, color="gray", ls=":", lw=1)
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels)
            if col_i == 0:
                ax.set_ylabel(f"{n_exp} experts\n{title}")
            else:
                ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Dataset size (20k vs 100k) — compared separately for 4- and 8-expert runs\n"
        "Routing metric scaled by N so expert counts are comparable",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "19_dataset_20k_vs_100k.png")


def plot_branch_comparison(df: pd.DataFrame):
    sub = df[df["dataset"] == "coco100k"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["val_ber", "val_max_use", "val_noisy_ber"]):
        sub2 = sub[sub[col].notna()]
        branches = sorted(sub2["branch"].unique())
        data = [sub2[sub2["branch"] == b][col].values * (100 if "ber" in col else 1) for b in branches]
        bp = ax.boxplot(data, tick_labels=branches, patch_artist=True)
        for patch, b in zip(bp["boxes"], branches):
            patch.set_facecolor({"unfrozen_moe": "#55a868", "frozen_moe": "#c44e52", "legacy_early": "#4c72b0"}.get(b, "#888"))
            patch.set_alpha(0.7)
        ax.set_title(col.replace("_", " "))
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("By experiment branch (COCO-100k)", fontsize=12)
    fig.tight_layout()
    save(fig, "18_branch_boxplots.png")


def main():
    df = load_all_runs()
    if df.empty:
        print("No experiment CSVs found under", EXP_ROOT)
        return

    print(f"Loaded {len(df)} runs with train.csv")
    export_summary_table(df)

    plot_summary_dashboard(df)
    plot_collapse_comparison(df)
    plot_frozen_vs_unfrozen(df)
    plot_batch_size_effects(df)
    plot_num_experts_topk(df)
    plot_hyperparameter_scatters(df)
    plot_ber_vs_max_use(df)
    plot_training_curves(df)
    plot_expert_load_heatmap(df)
    plot_expert_load_over_epochs()
    plot_attack_comparison()
    plot_effective_experts(df)
    plot_legacy_batch_sweep(df)
    plot_run_ranking(df)
    plot_correlation_heatmap(df)
    plot_branch_comparison(df)
    plot_dataset_20k_vs_100k(df)

    print("Plots saved to:")
    for d in OUT_DIRS:
        print(f"  {d}/")
        for p in sorted(d.glob("*.png")):
            print(f"    {p.name}")


if __name__ == "__main__":
    main()
