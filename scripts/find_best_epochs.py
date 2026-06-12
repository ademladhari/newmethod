"""
For each experiment run, find the best epoch by BER + routing stability
among epochs that have a saved checkpoint.
"""
import csv
import json
import re
from pathlib import Path

RESULTS = Path("results")
EXP_BASE = RESULTS / "experiments"
ARCHIVE_CHK = RESULTS / "archive" / "moe_run"
OUT_CSV = RESULTS / "BEST_EPOCHS_PER_RUN.csv"
OUT_MD = RESULTS / "BEST_EPOCHS_PER_RUN.md"

KEY_BER = "bitwise-error"
KEY_MU = "expert_max_use"
KEY_LL = "train_val_load_l1"
KEY_EFF = "effective_experts"


def parse_checkpoint_epochs(chk_dir: Path):
    """Return set of epoch ints from .pyt filenames."""
    if not chk_dir.exists():
        return set()
    epochs = set()
    for f in chk_dir.glob("*.pyt"):
        m = re.search(r"epoch-(\d+)", f.name)
        if m:
            epochs.add(int(m.group(1)))
    return epochs


def read_val_rows(val_csv: Path):
    if not val_csv.exists():
        return []
    try:
        return list(csv.DictReader(open(val_csv, encoding="utf-8")))
    except Exception:
        return []


def ffloat(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def collapse_score(max_use, n_experts):
    """0 = healthy uniform, 1 = full collapse."""
    if max_use is None or n_experts <= 0:
        return None
    uniform = 1.0 / n_experts
    return max(0.0, (max_use - uniform) / (1.0 - uniform + 1e-9))


def n_experts_from_path(parts):
  for p in parts:
    m = re.match(r"(\d+)exp", p)
    if m:
      return int(m.group(1))
  return 4


def stability_score(row, n_exp):
    """Higher = more stable. Combines routing health."""
    mu = ffloat(row.get(KEY_MU))
    ll = ffloat(row.get(KEY_LL))
    eff = ffloat(row.get(KEY_EFF))
    if mu is None:
        return None
    c = collapse_score(mu, n_exp)
    # penalize collapse and train-val gap
    score = 1.0
    if c is not None:
        score -= 0.5 * min(1.0, c)
    if ll is not None:
        score -= 0.3 * min(1.0, ll / 1.5)
    if eff is not None and n_exp > 0:
        score += 0.2 * min(1.0, eff / n_exp)
    return max(0.0, min(1.0, score))


def composite_score(ber_pct, stab, ber_weight=0.6):
    """Lower BER better; higher stability better. Returns score to maximize."""
    if ber_pct is None or stab is None:
        return None
    ber_norm = max(0.0, 1.0 - ber_pct / 25.0)  # 25% = worst in archive
    return ber_weight * ber_norm + (1 - ber_weight) * stab


def pick_best(rows, saved_epochs, n_exp, min_epoch=0, mode="composite"):
    """Pick best epoch among saved checkpoints only."""
    by_ep = {int(r["epoch"]): r for r in rows}
    candidates = []
    for ep in sorted(saved_epochs):
        if ep < min_epoch:
            continue
        if ep not in by_ep:
            continue
        r = by_ep[ep]
        ber = ffloat(r.get(KEY_BER))
        if ber is None:
            continue
        ber_pct = ber * 100
        mu = ffloat(r.get(KEY_MU))
        ll = ffloat(r.get(KEY_LL))
        stab = stability_score(r, n_exp)
        comp = composite_score(ber_pct, stab)
        candidates.append({
            "epoch": ep,
            "ber_pct": ber_pct,
            "max_use": mu,
            "load_l1": ll,
            "stability": stab,
            "composite": comp,
        })

    if not candidates:
        return None, [], None

    if mode == "ber_only":
        best = min(candidates, key=lambda x: x["ber_pct"])
    elif mode == "stability":
        best = max(candidates, key=lambda x: x["stability"] or 0)
    else:
        best = max(candidates, key=lambda x: x["composite"] or 0)

    # Also track pure lowest BER among saved (for reference)
    lowest_ber = min(candidates, key=lambda x: x["ber_pct"])
    return best, candidates, lowest_ber


def extra_checkpoints_for_run(run_name: str):
    """archive/moe_run for continuation run."""
    if "continue" in run_name or run_name == "sym_t14_unfrozen_continue_ep60_2026-06-01":
        return parse_checkpoint_epochs(ARCHIVE_CHK)
    return set()


def run_label(run_name, branch):
    if "sym_t14_unfrozen_bal004warm10" in run_name:
        return "R0 PRIMARY"
    if "continue" in run_name:
        return "R0 continuation"
    if "attack_v1" in run_name:
        return "Attack-trained"
    if "200k" in run_name:
        return "200k data scale"
    if "unfrozen_4exp_top2" in run_name:
        return "k=2 ablation (100k)"
    if "dense_k4" in run_name:
        return "k=4 dense"
    if "8exp_sparse" in run_name and "2026-06-02" in run_name:
        return "8exp run A"
    if "8exp_sparse" in run_name and "2026-06-03" in run_name:
        return "8exp run B"
    if "sym_bal08" in run_name and "ep20" in run_name:
        return "Frozen bal08 ep20"
    if "sym_bal08" in run_name and "ep30" in run_name:
        return "Frozen bal08 ep30"
    if "routing_isolation" in run_name:
        return "Routing isolation (collapse)"
    if "collapse_diag" in run_name:
        return "8exp collapse diag"
    if branch == "legacy_early":
        return "Legacy"
    return run_name[:40]


results = []

for val_csv in sorted(EXP_BASE.rglob("validation.csv")):
    if "images" in str(val_csv):
        continue
    run_dir = val_csv.parent
    parts = run_dir.relative_to(EXP_BASE).parts
    branch, dataset, ne, batch_dir, run_name = parts[0], parts[1], parts[2], parts[3], parts[4]
    n_exp = n_experts_from_path(parts)

    rows = read_val_rows(val_csv)
    if not rows:
        continue

    chk_epochs = parse_checkpoint_epochs(run_dir / "checkpoints")
    chk_epochs |= extra_checkpoints_for_run(run_name)

    if not chk_epochs:
        final = rows[-1]
        results.append({
            "run_folder": run_name,
            "label": run_label(run_name, branch),
            "branch": branch,
            "dataset": dataset,
            "n_exp": n_exp,
            "ne_k": ne,
            "batch": batch_dir,
            "saved_epochs": "",
            "n_saved": 0,
            "best_epoch": "",
            "best_ber_pct": "",
            "best_max_use": "",
            "best_load_l1": "",
            "best_stability": "",
            "best_composite": "",
            "lowest_ber_epoch": "",
            "lowest_ber_pct": "",
            "final_epoch": int(final["epoch"]),
            "final_ber_pct": ffloat(final.get(KEY_BER), 0) * 100,
            "checkpoint_saved": "NO",
            "note": "No checkpoints in folder",
        })
        continue

    best, candidates, lowest = pick_best(rows, chk_epochs, n_exp)
    if best is None:
        continue

    # For continuation runs, also find best among epochs AFTER the R0 baseline (ep>20)
    post_best = None
    post_lowest = None
    min_post = 21 if "continue" in run_name else 0
    if min_post:
        post_best, _, post_lowest = pick_best(rows, chk_epochs, n_exp, min_epoch=min_post)

    saved_str = ",".join(str(e) for e in sorted(chk_epochs))
    final = rows[-1]
    final_ep = int(final["epoch"])
    final_ber = ffloat(final.get(KEY_BER), 0) * 100

    note = ""
    if lowest["epoch"] != best["epoch"]:
        note = f"Lowest-BER saved ep={lowest['epoch']} ({lowest['ber_pct']:.2f}%) vs composite-best ep={best['epoch']}"
    if best["epoch"] != final_ep and abs(best["ber_pct"] - final_ber) > 0.5:
        note += f"; final ep{final_ep} BER={final_ber:.2f}% worse"
    if post_best and post_best["epoch"] != best["epoch"]:
        note += f"; best AFTER ep20 (saved): ep{post_best['epoch']} BER={post_best['ber_pct']:.2f}% max_use={post_best['max_use']:.3f} load_l1={post_best['load_l1']:.3f}"

    results.append({
        "run_folder": run_name,
        "label": run_label(run_name, branch),
        "branch": branch,
        "dataset": dataset,
        "n_exp": n_exp,
        "ne_k": ne,
        "batch": batch_dir,
        "saved_epochs": saved_str,
        "n_saved": len(chk_epochs),
        "best_epoch": best["epoch"],
        "best_ber_pct": round(best["ber_pct"], 3),
        "best_max_use": round(best["max_use"], 4) if best["max_use"] is not None else "",
        "best_load_l1": round(best["load_l1"], 4) if best["load_l1"] is not None else "",
        "best_stability": round(best["stability"], 3) if best["stability"] is not None else "",
        "best_composite": round(best["composite"], 3) if best["composite"] is not None else "",
        "lowest_ber_epoch": lowest["epoch"],
        "lowest_ber_pct": round(lowest["ber_pct"], 3),
        "best_post_ep20_epoch": post_best["epoch"] if post_best else "",
        "best_post_ep20_ber_pct": round(post_best["ber_pct"], 3) if post_best else "",
        "best_post_ep20_max_use": round(post_best["max_use"], 4) if post_best and post_best["max_use"] is not None else "",
        "best_post_ep20_load_l1": round(post_best["load_l1"], 4) if post_best and post_best["load_l1"] is not None else "",
        "final_epoch": final_ep,
        "final_ber_pct": round(final_ber, 3),
        "checkpoint_saved": "YES",
        "note": note.strip("; "),
    })

# Write CSV
if results:
    fields = list(results[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

# Write MD
lines = [
    "# Best epoch per experiment (saved checkpoint required)",
    "",
    "Selection: among **saved checkpoints only**, maximize composite score = 60% low BER + 40% routing stability.",
    "Stability penalizes high `max_use` (collapse) and high `load_l1` (train-val routing gap).",
    "",
    "| # | Label | Run | Best ep | BER% | max_use | load_l1 | Stab | Saved eps | Final ep BER |",
    "|---|-------|-----|---------|------|---------|---------|------|-----------|--------------|",
]
for i, r in enumerate(sorted(results, key=lambda x: (x["branch"], x["best_ber_pct"] if isinstance(x.get("best_ber_pct"), (int, float)) else 999)), 1):
    if r["checkpoint_saved"] == "NO":
        lines.append(f"| {i} | {r['label']} | `{r['run_folder'][:45]}` | — | — | — | — | — | 0 | ep{r['final_epoch']} {r['final_ber_pct']}% |")
        continue
    lines.append(
        f"| {i} | **{r['label']}** | `{r['run_folder'][:40]}` | **{r['best_epoch']}** | {r['best_ber_pct']} | {r['best_max_use']} | {r['best_load_l1']} | {r['best_stability']} | {r['n_saved']} | ep{r['final_epoch']} {r['final_ber_pct']}% |"
    )

lines.append("")
lines.append("## Notes per run")
lines.append("")
for r in results:
    if r["checkpoint_saved"] == "NO":
        continue
    lines.append(f"### {r['label']} — `{r['run_folder']}`")
    lines.append(f"- **Use checkpoint:** epoch **{r['best_epoch']}** (BER {r['best_ber_pct']}%, stability {r['best_stability']})")
    lines.append(f"- Saved epochs: {r['saved_epochs']}")
    if r["lowest_ber_epoch"] != r["best_epoch"]:
        lines.append(f"- Lowest BER among saved: ep **{r['lowest_ber_epoch']}** ({r['lowest_ber_pct']}%) — less stable than composite pick")
    if r["note"]:
        lines.append(f"- {r['note']}")
    lines.append("")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_CSV} ({len(results)} runs)")
print(f"Wrote {OUT_MD}")

# Print thesis-relevant summary
print("\n=== THESIS RUNS — recommended checkpoint ===")
key_runs = ["R0 PRIMARY", "R0 continuation", "Attack-trained", "200k data scale", "k=2 ablation (100k)", "k=4 dense", "8exp run A", "8exp run B", "Frozen bal08 ep20", "Routing isolation (collapse)"]
for label in key_runs:
    for r in results:
        if r["label"] == label and r["checkpoint_saved"] == "YES":
            extra = ""
            if r.get("best_post_ep20_epoch"):
                extra = f"  | post-ep20: ep{r['best_post_ep20_epoch']} BER={r['best_post_ep20_ber_pct']}%"
            print(f"  {label:<28} ep{r['best_epoch']:>3}  BER={r['best_ber_pct']}%  max_use={r['best_max_use']}  load_l1={r['best_load_l1']}{extra}")
            break
