"""Best epoch per run from validation.csv — checkpoint not required."""
import csv
from pathlib import Path

EXP_BASE = Path("results/experiments")
OUT_CSV = Path("results/BEST_EPOCH_ALL_EPOCHS.csv")
OUT_MD = Path("results/BEST_EPOCH_ALL_EPOCHS.md")

KEY_BER = "bitwise-error"
KEY_MU = "expert_max_use"
KEY_LL = "train_val_load_l1"
KEY_EFF = "effective_experts"


def ffloat(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def n_experts_from_path(parts):
    for p in parts:
        if m := __import__("re").match(r"(\d+)exp", p):
            return int(m.group(1))
    return 4


def stability_score(mu, ll, eff, n_exp):
    if mu is None:
        return None
    uniform = 1.0 / n_exp
    c = max(0.0, (mu - uniform) / (1.0 - uniform + 1e-9))
    score = 1.0 - 0.5 * min(1.0, c)
    if ll is not None:
        score -= 0.3 * min(1.0, ll / 1.5)
    if eff is not None and n_exp > 0:
        score += 0.2 * min(1.0, eff / n_exp)
    return max(0.0, min(1.0, score))


def composite(ber_pct, stab):
    if ber_pct is None or stab is None:
        return None
    ber_norm = max(0.0, 1.0 - ber_pct / 25.0)
    return 0.6 * ber_norm + 0.4 * stab


def label(run_name, branch):
    m = {
        "sym_t14_unfrozen_bal004warm10": "R0 PRIMARY",
        "sym_t14_unfrozen_continue": "R0 continuation",
        "sym_t14_unfrozen_attack": "Attack-trained",
        "sym_t14_unfrozen_200k": "200k data scale",
        "unfrozen_4exp_top2": "k=2 ablation",
        "unfrozen_4exp_dense_k4": "k=4 dense",
        "unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02": "8exp run A",
        "unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03": "8exp run B",
        "sym_bal08_jitter0_bal08warm5_temp14to10_ep20": "Frozen bal08 ep20",
        "sym_bal08_jitter0_bal08warm5_temp14to10_ep30": "Frozen bal08 ep30",
        "routing_isolation": "Routing isolation",
        "collapse_diag": "8exp collapse diag",
        "sym_t14_temp09": "Frozen t14 temp0.9",
    }
    for k, v in m.items():
        if k in run_name:
            return v
    return "Legacy" if branch == "legacy_early" else run_name[:36]


rows_out = []

for val_csv in sorted(EXP_BASE.rglob("validation.csv")):
    if "images" in str(val_csv):
        continue
    run_dir = val_csv.parent
    parts = run_dir.relative_to(EXP_BASE).parts
    branch, dataset, ne, batch_dir, run_name = parts[0], parts[1], parts[2], parts[3], parts[4]
    n_exp = n_experts_from_path(parts)

    data = list(csv.DictReader(open(val_csv, encoding="utf-8")))
    if not data:
        continue

    candidates = []
    for r in data:
        ep = int(r["epoch"])
        ber_pct = ffloat(r.get(KEY_BER))
        if ber_pct is None:
            continue
        ber_pct *= 100
        mu = ffloat(r.get(KEY_MU))
        ll = ffloat(r.get(KEY_LL))
        eff = ffloat(r.get(KEY_EFF))
        stab = stability_score(mu, ll, eff, n_exp)
        comp = composite(ber_pct, stab)
        candidates.append({
            "epoch": ep, "ber_pct": ber_pct, "max_use": mu, "load_l1": ll,
            "stability": stab, "composite": comp,
        })

    if not candidates:
        continue

    best_ber = min(candidates, key=lambda x: x["ber_pct"])
    best_comp = max(candidates, key=lambda x: x["composite"] or 0)
    best_stab = max(candidates, key=lambda x: x["stability"] or 0)
    final = candidates[-1]

    post_best_ber = None
    post_best_comp = None
    if "continue" in run_name:
        post = [c for c in candidates if c["epoch"] > 20]
        if post:
            post_best_ber = min(post, key=lambda x: x["ber_pct"])
            post_best_comp = max(post, key=lambda x: x["composite"] or 0)

    rows_out.append({
        "label": label(run_name, branch),
        "run_folder": run_name,
        "branch": branch,
        "dataset": dataset,
        "n_exp": n_exp,
        "ne_k": ne,
        "final_epoch": final["epoch"],
        "final_ber_pct": round(final["ber_pct"], 3),
        "best_ber_epoch": best_ber["epoch"],
        "best_ber_pct": round(best_ber["ber_pct"], 3),
        "best_ber_max_use": round(best_ber["max_use"], 4) if best_ber["max_use"] else "",
        "best_ber_load_l1": round(best_ber["load_l1"], 4) if best_ber["load_l1"] else "",
        "best_composite_epoch": best_comp["epoch"],
        "best_composite_ber_pct": round(best_comp["ber_pct"], 3),
        "best_composite_max_use": round(best_comp["max_use"], 4) if best_comp["max_use"] else "",
        "best_composite_load_l1": round(best_comp["load_l1"], 4) if best_comp["load_l1"] else "",
        "best_stability_epoch": best_stab["epoch"],
        "best_stability_ber_pct": round(best_stab["ber_pct"], 3),
        "post20_best_ber_epoch": post_best_ber["epoch"] if post_best_ber else "",
        "post20_best_ber_pct": round(post_best_ber["ber_pct"], 3) if post_best_ber else "",
        "post20_best_composite_epoch": post_best_comp["epoch"] if post_best_comp else "",
        "post20_best_composite_ber_pct": round(post_best_comp["ber_pct"], 3) if post_best_comp else "",
    })

# CSV
if rows_out:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

# MD
md = [
    "# Best epoch per run (all epochs — checkpoint not required)",
    "",
    "| Label | Best BER ep | BER% | Best composite ep | BER% | max_use | load_l1 | Final ep BER% |",
    "|-------|-------------|------|-------------------|------|---------|---------|---------------|",
]
for r in sorted(rows_out, key=lambda x: x["best_ber_pct"]):
    md.append(
        f"| {r['label']} | **{r['best_ber_epoch']}** | {r['best_ber_pct']} | **{r['best_composite_epoch']}** | {r['best_composite_ber_pct']} | {r['best_composite_max_use']} | {r['best_composite_load_l1']} | ep{r['final_epoch']} {r['final_ber_pct']}% |"
    )
OUT_MD.write_text("\n".join(md), encoding="utf-8")

print(f"Wrote {OUT_CSV} ({len(rows_out)} runs)\n")
print("=== THESIS RUNS ===")
print(f"{'Label':<28} {'best BER ep':>11} {'BER%':>7}  {'composite ep':>12} {'BER%':>7}  {'max_use':>7} {'load_l1':>7}")
print("-" * 90)
for lbl in ["R0 PRIMARY", "R0 continuation", "Attack-trained", "200k data scale", "k=2 ablation", "k=4 dense", "8exp run A", "8exp run B", "Frozen bal08 ep20", "Frozen bal08 ep30", "Routing isolation", "8exp collapse diag"]:
    for r in rows_out:
        if r["label"] == lbl:
            extra = ""
            if r["post20_best_ber_epoch"]:
                extra = f"  | post-20 BER: ep{r['post20_best_ber_epoch']} ({r['post20_best_ber_pct']}%)"
            print(f"{lbl:<28} ep{r['best_ber_epoch']:>8} {r['best_ber_pct']:>7.2f}  ep{r['best_composite_epoch']:>10} {r['best_composite_ber_pct']:>7.2f}  {str(r['best_composite_max_use']):>7} {str(r['best_composite_load_l1']):>7}{extra}")
            break
