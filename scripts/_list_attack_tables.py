import csv
from pathlib import Path

ROOT = Path(r"D:/new method/results")
files = sorted(ROOT.rglob("*attack*summary*.csv"))
# exclude per-image stuff if any
files = [f for f in files if "identity_test" not in f.name]

RUN_LABELS = {
    "attack_summary_main_ep20.csv": "R0 primary (ep20)",
    "attack_summary.csv": "R0 early/legacy eval",
    "attack_summary_moe_r0.csv": "R0 (ber_500, 512 img)",
    "attack_summary_frozen_ep30.csv": "Frozen backbone ep30",
    "attack_summary_attack_v1_ep30.csv": "Attack-trained ep30",
    "attack_summary_8exp_k1_ep20.csv": "8exp run A — 100k ep20",
    "attack_summary_8exp_300k_ep15.csv": "8exp run B — ~208k ep15",
    "attack_summary_dense_4exp_k4_ep20.csv": "Dense k=4 ep20",
    "attack_summary_top2_ep20.csv": "Top-2 k=2 ep20",
    "moe_continue_ep38_attack_summary.csv": "R0 continue ep38",
    "moe_continue_ep52_attack_summary.csv": "R0 continue ep52",
    "moe_semicollapsed_attack_ep65_attack_summary.csv": "Attack-trained ep65",
    "moe_collapsed_routing_isolation_attack_summary.csv": "Routing isolation (collapse)",
    "moe_8exp_collapse_diag_attack_summary.csv": "8exp collapse diag",
}

ATTACKS = [
    "identity", "jpeg", "quant", "crop", "cropout",
    "dropout", "resize", "gaussian", "combined",
]

print("=" * 100)
print("ALL ATTACK EVAL TABLES (HiDDeN vs MoE)")
print("=" * 100)

for f in files:
    rel = f.relative_to(ROOT)
    label = RUN_LABELS.get(f.name, f.name)
    rows = {r["attack"]: r for r in csv.DictReader(open(f))}
    n_img = rows.get("identity", {}).get("images", "?")
    wins = losses = ties = 0
    for a in ATTACKS:
        if a not in rows:
            continue
        d = float(rows[a]["moe_minus_hidden_acc"]) * 100
        if d > 1.5:
            wins += 1
        elif d < -1.5:
            losses += 1
        else:
            ties += 1
    id_moe = float(rows["identity"]["moe_bit_acc"]) * 100 if "identity" in rows else float("nan")
    id_h = float(rows["identity"]["hidden_bit_acc"]) * 100 if "identity" in rows else float("nan")

    print()
    print(f"FILE: {rel}")
    print(f"RUN:  {label}")
    print(f"Images: {n_img}  |  MoE wins (>1.5pp): {wins}/9  losses: {losses}  near-tie: {ties}")
    print(f"Identity — HiDDeN: {id_h:.2f}%  MoE: {id_moe:.2f}%  (BER MoE: {float(rows['identity']['moe_ber'])*100:.3f}%)")
    print(f"{'Attack':<10} {'HiDDeN%':>8} {'MoE%':>8} {'Δ pp':>8}")
    print("-" * 38)
    for a in ATTACKS:
        if a not in rows:
            continue
        r = rows[a]
        h = float(r["hidden_bit_acc"]) * 100
        m = float(r["moe_bit_acc"]) * 100
        d = float(r["moe_minus_hidden_acc"]) * 100
        mark = " +" if d > 1.5 else (" -" if d < -1.5 else "  ")
        print(f"{a:<10} {h:8.1f} {m:8.1f} {d:+8.1f}{mark}")
