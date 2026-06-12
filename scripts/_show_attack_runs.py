import csv
from pathlib import Path

ROOT = Path(r"D:/new method/results")

RUNS = [
    ("Attack-trained ep30", ROOT / "comparison_hidden_vs_moe/attack_summary_attack_v1_ep30.csv"),
    ("Attack-trained ep65", ROOT / "ber_500_attacks/moe_semicollapsed_attack_ep65_attack_summary.csv"),
    ("Dense k=4 ep20", ROOT / "comparison_hidden_vs_moe/attack_summary_dense_4exp_k4_ep20.csv"),
    ("R0 main ep20", ROOT / "comparison_hidden_vs_moe/attack_summary_main_ep20.csv"),
    ("8exp ~208k ep15", ROOT / "comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15.csv"),
    ("8exp 100k ep20", ROOT / "comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20.csv"),
    ("Frozen ep30", ROOT / "ber_500_attacks/attack_summary_frozen_ep30.csv"),
    ("Routing isolation", ROOT / "ber_500_attacks/moe_collapsed_routing_isolation_attack_summary.csv"),
]

ATTACKS = [
    "identity", "jpeg", "quant", "crop", "cropout",
    "dropout", "resize", "gaussian", "combined",
]

for label, path in RUNS:
    rows = {r["attack"]: r for r in csv.DictReader(open(path))}
    n_img = rows["identity"]["images"]
    id_ber = float(rows["identity"]["moe_ber"]) * 100
    wins = losses = ties = 0
    best_gains = []
    for a in ATTACKS:
        d = float(rows[a]["moe_minus_hidden_acc"]) * 100
        if d > 1.5:
            wins += 1
            best_gains.append((d, a))
        elif d < -1.5:
            losses += 1
        else:
            ties += 1
    best_gains.sort(reverse=True)
    top = ", ".join(f"{a} {d:+.1f}" for d, a in best_gains[:3]) or "—"

    print("=" * 72)
    print(f"{label}")
    print(f"File: {path.relative_to(ROOT)}")
    print(f"Images: {n_img}  |  Identity MoE BER: {id_ber:.2f}%  |  Wins: {wins}  Losses: {losses}  Ties: {ties}")
    print(f"Best gains: {top}")
    print(f"{'Attack':<10} {'HiDDeN%':>8} {'MoE%':>8} {'Δ pp':>8}  {'':>3}")
    print("-" * 42)
    for a in ATTACKS:
        r = rows[a]
        h = float(r["hidden_bit_acc"]) * 100
        m = float(r["moe_bit_acc"]) * 100
        d = float(r["moe_minus_hidden_acc"]) * 100
        tag = "WIN" if d > 1.5 else ("LOSS" if d < -1.5 else "")
        print(f"{a:<10} {h:8.1f} {m:8.1f} {d:+8.1f}  {tag:>3}")
    print()
