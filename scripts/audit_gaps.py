"""Print a clear gap report: what's in the results folder but not in the thesis narrative."""
import csv
from pathlib import Path

def read_atk(fname):
    p = Path(fname)
    if not p.exists(): return {}
    return {r["attack"]: float(r["moe_bit_acc"])*100
            for r in csv.DictReader(open(p))}

hidden_ref = {r["attack"]: float(r["hidden_bit_acc"])*100
              for r in csv.DictReader(open("results/comparison_hidden_vs_moe/attack_summary_main_ep20.csv"))}

runs = {
    "R0 ep20 (THESIS)":          "results/ber_500_attacks/attack_summary_moe_r0.csv",
    "R0 ep20 (comparison/)":     "results/comparison_hidden_vs_moe/attack_summary_main_ep20.csv",
    "Atk-trained ep30 *** NEW":"results/comparison_hidden_vs_moe/attack_summary_attack_v1_ep30.csv",
    "Atk-trained ep65 (THESIS)": "results/ber_500_attacks/moe_semicollapsed_attack_ep65_attack_summary.csv",
    "8exp 100k k=1 ep20":        "results/comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20.csv",
    "8exp 300k k=1 ep15 *** NEW":"results/comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15.csv",
    "Dense 4exp k=4 ep20":       "results/comparison_hidden_vs_moe/attack_summary_dense_4exp_k4_ep20.csv",
    "Top-2 k=2 ep20 (top2 ep20)":"results/comparison_hidden_vs_moe/attack_summary_top2_ep20.csv",
    "Routing isolation (collapse)":"results/ber_500_attacks/moe_collapsed_routing_isolation_attack_summary.csv",
    "ep38 continued":            "results/ber_500_attacks/moe_continue_ep38_attack_summary.csv",
    "ep52 continued":            "results/ber_500_attacks/moe_continue_ep52_attack_summary.csv",
}

ATK = ["identity","jpeg","quant","crop","cropout","dropout","resize","gaussian","combined"]

def wins(data):
    return sum(1 for a in ATK if data.get(a,0) > hidden_ref.get(a,0) + 0.5)

print("\n=== ATTACK COMPARISON — all evaluated models ===")
print(f"{'Model':<40} {'Win':>4}  " + "  ".join(f"{a[:6]:>7}" for a in ATK))
print("-"*120)
for label, fpath in runs.items():
    d = read_atk(fpath)
    if not d:
        print(f"  {label:<38}  FILE MISSING")
        continue
    w = wins(d)
    vals = "  ".join(f"{d.get(a,0):>7.1f}" for a in ATK)
    print(f"  {label:<38} {w:>3}/9  {vals}")

print("\n=== KEY NUMBERS — attack-trained ep30 vs ep65 ===")
ep30 = read_atk("results/comparison_hidden_vs_moe/attack_summary_attack_v1_ep30.csv")
ep65 = read_atk("results/ber_500_attacks/moe_semicollapsed_attack_ep65_attack_summary.csv")
ep68_raw = "results/ber_500_attacks/moe_semicollapsed_attack_ep65_attack_summary.csv"
r0   = read_atk("results/comparison_hidden_vs_moe/attack_summary_main_ep20.csv")
print(f"{'Attack':<12} {'HiDDeN':>8} {'R0 ep20':>8} {'atk ep30':>9} {'Δep30':>7} {'atk ep65':>9} {'Δep65':>7}")
print("-"*70)
for a in ATK:
    h = hidden_ref.get(a, 0)
    r0v = r0.get(a, 0)
    e30 = ep30.get(a, 0)
    e65 = ep65.get(a, 0)
    d30 = e30 - h
    d65 = e65 - h
    flag = "  <-- !!" if abs(d30) > 10 else ""
    print(f"{a:<12} {h:>8.1f} {r0v:>8.1f} {e30:>9.1f} {d30:>+7.1f} {e65:>9.1f} {d65:>+7.1f}{flag}")

print(f"\n  ep30 wins: {wins(ep30)}/9    ep65 wins: {wins(ep65)}/9")

print("\n=== 8exp 300k vs 8exp 100k ===")
exp_300k = read_atk("results/comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15.csv")
exp_100k = read_atk("results/comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20.csv")
print(f"{'Attack':<12} {'HiDDeN':>8} {'8exp 100k':>10} {'Δ100k':>7} {'8exp 300k':>10} {'Δ300k':>7}")
print("-"*65)
for a in ATK:
    h = hidden_ref.get(a,0)
    v1 = exp_100k.get(a,0)
    v3 = exp_300k.get(a,0)
    print(f"{a:<12} {h:>8.1f} {v1:>10.1f} {v1-h:>+7.1f} {v3:>10.1f} {v3-h:>+7.1f}")
print(f"\n  8exp 100k wins: {wins(exp_100k)}/9    8exp 300k wins: {wins(exp_300k)}/9")
