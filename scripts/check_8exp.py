import csv
from pathlib import Path

# 8exp run B epoch-by-epoch
f = Path("results/experiments/unfrozen_moe/coco100k/8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03/validation.csv")
rows = list(csv.DictReader(open(f)))
print("8exp run B (2026-06-03) by epoch:")
for r in rows:
    ber = float(r["bitwise-error"])*100
    mu = float(r["expert_max_use"])
    ll_v = r.get("train_val_load_l1","")
    ll_s = f"{float(ll_v):.3f}" if ll_v else "?"
    print(f"  ep{r['epoch']:>3}: BER={ber:.3f}%  max_use={mu:.3f}  load_l1={ll_s}")

print()
print("attack_summary_8exp_300k_ep15 identity BER:")
f2 = Path("results/comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15.csv")
for r in csv.DictReader(open(f2)):
    if r["attack"] == "identity":
        print(f"  identity moe_bit_acc={float(r['moe_bit_acc'])*100:.3f}%  BER={float(r['moe_ber'])*100:.3f}%")

print()
print("attack_summary_8exp_k1_ep20 identity BER:")
f3 = Path("results/comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20.csv")
for r in csv.DictReader(open(f3)):
    if r["attack"] == "identity":
        print(f"  identity moe_bit_acc={float(r['moe_bit_acc'])*100:.3f}%  BER={float(r['moe_ber'])*100:.3f}%")

# Check 200k run duration per epoch (proxy for dataset size)
print()
f4 = Path("results/experiments/unfrozen_moe/coco200k/4exp_k1/batch128/sym_t14_unfrozen_200k_v1_ep16_2026-06-05/train.csv")
rows4 = list(csv.DictReader(open(f4)))
print(f"200k run train.csv: {len(rows4)} rows")
for r in rows4[:3]:
    dur = r.get("duration","?")
    print(f"  ep{r['epoch']}: duration_s={dur}")

# Compare duration with R0 run
print()
f5 = Path("results/experiments/unfrozen_moe/coco100k/4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/train.csv")
rows5 = list(csv.DictReader(open(f5)))
print(f"R0 (100k) train.csv: {len(rows5)} rows")
for r in rows5[:3]:
    dur = r.get("duration","?")
    print(f"  ep{r['epoch']}: duration_s={dur}")

# Also check the train4exp200k.txt log for dataset info
print()
txt = Path("results/train4exp200k.txt").read_text(errors="replace")
lines = [l.strip() for l in txt.splitlines() if l.strip()]
# Find data-dir and epoch/step info
for l in lines:
    if "data-dir" in l or "coco" in l.lower() or "epoch" in l.lower() and "step" in l.lower():
        print(f"  log: {l[:120]}")
        break
# Find step counts
step_lines = [l for l in lines if "/step" in l.lower() or "step:" in l.lower()][:3]
for l in step_lines:
    print(f"  steps: {l[:80]}")
