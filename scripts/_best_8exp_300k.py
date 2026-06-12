import csv
import re
from pathlib import Path

val = Path(
    r"D:/new method/results/experiments/unfrozen_moe/coco100k/8exp_k1/batch128/"
    r"unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03/validation.csv"
)
rows = list(csv.DictReader(open(val)))
chk_dir = val.parent / "checkpoints"
saved = sorted(
    int(m.group(1))
    for f in chk_dir.glob("*.pyt")
    if (m := re.search(r"epoch-(\d+)", f.name))
)

print("Run: 8exp run B — N=8, k=1, coco300k (~208k images/epoch)")
print("Saved checkpoint epochs:", saved or "none")
print()
print(f"{'ep':>3}  {'BER%':>7}  {'max_use':>8}  {'load_l1':>8}  saved")
for r in rows:
    ep = int(r["epoch"])
    ber = float(r["bitwise-error"]) * 100
    mu = float(r["expert_max_use"])
    ll = float(r["train_val_load_l1"])
    sav = "yes" if ep in saved else ""
    print(f"{ep:3d}  {ber:7.3f}  {mu:8.4f}  {ll:8.4f}  {sav}")

saved_rows = [r for r in rows if int(r["epoch"]) in saved] if saved else rows

best_ber = min(rows, key=lambda r: float(r["bitwise-error"]))
best_saved_ber = min(saved_rows, key=lambda r: float(r["bitwise-error"]))

# load_l1 < 0.20 healthy threshold
healthy = [r for r in rows if float(r["train_val_load_l1"]) <= 0.20]
best_healthy = min(healthy, key=lambda r: float(r["bitwise-error"])) if healthy else None

healthy_saved = [r for r in saved_rows if float(r["train_val_load_l1"]) <= 0.20]
best_healthy_saved = (
    min(healthy_saved, key=lambda r: float(r["bitwise-error"])) if healthy_saved else None
)

def composite(r):
    ber = float(r["bitwise-error"]) * 100
    mu = float(r["expert_max_use"])
    ll = float(r["train_val_load_l1"])
    ber_score = max(0.0, 1.0 - ber / 5.0)
    mu_pen = max(0.0, (mu - 0.125) / 0.875)
    ll_pen = min(1.0, ll / 0.5)
    stab = 1.0 - 0.5 * mu_pen - 0.5 * ll_pen
    return 0.6 * ber_score + 0.4 * stab

best_comp_saved = max(saved_rows, key=composite)

print()
print("--- Best epoch picks ---")
for label, r in [
    ("Lowest BER (any epoch)", best_ber),
    ("Lowest BER (saved checkpoint)", best_saved_ber),
    ("Best composite BER+routing (saved)", best_comp_saved),
]:
    print(
        f"{label}: ep{r['epoch']}  "
        f"BER={float(r['bitwise-error'])*100:.3f}%  "
        f"max_use={float(r['expert_max_use']):.4f}  "
        f"load_l1={float(r['train_val_load_l1']):.4f}"
    )
if best_healthy:
    print(
        f"Lowest BER with load_l1<=0.20: ep{best_healthy['epoch']}  "
        f"BER={float(best_healthy['bitwise-error'])*100:.3f}%  "
        f"load_l1={float(best_healthy['train_val_load_l1']):.4f}"
    )
if best_healthy_saved:
    print(
        f"Best saved with load_l1<=0.20: ep{best_healthy_saved['epoch']}  "
        f"BER={float(best_healthy_saved['bitwise-error'])*100:.3f}%  "
        f"load_l1={float(best_healthy_saved['train_val_load_l1']):.4f}"
    )

atk = Path(r"D:/new method/results/comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15.csv")
if atk.exists():
    print()
    print("Attack eval at ep15 (attack_summary_8exp_300k_ep15.csv):")
    wins = 0
    for r in csv.DictReader(open(atk)):
        h = float(r["hidden_bit_acc"]) * 100
        m = float(r["moe_bit_acc"]) * 100
        d = m - h
        if d > 1.5:
            wins += 1
        print(f"  {r['attack']:12s}  MoE={m:5.1f}%  HiDDeN={h:5.1f}%  Δ={d:+.1f}pp")
    print(f"  MoE wins (>1.5pp): {wins}/9")
