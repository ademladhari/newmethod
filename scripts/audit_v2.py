"""Full audit: read actual validation metrics and compare to thesis documentation."""
import csv
from pathlib import Path

base = Path("results/experiments")

KEY_BER   = "bitwise-error"
KEY_MU    = "expert_max_use"
KEY_LL    = "train_val_load_l1"

# What the thesis says for each run (clean BER % at report epoch)
THESIS_BER = {
    "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01":       0.32,
    "sym_t14_unfrozen_continue_ep60_2026-06-01":           4.01,
    "sym_t14_unfrozen_attack_v1_ep68_2026-06-04":          0.44,  # in progress, ep65 was last noted
    "sym_t14_unfrozen_200k_v1_ep16_2026-06-05":            0.21,  # in progress ep11 best
    "sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01": 2.72,
    "sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04": 7.04,
    "routing_isolation_jitter005_adv0_ep40_2026-05-31":    0.98,
    "sym_t14_temp09_bal004warm10_ep20_2026-06-01":         0.96,
    "unfrozen_4exp_top2_b128_ep30_2026-06-03":             2.66,
    "unfrozen_4exp_dense_k4_b128_ep20_2026-06-02":         2.62,
    "unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02":        1.70,
    "unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03":        1.36,
    "collapse_diag_jitter02_loadpen01_bal06_ep34_2026-05-31": 0.44,
}

rows_out = []

for val_csv in sorted(base.rglob("validation.csv")):
    if "images" in str(val_csv):
        continue
    parts = val_csv.relative_to(base).parts
    branch = parts[0]; ds = parts[1]; ne = parts[2]; batch = parts[3]; run = parts[4]
    
    try:
        data = list(csv.DictReader(open(val_csv)))
        if not data:
            continue
        last = data[-1]
        ep = int(last["epoch"])
        ber = float(last[KEY_BER]) * 100
        mu  = float(last[KEY_MU])
        ll_val = last.get(KEY_LL, "")
        ll  = float(ll_val) if ll_val else None
        
        # best BER row
        best_row = min(data, key=lambda r: float(r[KEY_BER]))
        best_ep  = int(best_row["epoch"])
        best_ber = float(best_row[KEY_BER]) * 100
        
        thesis = THESIS_BER.get(run)
        mismatch = ""
        if thesis is not None:
            diff = abs(ber - thesis)
            if diff > 1.0:
                mismatch = f"  !! thesis={thesis:.2f}"
        
        rows_out.append((branch, ds, ne, batch, run, ep, ber, mu, ll, best_ep, best_ber, mismatch))
    except Exception as e:
        rows_out.append((branch, ds, ne, batch, run, "ERR", None, None, None, None, None, str(e)[:50]))

print(f"\n{'Branch':<16} {'Dataset':<12} {'N/k':<10} {'Batch':<9} {'Run':<45} {'ep':>4} {'BER%':>6} {'mu':>6} {'ll':>6} {'bestEp':>6} {'bestBER':>8}  note")
print("-"*160)
for (branch, ds, ne, batch, run, ep, ber, mu, ll, bep, bber, note) in rows_out:
    ber_s   = f"{ber:.2f}" if ber is not None else "  —"
    mu_s    = f"{mu:.3f}" if mu is not None else "  —"
    ll_s    = f"{ll:.3f}" if ll is not None else "  —"
    bber_s  = f"{bber:.2f}" if bber is not None else "  —"
    run_s   = run[:44]
    print(f"{branch:<16} {ds:<12} {ne:<10} {batch:<9} {run_s:<45} {str(ep):>4} {ber_s:>6} {mu_s:>6} {ll_s:>6} {str(bep):>6} {bber_s:>8}  {note}")

print(f"\nTotal runs: {len(rows_out)}")

# Runs NOT in thesis BER dict (undocumented metrics)
nodoc = [(r[4], r[5], r[6]) for r in rows_out if r[4] not in THESIS_BER]
if nodoc:
    print(f"\n=== {len(nodoc)} runs with no thesis BER reference (legacy/undocumented) ===")
    for name, ep, ber in nodoc:
        ber_s = f"{ber:.2f}%" if ber is not None else "—"
        print(f"  {name}  ep={ep} BER={ber_s}")
