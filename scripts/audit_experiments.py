"""
Audit all experiment folders and compare against what's documented in the thesis.
Reads val_metrics.csv from each run to get actual BER.
"""
import csv
from pathlib import Path

def get_val_metrics(run_dir):
    """Return (final_epoch, final_ber_pct, max_use, load_l1) from val CSV."""
    for name in ["val_metrics.csv", "val_router_metrics.csv"]:
        f = run_dir / name
        if f.exists():
            try:
                rows = list(csv.DictReader(open(f)))
                if rows:
                    last = rows[-1]
                    ep = last.get("epoch", "?")
                    # BER
                    ber_key = next((k for k in last if "bit_err" in k.lower() or ("ber" in k.lower() and "val" in k.lower())), None)
                    if not ber_key:
                        ber_key = next((k for k in last if "ber" in k.lower()), None)
                    ber = float(last[ber_key]) * 100 if ber_key and last[ber_key] else None
                    # routing
                    mu_key = next((k for k in last if "max_use" in k.lower()), None)
                    ll_key = next((k for k in last if "load_l1" in k.lower()), None)
                    mu = float(last[mu_key]) if mu_key and last[mu_key] else None
                    ll = float(last[ll_key]) if ll_key and last[ll_key] else None
                    return ep, ber, mu, ll
            except Exception as e:
                return "err", None, None, None
    return None, None, None, None


# ── thesis documented runs (from by_4experts.csv + by_8experts.csv) ──────────
DOCUMENTED = {
    "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
    "sym_t14_unfrozen_continue_ep60_2026-06-01",
    "moe4_balance_v2_100k_ep24_2026-05-30",
    "sym_t14_temp09_bal004warm10_ep20_2026-06-01",
    "moe4_uniform_first_100k_v1_ep19_2026-05-29",
    "routing_isolation_jitter005_adv0_ep40_2026-05-31",
    "unfrozen_4exp_dense_k4_b128_ep20_2026-06-02",
    "sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01",
    "unfrozen_4exp_top2_b128_ep30_2026-06-03",
    "sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04",
    "sym_t14_unfrozen_attack_v1_ep68_2026-06-04",
    "router_diag_v2_4exp_ep17_2026-05-30",
    "moe4_balance_v3_100k_ep26_2026-05-30",
    "moe4_hidden178_frozen_v1_ep88_2026-05-28",
    "moe4_ber_route_v1_ep43_2026-05-29",
    "stabilized_4exp_ep54_2026-05-21",
    "sym_t14_unfrozen_200k_v1_ep16_2026-06-05",
    # 8-exp runs
    "collapse_diag_jitter02_loadpen01_bal06_ep34_2026-05-31",
    "moe8_balance_v2_100k_ep16_2026-05-30",
    "moe8_uniform_first_100k_v1_ep13_2026-05-29",
    "unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02",
    "router_diag_v2_8exp_ep13_2026-05-30",
    "moe8_balance_v3_100k_ep20_2026-05-30",
    "moe8_ber_route_v1_ep33_2026-05-29",
    "hidden178_frozen_anticollapse_v2_ep81_2026-05-28",
    "moe_hidden_ep20_2026-05-18",
    "moe8_top2_ber_retry_ep76_unknown",
    "stabilized_v2_milder_ep113_unknown",
    "unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03",
}

# ── scan ──────────────────────────────────────────────────────────────────────
exp_base = Path("results/experiments")

found = []
missing_docs = []

print("\n=== ALL EXPERIMENT RUNS ON DISK ===")
print(f"{'#':>3}  {'Run folder':<55} {'Ep':>4}  {'BER%':>6}  {'max_use':>8}  {'load_l1':>8}  {'In thesis?'}")
print("-" * 110)

idx = 0
for branch in sorted(p for p in exp_base.iterdir() if p.is_dir()):
    for ds in sorted(p for p in branch.iterdir() if p.is_dir()):
        for ne in sorted(p for p in ds.iterdir() if p.is_dir()):
            for top in sorted(p for p in ne.iterdir() if p.is_dir()):
                for run in sorted(p for p in top.iterdir() if p.is_dir()):
                    if not run.is_dir() or "images" in run.name:
                        continue
                    idx += 1
                    ep, ber, mu, ll = get_val_metrics(run)
                    in_thesis = "YES" if run.name in DOCUMENTED else "*** MISSING ***"
                    if run.name not in DOCUMENTED:
                        missing_docs.append(run)
                    ber_s = f"{ber:.2f}" if ber is not None else "  —"
                    mu_s = f"{mu:.3f}" if mu is not None else "  —"
                    ll_s = f"{ll:.3f}" if ll is not None else "  —"
                    print(f"{idx:>3}  {run.name:<55} {str(ep):>4}  {ber_s:>6}  {mu_s:>8}  {ll_s:>8}  {in_thesis}")
                    found.append(run.name)

print(f"\nTotal runs on disk: {idx}")
print(f"Documented in thesis: {len(DOCUMENTED)}")

if missing_docs:
    print(f"\n*** {len(missing_docs)} RUNS NOT IN THESIS DOCUMENTATION ***")
    for r in missing_docs:
        ep, ber, mu, ll = get_val_metrics(r)
        ber_s = f"{ber:.2f}%" if ber is not None else "unknown"
        print(f"  - {r.name}  (BER={ber_s})")
else:
    print("\nAll disk runs are documented.")

# Check documented runs missing from disk
on_disk = set(found)
not_on_disk = [d for d in DOCUMENTED if d not in on_disk]
if not_on_disk:
    print(f"\n*** {len(not_on_disk)} DOCUMENTED RUNS NOT ON DISK ***")
    for r in not_on_disk:
        print(f"  - {r}")
