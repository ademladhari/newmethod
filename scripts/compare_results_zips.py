"""Compare results/*.zip runs: BER + routing stability."""
import csv
import zipfile
from io import TextIOWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE_DIR = (
    RESULTS
    / "experiments/unfrozen_moe/coco100k/4exp_k1/batch128"
    / "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01"
)

RUNS = [
    {
        "label": "MAIN (sym_t14 unfrozen)",
        "source": "installed",
        "config": "4e, k=1, b128, ep20",
        "zip": None,
        "val": BASELINE_DIR / "validation.csv",
        "noisy": BASELINE_DIR / "validation_noisy.csv",
        "report_ep": 20,
    },
    {
        "label": "Continue (final ep60)",
        "zip": RESULTS / "exp4continue.zip",
        "val": "moe_unfrozen_sym_t14_v1_continue/validation.csv",
        "noisy": "moe_unfrozen_sym_t14_v1_continue/validation_noisy.csv",
        "config": "4e, k=1, b128, ep21-60",
        "report_ep": 60,
    },
    {
        "label": "Continue (ep20 checkpoint)",
        "zip": RESULTS / "exp4continue.zip",
        "val": "moe_unfrozen_sym_t14_v1_continue/validation.csv",
        "noisy": "moe_unfrozen_sym_t14_v1_continue/validation_noisy.csv",
        "config": "4e, k=1, b128",
        "report_ep": 20,
    },
    {
        "label": "8-exp sparse",
        "zip": RESULTS / "moe_unfrozen_8exp_sparse_b128 2026.06.02--21-36-20.zip",
        "val": "moe_unfrozen_8exp_sparse_b128 2026.06.02--21-36-20/validation.csv",
        "noisy": "moe_unfrozen_8exp_sparse_b128 2026.06.02--21-36-20/validation_noisy.csv",
        "config": "8e, k=1, b128, ep20",
        "report_ep": 20,
    },
    {
        "label": "4-exp dense (k=4)",
        "zip": RESULTS / "moe_unfrozen_4exp_sparse_b128_run.zip",
        "val": "moe_unfrozen_4exp_sparse_b128 2026.06.02--21-14-20/validation.csv",
        "noisy": "moe_unfrozen_4exp_sparse_b128 2026.06.02--21-14-20/validation_noisy.csv",
        "config": "4e, k=4, b128, ep20",
        "report_ep": 20,
    },
    {
        "label": "4-exp top-2 (30ep)",
        "zip": RESULTS / "moe_collapse_diag_v1 2026.05.31--08-55-29.zip",
        "val": "newmethod/hidden_moe_unfrozen/runs/moe_unfrozen_4exp_sparse_top2_b128 2026.06.03--08-01-13/validation.csv",
        "noisy": "newmethod/hidden_moe_unfrozen/runs/moe_unfrozen_4exp_sparse_top2_b128 2026.06.03--08-01-13/validation_noisy.csv",
        "config": "4e, k=2, b128, ep30",
        "report_ep": 30,
    },
    {
        "label": "4-exp top-2 @ep20",
        "zip": RESULTS / "moe_collapse_diag_v1 2026.05.31--08-55-29.zip",
        "val": "newmethod/hidden_moe_unfrozen/runs/moe_unfrozen_4exp_sparse_top2_b128 2026.06.03--08-01-13/validation.csv",
        "noisy": "newmethod/hidden_moe_unfrozen/runs/moe_unfrozen_4exp_sparse_top2_b128 2026.06.03--08-01-13/validation_noisy.csv",
        "config": "4e, k=2, b128",
        "report_ep": 20,
    },
]


def load_csv(path_or_z, member=None):
    if member is None:
        with open(path_or_z, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with zipfile.ZipFile(path_or_z) as z:
        with z.open(member) as f:
            return list(csv.DictReader(TextIOWrapper(f, encoding="utf-8")))


def g(row, *keys):
    for k in keys:
        for c in (k, k + " ", k.strip()):
            if c in row and row[c] not in ("", None):
                return float(row[c])
    return None


def row_at(rows, ep):
    return next(r for r in rows if int(r["epoch"]) == ep)


def metrics(val_rows, noisy_rows, ep, num_experts=4):
    v = row_at(val_rows, ep)
    n = row_at(noisy_rows, ep) if noisy_rows else None
    max_use = g(v, "expert_max_use")
    l1 = g(v, "train_val_load_l1")
    eff = g(v, "effective_experts")
    collapse = max_use * num_experts if max_use is not None else None
    # stability score: lower is better (heuristic)
    stab = 0.0
    if l1 is not None:
        stab += min(l1 / 0.20, 2.0)  # 0.20 threshold
    if collapse is not None:
        stab += abs(collapse - 1.0)  # 1.0 = balanced
    return {
        "ep": ep,
        "clean_ber": g(v, "bitwise-error"),
        "noisy_ber": g(n, "bitwise-error") if n else None,
        "max_use": max_use,
        "load_l1": l1,
        "eff_exp": eff,
        "collapse": collapse,
        "stab_score": stab,
    }


def stability_label(m, num_experts=4):
    l1, col, ber = m["load_l1"], m["collapse"], m["clean_ber"]
    if ber is None:
        return "?"
    if l1 is not None and l1 <= 0.10 and col is not None and col <= 1.35:
        return "Excellent"
    if l1 is not None and l1 <= 0.20 and col is not None and col <= 1.60:
        return "Good"
    if l1 is not None and l1 <= 0.35 and col is not None and col <= 2.0:
        return "Fair"
    return "Poor"


def main():
    print("BER & ROUTING STABILITY COMPARISON")
    print("Baseline: MAIN sym_t14 @ ep20 — clean 0.32%, collapse score 1.22, load_l1 0.08")
    print("Collapse score = expert_max_use × N  (1.0 = balanced for sparse routing)\n")

    rows_out = []
    for spec in RUNS:
        ne = 8 if "8e" in spec["config"] or "8-exp" in spec["label"] else 4
        if spec.get("source") == "installed":
            val = load_csv(spec["val"])
            noisy = load_csv(spec["noisy"])
        else:
            val = load_csv(spec["zip"], spec["val"])
            noisy = load_csv(spec["zip"], spec["noisy"]) if spec["noisy"] in zipfile.ZipFile(spec["zip"]).namelist() else []
        ep = spec["report_ep"]
        m = metrics(val, noisy, ep, ne)
        m["label"] = spec["label"]
        m["config"] = spec["config"]
        m["stab_label"] = stability_label(m, ne)
        rows_out.append(m)

    hdr = f"{'Run':<28} {'Config':<22} {'ep':>3} {'Clean%':>7} {'Noisy%':>7} {'max_use':>8} {'load_l1':>8} {'Collap':>7} {'EffExp':>6} {'Stability':>10}"
    print(hdr)
    print("-" * len(hdr))
    for m in rows_out:
        print(
            f"{m['label']:<28} {m['config']:<22} {m['ep']:>3} "
            f"{m['clean_ber']*100:>7.3f} {(m['noisy_ber'] or 0)*100:>7.3f} "
            f"{m['max_use']:>8.3f} {m['load_l1']:>8.3f} {m['collapse']:>7.2f} "
            f"{m['eff_exp']:>6.2f} {m['stab_label']:>10}"
        )

    print("\n--- Ranking (clean BER, lower better) ---")
    ranked = sorted(rows_out, key=lambda x: x["clean_ber"])
    for i, m in enumerate(ranked, 1):
        print(f"  {i}. {m['label']} @ ep{m['ep']}: {m['clean_ber']*100:.3f}% clean, stability={m['stab_label']}")

    print("\n--- Ranking (routing stability: load_l1, lower better) ---")
    with_l1 = [m for m in rows_out if m["load_l1"] is not None]
    ranked2 = sorted(with_l1, key=lambda x: (x["load_l1"], x["collapse"] or 99))
    for i, m in enumerate(ranked2, 1):
        print(f"  {i}. {m['label']}: load_l1={m['load_l1']:.3f}, collapse={m['collapse']:.2f}")


if __name__ == "__main__":
    main()
