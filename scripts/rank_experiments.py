"""Rank installed MoE experiments by val BER + routing stability."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "experiments"


def last_row(csv_path: Path) -> dict | None:
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def f(row: dict, key: str, default=float("nan")):
    for k, v in row.items():
        if k.strip() == key.strip():
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
    return default


def main():
    records = []
    for train_csv in sorted(ROOT.rglob("train.csv")):
        run_dir = train_csv.parent
        rel = run_dir.relative_to(ROOT)
        branch = rel.parts[0]
        val = last_row(run_dir / "validation.csv")
        valn = last_row(run_dir / "validation_noisy.csv")
        if not val:
            continue
        ep = int(f(val, "epoch", 0))
        records.append({
            "path": str(rel).replace("\\", "/"),
            "branch": branch,
            "name": run_dir.name,
            "epoch": ep,
            "val_ber": f(val, "bitwise-error"),
            "val_noisy_ber": f(valn, "bitwise-error") if valn else float("nan"),
            "expert_max_use": f(val, "expert_max_use"),
            "train_val_load_l1": f(val, "train_val_load_l1"),
            "routing_repeat": f(val, "routing_repeat_match"),
            "effective_experts": f(val, "effective_experts"),
            "has_noisy": valn is not None,
        })

    # Score: lower is better for all
    def stability_score(r):
        mx = r["expert_max_use"]
        l1 = r["train_val_load_l1"]
        if mx != mx or l1 != l1:  # nan
            return 999.0
        return 0.6 * mx + 0.4 * l1

    def accuracy_score(r):
        ber = r["val_noisy_ber"] if r["has_noisy"] and r["val_noisy_ber"] == r["val_noisy_ber"] else r["val_ber"]
        return ber

    print("=== FINAL EPOCH: validation (clean) ===\n")
    print(f"{'branch':14} {'val_ber':>8} {'max_use':>8} {'load_l1':>8} {'eff_exp':>8}  run")
    for r in sorted(records, key=lambda x: (x["val_ber"], stability_score(x))):
        print(
            f"{r['branch']:14} {r['val_ber']:8.4f} {r['expert_max_use']:8.3f} "
            f"{r['train_val_load_l1']:8.3f} {r['effective_experts']:8.2f}  {r['name'][:55]}"
        )

    print("\n=== With noisy val (thesis-relevant) ===\n")
    noisy = [r for r in records if r["has_noisy"]]
    print(f"{'branch':14} {'clean':>8} {'noisy':>8} {'max_use':>8} {'load_l1':>8}  run")
    for r in sorted(noisy, key=lambda x: (x["val_noisy_ber"], stability_score(x))):
        print(
            f"{r['branch']:14} {r['val_ber']:8.4f} {r['val_noisy_ber']:8.4f} "
            f"{r['expert_max_use']:8.3f} {r['train_val_load_l1']:8.3f}  {r['name'][:50]}"
        )

    # Pareto-ish pick: gates expert_max_use < 0.5, load_l1 < 0.2
    gated = [r for r in noisy if r["expert_max_use"] < 0.5 and r["train_val_load_l1"] < 0.2]
    print("\n=== Pass routing gates (max_use<0.5, load_l1<0.2) + has noisy val ===\n")
    if not gated:
        print("(none)")
    else:
        for r in sorted(gated, key=lambda x: (x["val_noisy_ber"], x["val_ber"])):
            print(f"  {r['branch']:14} clean={r['val_ber']:.4f} noisy={r['val_noisy_ber']:.4f} "
                  f"max_use={r['expert_max_use']:.3f} l1={r['train_val_load_l1']:.3f}")
            print(f"    {r['path']}")

    # Combined rank
    print("\n=== Combined rank (noisy BER + 2*stability penalty) ===\n")
    for r in sorted(noisy, key=lambda x: accuracy_score(x) + 2.0 * stability_score(x)):
        comb = accuracy_score(r) + 2.0 * stability_score(r)
        print(f"  score={comb:.4f}  noisy_ber={r['val_noisy_ber']:.4f}  stab={stability_score(r):.3f}  {r['name'][:52]}")


if __name__ == "__main__":
    main()
