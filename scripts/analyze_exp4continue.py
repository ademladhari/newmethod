"""Compare exp4continue.zip validation curves vs original epoch-20 baseline."""
import csv
import zipfile
from io import TextIOWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "results" / "exp4continue.zip"
BASELINE_VAL = (
    ROOT
    / "results/experiments/unfrozen_moe/coco100k/4exp_k1/batch128"
    / "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/validation.csv"
)
BASELINE_NOISY = BASELINE_VAL.parent / "validation_noisy.csv"


def load_csv_from_zip(z, name):
    with z.open(name) as f:
        return list(csv.DictReader(TextIOWrapper(f, encoding="utf-8")))


def load_csv_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ber(row):
    return float(row["bitwise-error"])


def max_use(row):
    return float(row.get("expert_max_use ", row.get("expert_max_use", 0)))


def load_l1(row):
    v = row.get("train_val_load_l1")
    return float(v) if v not in (None, "") else None


def main():
    z = zipfile.ZipFile(ZIP_PATH)
    prefix = "moe_unfrozen_sym_t14_v1_continue/"
    val = load_csv_from_zip(z, prefix + "validation.csv")
    noisy = load_csv_from_zip(z, prefix + "validation_noisy.csv")

    base_val = load_csv_file(BASELINE_VAL)
    base_noisy = load_csv_file(BASELINE_NOISY)
    ep20 = next(r for r in base_val if int(r["epoch"]) == 20)
    ep20n = next(r for r in base_noisy if int(r["epoch"]) == 20)

    b_ber, b_mu, b_l1 = ber(ep20), max_use(ep20), load_l1(ep20)
    b_nber = ber(ep20n)

    print("=" * 60)
    print("BASELINE (original sym_t14, epoch 20)")
    print(f"  clean BER:         {b_ber * 100:.4f}%")
    print(f"  noisy BER:         {b_nber * 100:.4f}%")
    print(f"  expert_max_use:    {b_mu:.4f}")
    print(f"  train_val_load_l1: {b_l1:.4f}")

    ep20c = next(r for r in val if int(r["epoch"]) == 20)
    print("\nCONTINUE zip at epoch 20 (resume point)")
    print(f"  clean BER:         {ber(ep20c) * 100:.4f}%")

    best = min(val, key=ber)
    print("\nBEST clean BER in continue run (epochs 1-60)")
    print(f"  epoch {best['epoch']}: {ber(best) * 100:.4f}%  max_use {max_use(best):.4f}")

    better = sorted([r for r in val if ber(r) < b_ber - 1e-12], key=ber)
    print(f"\nEpochs with clean BER STRICTLY better than baseline ep20 ({b_ber*100:.4f}%):")
    if not better:
        print("  >>> NONE — epoch 20 remains best for clean BER")
    else:
        for r in better:
            ep = int(r["epoch"])
            print(
                f"  epoch {ep:3d}: BER {ber(r)*100:.4f}%  "
                f"max_use {max_use(r):.4f}  load_l1 {load_l1(r)}"
            )

    better_n = sorted([r for r in noisy if ber(r) < b_nber - 1e-12], key=ber)
    print(f"\nEpochs with noisy BER better than baseline ep20 ({b_nber*100:.4f}%):")
    if not better_n:
        print("  >>> NONE")
    else:
        for r in better_n[:20]:
            print(f"  epoch {int(r['epoch']):3d}: {ber(r)*100:.4f}%")

    print("\n--- All epochs 21-60 (clean BER) ---")
    for r in val:
        ep = int(r["epoch"])
        if ep <= 20:
            continue
        b = ber(r)
        mark = " *** BETTER" if b < b_ber else ""
        print(f"  {ep:3d}: {b*100:.4f}%  max_use {max_use(r):.4f}{mark}")

    # Within 10% of baseline but not better
    near = [r for r in val if int(r["epoch"]) > 20 and ber(r) <= b_ber * 1.1]
    if near:
        print("\nEpochs 21+ within 10% of ep20 BER (still worse if > baseline):")
        for r in sorted(near, key=ber)[:5]:
            ep = int(r["epoch"])
            print(f"  epoch {ep}: {ber(r)*100:.4f}%")

    print("\n--- Key epochs detail (clean / noisy / routing) ---")
    keys = [20, 26, 31, 36, 37, 44, 47, 50, 55, 60]
    for ep in keys:
        v = next(r for r in val if int(r["epoch"]) == ep)
        n = next(r for r in noisy if int(r["epoch"]) == ep)
        l1 = load_l1(v)
        print(
            f"  ep {ep:2d}: clean {ber(v)*100:.3f}%  noisy {ber(n)*100:.3f}%  "
            f"max_use {max_use(v):.3f}  load_l1 {l1}"
        )


if __name__ == "__main__":
    main()
