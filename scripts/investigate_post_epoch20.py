"""Investigate why exp4continue degrades after epoch 20."""
import csv
import math
import zipfile
from io import TextIOWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "results" / "exp4continue.zip"
PREFIX = "moe_unfrozen_sym_t14_v1_continue/"


def load(z, name):
    with z.open(PREFIX + name) as f:
        return list(csv.DictReader(TextIOWrapper(f, encoding="utf-8")))


def f(row, col):
  for k in (col, col.strip(), col.strip() + " "):
    if k in row:
      return float(row[k])
  raise KeyError(col)


def max_use(row):
  return f(row, "expert_max_use")


def attack_nonidentity_frac(epoch, max_ep=60):
    """Matches moe_noise.set_training_schedule (noisy val uses these weights)."""
    if epoch <= 20:
        return 0.0
    if epoch >= 80:
        return 1.0
    return min(1.0, max(0.0, (epoch - 20) / 60.0))


def router_temp(epoch, start=1.4, end=1.0, max_ep=60):
    progress = max(0.0, min(1.0, (epoch - 1) / float(max_ep)))
    return start + (end - start) * progress


def main():
    z = zipfile.ZipFile(ZIP)
    train = load(z, "train.csv")
    val = load(z, "validation.csv")
    noisy = load(z, "validation_noisy.csv")

    print("=" * 70)
    print("1) TRAIN vs CLEAN VAL BER (overfitting signal)")
    print("=" * 70)
    print(f"{'ep':>4} {'train_BER%':>10} {'clean_BER%':>10} {'gap':>8} {'train_max':>10} {'val_max':>10} {'load_l1':>8}")
    for ep in list(range(18, 23)) + [26, 30, 31, 40, 50, 60]:
        tr = next(r for r in train if int(r["epoch"]) == ep)
        va = next(r for r in val if int(r["epoch"]) == ep)
        tber, vber = f(tr, "bitwise-error"), f(va, "bitwise-error")
        tmu = max_use(tr)
        vmu = max_use(va)
        l1 = va.get("train_val_load_l1", "")
        l1f = float(l1) if l1 else float("nan")
        print(
            f"{ep:4d} {tber*100:10.3f} {vber*100:10.3f} {(tber-vber)*100:8.3f} "
            f"{tmu:10.3f} {vmu:10.3f} {l1f:8.3f}"
        )

    print("\n" + "=" * 70)
    print("2) NOISY VAL vs attack mix strength (metric artifact)")
    print("=" * 70)
    print(f"{'ep':>4} {'noisy_BER%':>11} {'non-id_frac':>12} {'clean_BER%':>11}")
    for ep in range(18, 61, 2):
        va = next(r for r in val if int(r["epoch"]) == ep)
        no = next(r for r in noisy if int(r["epoch"]) == ep)
        frac = attack_nonidentity_frac(ep)
        print(
            f"{ep:4d} {f(no,'bitwise-error')*100:11.3f} {frac:12.2f} "
            f"{f(va,'bitwise-error')*100:11.3f}"
        )

    # correlation noisy BER vs attack frac
    pairs = []
    for r in noisy:
        ep = int(r["epoch"])
        if ep >= 21:
            pairs.append((attack_nonidentity_frac(ep), f(r, "bitwise-error")))
    if len(pairs) > 2:
        xs, ys = zip(*pairs)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        r = num / den if den else 0
        print(f"\nPearson r(noisy_BER, attack_mix_frac) epochs 21-60: {r:.3f}")

    print("\n" + "=" * 70)
    print("3) CLEAN VAL BER trend epochs 20-60")
    print("=" * 70)
    ep20 = f(next(r for r in val if int(r["epoch"]) == 20), "bitwise-error")
    worse = sum(1 for r in val if int(r["epoch"]) > 20 and f(r, "bitwise-error") > ep20)
    better = sum(1 for r in val if int(r["epoch"]) > 20 and f(r, "bitwise-error") < ep20)
    total = sum(1 for r in val if int(r["epoch"]) > 20)
    print(f"Epochs 21-60: {better} better than ep20 clean, {worse} worse, on {total} epochs")
    print(f"Mean clean BER 21-60: {100*sum(f(r,'bitwise-error') for r in val if int(r['epoch'])>20)/total:.2f}%")
    print(f"Median clean BER 21-60: {100*sorted(f(r,'bitwise-error') for r in val if int(r['epoch'])>20)[total//2]:.2f}%")

    print("\n" + "=" * 70)
    print("4) Discriminator / adversarial on val (instability)")
    print("=" * 70)
    for ep in [20, 21, 30, 40, 60]:
        va = next(r for r in val if int(r["epoch"]) == ep)
        print(
            f"ep {ep}: adv_bce={f(va,'adversarial_bce'):.2f} "
            f"discr_cover={f(va,'discr_cover_bce'):.3f} discr_enc={f(va,'discr_encod_bce'):.4f}"
        )

    print("\n" + "=" * 70)
    print("5) Router temperature (continues annealing 21-60)")
    print("=" * 70)
    for ep in [20, 30, 40, 50, 60]:
        print(f"ep {ep}: T={router_temp(ep):.3f}")

    print("\n" + "=" * 70)
    print("6) Train BER still falling while val worsens (classic overfit)")
    print("=" * 70)
    tr20 = f(next(r for r in train if int(r["epoch"]) == 20), "bitwise-error")
    tr60 = f(next(r for r in train if int(r["epoch"]) == 60), "bitwise-error")
    v20 = ep20
    v60 = f(next(r for r in val if int(r["epoch"]) == 60), "bitwise-error")
    print(f"Train BER  ep20={tr20*100:.3f}% -> ep60={tr60*100:.3f}%")
    print(f"Clean val  ep20={v20*100:.3f}% -> ep60={v60*100:.3f}%")


if __name__ == "__main__":
    main()
