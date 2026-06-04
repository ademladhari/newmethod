"""Audit zip runs in results/ root — good vs bad vs baseline."""
import csv
import zipfile
from io import TextIOWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
BASELINE_BER = 0.0032  # moe_unfrozen_sym_t14_v1 ep20 clean
BASELINE_NOISY = 0.0031
BASELINE_LOAD_L1 = 0.0824
BASELINE_MAX_USE = 0.3055

ZIPS = sorted(ROOT.glob("*.zip"))


def ber(row):
    return float(row["bitwise-error"])


def max_use(row):
    for k in ("expert_max_use ", "expert_max_use"):
        if k in row:
            return float(row[k])
    return float("nan")


def load_l1(row):
    v = row.get("train_val_load_l1", "")
    return float(v) if v not in (None, "") else None


def last_epoch(rows):
    return max(int(r["epoch"]) for r in rows)


def find_run_csv(z: zipfile.ZipFile, suffix: str):
    hits = [n for n in z.namelist() if n.endswith(suffix) and "checkpoints" not in n]
    if not hits:
        return None
  # prefer shortest path (run root)
    hits.sort(key=len)
    return hits[0]


def audit_zip(zpath: Path):
    out = {
        "zip": zpath.name,
        "size_mb": round(zpath.stat().st_size / 1e6, 1),
        "run_folder": None,
        "has_train": False,
        "has_val": False,
        "epochs": 0,
        "final_ep": 0,
        "clean_ber": None,
        "noisy_ber": None,
        "max_use": None,
        "load_l1": None,
        "best_clean_ep": None,
        "best_clean_ber": None,
        "checkpoints": [],
        "experiment_name": None,
        "error": None,
    }
    try:
        with zipfile.ZipFile(zpath) as z:
            out["checkpoints"] = sorted(
                [Path(n).name for n in z.namelist() if n.endswith(".pyt")]
            )
            val_name = find_run_csv(z, "validation.csv")
            train_name = find_run_csv(z, "train.csv")
            if val_name:
                out["run_folder"] = val_name.split("/")[0] if "/" in val_name else val_name
                rows = list(csv.DictReader(TextIOWrapper(z.open(val_name), encoding="utf-8")))
                out["has_val"] = True
                out["epochs"] = len(rows)
                out["final_ep"] = last_epoch(rows)
                final = next(r for r in rows if int(r["epoch"]) == out["final_ep"])
                out["clean_ber"] = ber(final)
                out["max_use"] = max_use(final)
                out["load_l1"] = load_l1(final)
                best = min(rows, key=ber)
                out["best_clean_ep"] = int(best["epoch"])
                out["best_clean_ber"] = ber(best)
            if train_name:
                out["has_train"] = True
            noisy_name = find_run_csv(z, "validation_noisy.csv")
            if noisy_name:
                nrow = list(csv.DictReader(TextIOWrapper(z.open(noisy_name), encoding="utf-8")))
                if nrow:
                    fe = last_epoch(nrow)
                    final_n = next(r for r in nrow if int(r["epoch"]) == fe)
                    out["noisy_ber"] = ber(final_n)
            # guess name from folder
            if out["run_folder"]:
                out["experiment_name"] = out["run_folder"].split()[0].split("/")[0]
    except Exception as e:
        out["error"] = str(e)
    return out


def verdict(a):
    if a.get("error"):
        return "BROKEN", a["error"]
    if not a["has_val"]:
        return "INCOMPLETE", "no validation.csv"
    cb = a["clean_ber"]
    l1 = a["load_l1"]
    if cb is None:
        return "INCOMPLETE", "no BER"
    tags = []
    if cb < BASELINE_BER - 1e-5:
        tags.append("clean_BER_beats_baseline")
    elif cb <= BASELINE_BER * 1.1:
        tags.append("clean_BER_near_baseline")
    else:
        tags.append("clean_BER_worse")
    if l1 is not None:
        if l1 <= 0.20:
            tags.append("routing_ok")
        else:
            tags.append("routing_gap_high")
    if a["noisy_ber"] is not None and a["noisy_ber"] > 0.05:
        tags.append("noisy_val_high")
    # overall tier
    if "clean_BER_beats_baseline" in tags and (l1 is None or l1 <= 0.20):
        tier = "GOOD"
    elif cb <= 0.02 and (l1 is None or l1 <= 0.25):
        tier = "OK"
    elif cb <= 0.05:
        tier = "MARGINAL"
    else:
        tier = "BAD"
    return tier, "; ".join(tags)


def main():
    print("Baseline (moe_unfrozen_sym_t14_v1 @ ep20): clean BER {:.4f}%, noisy {:.4f}%, load_l1 {:.4f}\n".format(
        BASELINE_BER * 100, BASELINE_NOISY * 100, BASELINE_LOAD_L1
    ))
    rows = [audit_zip(z) for z in ZIPS]
    for a in rows:
        tier, detail = verdict(a)
        print("=" * 72)
        print(f"ZIP: {a['zip']} ({a['size_mb']} MB)")
        print(f"  Tier: {tier}  |  {detail}")
        if a["run_folder"]:
            print(f"  Run folder: {a['run_folder']}")
        if a["clean_ber"] is not None:
            print(
                f"  Final ep {a['final_ep']}: clean {a['clean_ber']*100:.3f}%"
                f" | noisy {(a['noisy_ber'] or 0)*100:.3f}%"
                f" | max_use {a['max_use']:.3f} | load_l1 {a['load_l1']}"
            )
            print(
                f"  Best clean: ep {a['best_clean_ep']} = {a['best_clean_ber']*100:.3f}%"
            )
        print(f"  Checkpoints: {len(a['checkpoints'])} — {', '.join(a['checkpoints'][:6])}")
        if len(a["checkpoints"]) > 6:
            print(f"    ... +{len(a['checkpoints'])-6} more")


if __name__ == "__main__":
    main()
