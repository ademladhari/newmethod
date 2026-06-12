from pathlib import Path
import csv
import json

base = Path(r"D:/new method/results/experiments/unfrozen_moe")
print("=== All unfrozen_moe runs with validation.csv ===")
for val in sorted(base.rglob("validation.csv")):
    run = val.parent
    rel = run.relative_to(base)
    rows = list(csv.DictReader(open(val)))
    train = run / "train.csv"
    train_rows = list(csv.DictReader(open(train))) if train.exists() else []
    print(f"RUN: {rel}")
    if rows:
        last = rows[-1]
        print(
            f"  epochs={len(rows)} final_ep={last.get('epoch')} "
            f"ber={float(last['bitwise-error'])*100:.3f}% "
            f"max_use={last.get('expert_max_use')} "
            f"load_l1={last.get('train_val_load_l1','?')}"
        )
    if train_rows:
        steps = [int(r["step"]) for r in train_rows if r.get("step")]
        print(f"  train_log_rows={len(train_rows)} max_step={max(steps) if steps else '?'}")
    for cf in run.glob("*.json"):
        try:
            cfg = json.load(open(cf))
            if isinstance(cfg, dict):
                interesting = {k: v for k, v in cfg.items() if any(x in str(k).lower() for x in ["expert", "data", "dir", "batch", "epoch", "coco"])}
                if interesting:
                    print(f"  {cf.name}: {interesting}")
        except Exception:
            pass
    for cf in run.glob("*.txt"):
        txt = cf.read_text(errors="replace")
        if any(x in txt.lower() for x in ["data-dir", "num-experts", "coco", "300", "240", "200k"]):
            print(f"  {cf.name} snippet: {txt[:300].replace(chr(10), ' | ')}")
    print()

print("=== Any 8exp under coco200k or coco300k? ===")
for p in sorted(base.rglob("*")):
    if p.is_dir() and "8exp" in p.name.lower():
        print(p)

print("=== Search all results for 8 experts + large dataset mentions ===")
results = Path(r"D:/new method/results")
for f in sorted(results.rglob("*")):
    if not f.is_file():
        continue
    if f.suffix not in {".txt", ".json", ".csv", ".md", ".log"}:
        continue
    if f.stat().st_size > 2_000_000:
        continue
    try:
        txt = f.read_text(errors="replace").lower()
    except Exception:
        continue
    if ("num-experts" in txt or "num_experts" in txt or '"num_experts"' in txt) and "8" in txt:
        if any(x in txt for x in ["300k", "300000", "240k", "242k", "200k", "coco200", "coco300"]):
            print(f"MATCH: {f.relative_to(results)}")
