"""Table: best val epoch vs best SAVED checkpoint epoch per run."""
import csv
import re
from pathlib import Path

EXP = Path("results/experiments")
KEY_BER, KEY_MU, KEY_LL, KEY_EFF = "bitwise-error", "expert_max_use", "train_val_load_l1", "effective_experts"


def ffloat(v):
    try:
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def n_exp(parts):
    for p in parts:
        if m := re.match(r"(\d+)exp", p):
            return int(m.group(1))
    return 4


def stability(mu, ll, eff, n):
    if mu is None:
        return None
    c = max(0.0, (mu - 1 / n) / (1 - 1 / n + 1e-9))
    s = 1.0 - 0.5 * min(1.0, c)
    if ll is not None:
        s -= 0.3 * min(1.0, ll / 1.5)
    if eff is not None:
        s += 0.2 * min(1.0, eff / n)
    return max(0.0, min(1.0, s))


def composite(ber_pct, stab):
    if ber_pct is None or stab is None:
        return None
    return 0.6 * max(0, 1 - ber_pct / 25) + 0.4 * stab


def label(run, branch):
    m = [
        ("sym_t14_unfrozen_bal004warm10", "R0 PRIMARY"),
        ("sym_t14_unfrozen_continue", "R0 continuation"),
        ("sym_t14_unfrozen_attack", "Attack-trained"),
        ("sym_t14_unfrozen_200k", "200k data scale"),
        ("unfrozen_4exp_top2", "k=2 ablation"),
        ("unfrozen_4exp_dense_k4", "k=4 dense"),
        ("unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02", "8exp run A"),
        ("unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03", "8exp run B"),
        ("sym_bal08_jitter0_bal08warm5_temp14to10_ep20", "Frozen bal08 ep20"),
        ("sym_bal08_jitter0_bal08warm5_temp14to10_ep30", "Frozen bal08 ep30"),
        ("routing_isolation", "Routing isolation"),
        ("collapse_diag", "8exp collapse diag"),
        ("sym_t14_temp09", "Frozen t14 temp0.9"),
    ]
    for k, v in m:
        if k in run:
            return v
    return "Legacy" if branch == "legacy_early" else run[:32]


def metrics_at(rows, ep):
    for r in rows:
        if int(r["epoch"]) == ep:
            ber = ffloat(r[KEY_BER]) * 100
            mu = ffloat(r[KEY_MU])
            ll = ffloat(r.get(KEY_LL))
            return ber, mu, ll
    return None, None, None


def best_epoch(rows, n, saved_only=False, saved_eps=None, mode="composite"):
    cands = []
    for r in rows:
        ep = int(r["epoch"])
        if saved_only and ep not in saved_eps:
            continue
        ber = ffloat(r[KEY_BER])
        if ber is None:
            continue
        ber_pct = ber * 100
        mu = ffloat(r[KEY_MU])
        ll = ffloat(r.get(KEY_LL))
        eff = ffloat(r.get(KEY_EFF))
        stab = stability(mu, ll, eff, n)
        comp = composite(ber_pct, stab)
        cands.append((ep, ber_pct, mu, ll, stab, comp))
    if not cands:
        return None
    if mode == "ber":
        return min(cands, key=lambda x: x[1])
    return max(cands, key=lambda x: x[5] or 0)


rows_out = []
for val_csv in sorted(EXP.rglob("validation.csv")):
    if "images" in str(val_csv):
        continue
    run_dir = val_csv.parent
    parts = run_dir.relative_to(EXP).parts
    branch, run_name = parts[0], parts[4]
    n = n_exp(parts)
    data = list(csv.DictReader(open(val_csv, encoding="utf-8")))
    if not data:
        continue

    saved = set()
    chk = run_dir / "checkpoints"
    if chk.is_dir():
        for f in chk.glob("*.pyt"):
            if m := re.search(r"epoch-(\d+)", f.name):
                saved.add(int(m.group(1)))

    b_all = best_epoch(data, n, saved_only=False, mode="composite")
    b_ber = best_epoch(data, n, saved_only=False, mode="ber")
    b_saved = best_epoch(data, n, saved_only=True, saved_eps=saved, mode="composite") if saved else None
    b_saved_ber = best_epoch(data, n, saved_only=True, saved_eps=saved, mode="ber") if saved else None

  # use composite for "best"; also show if pure-BER best differs
    if not b_all:
        continue

    same = b_saved and b_all[0] == b_saved[0]
    rows_out.append({
        "label": label(run_name, branch),
        "run_folder": run_name,
        "best_ep": b_all[0],
        "best_ber": round(b_all[1], 3),
        "best_max_use": round(b_all[2], 4) if b_all[2] else "",
        "best_load_l1": round(b_all[3], 4) if b_all[3] else "",
        "best_ber_only_ep": b_ber[0],
        "best_ber_only_pct": round(b_ber[1], 3),
        "saved_ep": b_saved[0] if b_saved else "—",
        "saved_ber": round(b_saved[1], 3) if b_saved else "—",
        "saved_max_use": round(b_saved[2], 4) if b_saved and b_saved[2] else "—",
        "saved_load_l1": round(b_saved[3], 4) if b_saved and b_saved[3] else "—",
        "saved_epochs_list": ",".join(str(e) for e in sorted(saved)) if saved else "none",
        "match": "yes" if same else "NO",
    })

# sort thesis first
priority = ["R0 PRIMARY", "R0 continuation", "Attack-trained", "200k data scale", "k=2 ablation", "k=4 dense", "8exp run A", "8exp run B", "Frozen bal08 ep20", "Frozen bal08 ep30", "Routing isolation", "8exp collapse diag", "Frozen t14 temp0.9"]
def sort_key(r):
    if r["label"] in priority:
        return (0, priority.index(r["label"]))
    return (1, r["best_ber"])

rows_out.sort(key=sort_key)

out = Path("results/BEST_VS_SAVED_CHECKPOINTS.md")
lines = [
    "# Best validation epoch vs best SAVED checkpoint",
    "",
    "**Best (val):** lowest BER + routing stability (composite score), any epoch in `validation.csv`.",
    "**Best saved:** same criterion but only among epochs with a `.pyt` in `checkpoints/`.",
    "",
    "| Label | Best ep | BER% | max_use | load_l1 | Saved ep | BER% | max_use | load_l1 | Match? | Saved epochs on disk |",
    "|-------|--------:|-----:|--------:|--------:|---------:|-----:|--------:|--------:|:------:|----------------------|",
]
for r in rows_out:
    lines.append(
        f"| {r['label']} | {r['best_ep']} | {r['best_ber']} | {r['best_max_use']} | {r['best_load_l1']} | "
        f"{r['saved_ep']} | {r['saved_ber']} | {r['saved_max_use']} | {r['saved_load_l1']} | {r['match']} | {r['saved_epochs_list']} |"
    )
lines.append("")
lines.append("**Pure lowest BER** (ignoring stability) may differ — see CSV `best_ber_only_ep`.")
out.write_text("\n".join(lines), encoding="utf-8")

csv_path = Path("results/BEST_VS_SAVED_CHECKPOINTS.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print(out.read_text())
