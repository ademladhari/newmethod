"""
Build a complete clarity/organization report for all results.
Writes results/CLARITY_REPORT.md
"""
import csv
from pathlib import Path

OUT = Path("results/CLARITY_REPORT.md")
base = Path("results")

lines = []
def w(*args):
    lines.append(" ".join(str(a) for a in args))

# ── helpers ───────────────────────────────────────────────────────────────────
def read_val(csv_path):
    """Return list of dicts from validation.csv."""
    try:
        return list(csv.DictReader(open(csv_path)))
    except:
        return []

def last_metrics(rows):
    if not rows:
        return None
    r = rows[-1]
    ep  = int(r["epoch"])
    ber = float(r["bitwise-error"])*100
    mu  = float(r["expert_max_use"])
    ll  = float(r["train_val_load_l1"]) if r.get("train_val_load_l1") else None
    return ep, ber, mu, ll

def at_epoch(rows, target_ep):
    for r in rows:
        if int(r["epoch"]) == target_ep:
            ber = float(r["bitwise-error"])*100
            mu  = float(r["expert_max_use"])
            ll  = float(r.get("train_val_load_l1","0") or "0")
            return ber, mu, ll
    return None, None, None

def atk_row(csv_path, attack="identity"):
    try:
        for r in csv.DictReader(open(csv_path)):
            if r["attack"] == attack:
                return float(r["moe_bit_acc"])*100, float(r["hidden_bit_acc"])*100
    except:
        pass
    return None, None

# ── SECTION 1: experiment folders ──────────────────────────────────────────
w("# CLARITY REPORT — MoE Experiment Organization\n")
w("Generated automatically. This is the ground truth for what each file/folder is.\n")

w("---")
w("## SECTION 1 — Training Runs (experiment folders)\n")
w("Each numbered entry is a unique run folder under `results/experiments/`.\n")

exp_base = base / "experiments"
run_num = 0

BRANCHES = [
    ("unfrozen_moe", "Unfrozen backbone (encoder + MoE trainable)"),
    ("frozen_moe",   "Frozen backbone (MoE head only)"),
    ("legacy_early", "Legacy early sweeps (pre-sym_t14 recipe)"),
]

for branch_name, branch_desc in BRANCHES:
    branch = exp_base / branch_name
    if not branch.exists():
        continue
    w(f"### {branch_name} — {branch_desc}\n")

    for val_csv in sorted(branch.rglob("validation.csv")):
        if "images" in str(val_csv):
            continue
        parts = val_csv.relative_to(exp_base).parts
        ds, ne, batch_dir, run = parts[1], parts[2], parts[3], parts[4]
        run_num += 1

        rows = read_val(val_csv)
        m = last_metrics(rows)
        if m:
            ep, ber, mu, ll = m
            ll_s = f"{ll:.3f}" if ll is not None else "?"
        else:
            ep, ber, mu, ll_s = "?", None, None, "?"

        ber_s = f"{ber:.2f}%" if ber is not None else "?"
        mu_s  = f"{mu:.3f}" if mu is not None else "?"

        # guess what this run IS based on path + config
        n_exp = ne.split("_k")[0].replace("exp","") + " experts"
        k_val = "k=" + ne.split("_k")[-1]
        batch = batch_dir.replace("batch","batch=")
        frozen = "frozen" if branch_name == "frozen_moe" else "unfrozen"

        w(f"**Run {run_num}: `{run}`**")
        w(f"- **Path:** `experiments/{branch_name}/{ds}/{ne}/{batch_dir}/{run}/`")
        w(f"- **Config:** {frozen}, {n_exp}, {k_val}, {batch}, {ds}")
        w(f"- **Final epoch:** {ep}  |  BER={ber_s}  |  max_use={mu_s}  |  load_l1={ll_s}")

        # special labels
        if "sym_t14_unfrozen_bal004warm10" in run:
            w("- **THESIS ROLE:** ✅ PRIMARY MODEL (R0) — reported in all tables")
        elif "sym_t14_unfrozen_continue_ep60" in run:
            w("- **THESIS ROLE:** continuation run R0 ep21→60 — used for training-duration ablation")
            w("- **NOTE:** Same model as R0, different epochs. archive/moe_run/ has checkpoints ep5–60")
        elif "sym_t14_unfrozen_attack_v1_ep68" in run:
            w("- **THESIS ROLE:** attack-trained variant of R0 (ep68/80)")
            w("- **NOTE:** ep30 checkpoint (in `checkpoints/`) shows dramatically better results than ep65/ep68")
            ber30, mu30, ll30 = at_epoch(rows, 30)
            if ber30 is not None:
                w(f"  - ep30: BER={ber30:.2f}%  max_use={mu30:.3f}  load_l1={ll30:.3f}")
        elif "sym_t14_unfrozen_200k" in run:
            w("- **THESIS ROLE:** data-scale ablation (~240k images vs 100k)")
            w("- **STATUS:** IN PROGRESS (ep16/30) — still training")
        elif "sym_bal08" in run and "ep20" in run:
            w("- **THESIS ROLE:** frozen backbone baseline (batch=128)")
        elif "sym_bal08" in run and "ep30" in run:
            w("- **THESIS ROLE:** frozen backbone extended (same run, ep30 checkpoint)")
        elif "routing_isolation" in run:
            w("- **THESIS ROLE:** fully-collapsed routing diagnostic — moe_routing_isolation_v1")
        elif "sym_t14_temp09" in run:
            w("- **THESIS ROLE:** frozen, lower temp ceiling (T→0.9) diagnostic")
        elif "collapse_diag" in run and "8exp" in str(val_csv):
            w("- **THESIS ROLE:** 8-expert partial-collapse diagnostic")
        elif "unfrozen_4exp_top2" in run:
            w("- **THESIS ROLE:** k=2 routing sparsity ablation")
        elif "unfrozen_4exp_dense_k4" in run:
            w("- **THESIS ROLE:** k=4 (dense) routing ablation")
        elif "unfrozen_8exp_sparse_k1" in run and "2026-06-02" in run:
            w("- **THESIS ROLE:** 8-expert run A (100k)")
            w("- **ATTACK EVAL:** comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20.csv (ep20)")
        elif "unfrozen_8exp_sparse_k1" in run and "2026-06-03" in run:
            w("- **THESIS ROLE:** 8-expert run B (100k) — **WARNING: mislabeled as '300k' in old RESULTS_AUDIT**")
            w("- **ATTACK EVAL:** comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15.csv (actually ep15)")
            ber15, mu15, ll15 = at_epoch(rows, 15)
            if ber15 is not None:
                w(f"  - ep15: BER={ber15:.2f}%  max_use={mu15:.3f}  load_l1={ll15:.3f}")
            w("  - **CORRECTION:** This is 100k data, not 300k. The filename is wrong.")
        else:
            w("- **THESIS ROLE:** legacy sweep (negative result)")
        w("")

# ── SECTION 2: attack eval CSVs ─────────────────────────────────────────────
w("---")
w("## SECTION 2 — Attack Evaluation CSVs\n")
w("### ber_500_attacks/ (500-image clean BER runs, full attack eval)\n")

atk_map = {
    "attack_summary_moe_r0.csv":
        "R0 ep20 (primary model). 500 images. Used for thesis attack comparison table.",
    "attack_summary_moe_r0_identity_test.csv":
        "R0 ep20 identity-only check (small). Sanity test, not used in thesis tables.",
    "attack_summary_frozen_ep30.csv":
        "Frozen sym_bal08 ep30. Used in collapse analysis.",
    "moe_8exp_collapse_diag_attack_summary.csv":
        "8-expert collapse_diag run (ep20 eval). Used in collapse analysis.",
    "moe_collapsed_routing_isolation_attack_summary.csv":
        "Routing isolation run (full collapse, ep40). Used in collapse analysis.",
    "moe_continue_ep38_attack_summary.csv":
        "R0 continuation checkpoint at ep38. Lower clean BER than ep20 but worse routing health.",
    "moe_continue_ep52_attack_summary.csv":
        "R0 continuation checkpoint at ep52. Best clean BER (0.12%) but highest routing gap.",
    "moe_semicollapsed_attack_ep65_attack_summary.csv":
        "Attack-trained run at ep65 (partial collapse). Used in thesis attack curriculum discussion.",
}

for fname, desc in atk_map.items():
    f = base / "ber_500_attacks" / fname
    exists = "✅" if f.exists() else "❌ MISSING"
    w(f"**`ber_500_attacks/{fname}`** {exists}")
    w(f"- {desc}")
    if f.exists():
        moe_id, h_id = atk_row(f)
        if moe_id:
            w(f"- Identity: MoE={moe_id:.1f}%  HiDDeN={h_id:.1f}%")
    w("")

w("### comparison_hidden_vs_moe/ (800-image full attack eval, HiDDeN vs MoE)\n")

cmp_map = {
    "attack_summary_main_ep20.csv":
        "R0 ep20 (primary). 800 images. Main HiDDeN vs MoE comparison. USE THIS for thesis.",
    "attack_summary.csv":
        "R0 ep20 or similar. Slightly different numbers from main_ep20 (different random seed or val set). DO NOT use — use main_ep20 instead.",
    "attack_summary_attack_v1_ep30.csv":
        "Attack-trained run at ep30. *** BEST ATTACK RESULTS IN ENTIRE EXPERIMENT *** Crop +34pp, Gaussian +19pp vs HiDDeN. ep30 is better than ep65 (before routing drift sets in).",
    "attack_summary_8exp_k1_ep20.csv":
        "8-expert run A (100k, ep20). BER=1.70%. Used for N-ablation attack comparison.",
    "attack_summary_8exp_300k_ep15.csv":
        "8-expert run B (100k, ep15). MISLABELED as '300k' — this is coco100k data. ep15 checkpoint, BER≈0.66% (better than ep20). Used for N-ablation.",
    "attack_summary_dense_4exp_k4_ep20.csv":
        "Dense routing k=4, ep20. BER=2.62% clean. Used for k-ablation attack comparison.",
    "attack_summary_top2_ep20.csv":
        "Top-2 routing k=2, ep20. BER=2.66% clean. Used for k-ablation attack comparison.",
    "summary.csv":
        "Per-image BER + PSNR comparison (200 images, no attack). Used for clean-channel quality figure.",
}

for fname, desc in cmp_map.items():
    f = base / "comparison_hidden_vs_moe" / fname
    exists = "✅" if f.exists() else "❌ MISSING"
    w(f"**`comparison_hidden_vs_moe/{fname}`** {exists}")
    w(f"- {desc}")
    if f.exists() and fname != "summary.csv":
        moe_id, h_id = atk_row(f)
        if moe_id:
            w(f"- Identity: MoE={moe_id:.1f}%  HiDDeN={h_id:.1f}%")
    w("")

# ── SECTION 3: archive ───────────────────────────────────────────────────────
w("---")
w("## SECTION 3 — Archive\n")

w("### archive/moe_run/ — R0 continuation checkpoints")
w("Contains checkpoints: ep5, ep10, ep15, ep20, ep22, ep24, ep26, ep28, ep30, ep32, ep34, ep36, ep38, ep40, ep42, ep44, ep46, ep48, ep50, ep52, ep54, ep56, ep58, ep60")
w("**These are the same run as `sym_t14_unfrozen_continue_ep60`** — all epochs are accessible here.")
w("ep38 and ep52 checkpoints were used in `ber_500_attacks/moe_continue_ep38/ep52_attack_summary.csv`.")
w("")

w("### archive/duplicates/ — superseded zip copies")
w("All are duplicates of runs already installed under experiments/. Safe to ignore.")
w("- `routing_isolation_v1_copy_moe_collapse_diag_v1/` — duplicate of routing_isolation run CSVs")
w("- `routing_isolation_v1_copy_moecollapsev1/` — another duplicate of routing_isolation")
w("- `*.source.txt` files — notes on which zip was the duplicate and which was kept")
w("")

w("### archive/loose/moecollapse_csv_exports/ — stale CSVs")
w("Old export of the routing_isolation run. Matches `experiments/frozen_moe/.../routing_isolation.../`.  Safe to ignore.")
w("")

w("### archive/partial/ — checkpoints-only exports")
w("Two folders: `moe_sym_t14_v1_checkpoints_only_2026-06-01` and `moe_sym_t14_v1_checkpoints_only_kaggle_2026-06-01`")
w("These contain ep5/10/15/20 checkpoints from the original R0 run. Same model as the primary run folder.")
w("")

# ── SECTION 4: log files ─────────────────────────────────────────────────────
w("---")
w("## SECTION 4 — Training Logs (*.txt)\n")

log_map = {
    "train.txt":
        "Frozen-backbone run on Colab (`/content/newmethod/hidden_frozen178_moe`). Likely sym_bal08 or sym_t14_t09 frozen run.",
    "trainexp4.txt":
        "Frozen sym_bal08_v1 run (bal=0.08 seen in config). Corresponds to `frozen_moe/.../sym_bal08_*` folder.",
    "train4exp200k.txt":
        "Unfrozen R0-variant on 200k dataset (Kaggle, `/kaggle/working/coco200k/train`, 1889 steps × batch=128 = ~242k images/epoch). Corresponds to `unfrozen_moe/coco200k/.../sym_t14_unfrozen_200k_v1_ep16` folder.",
    "train4exp4k.txt":
        "Unfrozen run, 1628 steps/epoch. Likely the attack-trained variant (`sym_t14_unfrozen_attack_v1`) — needs verification.",
    "continyuemain.txt":
        "R0 continuation from ep20 on coco100k (Kaggle). Identical to deletelater.txt — same run logged twice.",
    "deletelater.txt":
        "DUPLICATE of continyuemain.txt. Same R0 continuation run. Safe to delete after thesis is done.",
    "hiddencontinuetrain.txt":
        "HiDDeN (not MoE) continuation training, ep192/200, 620 steps/epoch. This is the baseline hidden experiment.",
}

for fname, desc in log_map.items():
    f = base / fname
    exists = "✅" if f.exists() else "❌ MISSING"
    size = f"{f.stat().st_size//1024}KB" if f.exists() else "?"
    w(f"**`{fname}`** {exists} ({size})")
    w(f"- {desc}")
    w("")

# ── SECTION 5: summary CSVs and md files ─────────────────────────────────────
w("---")
w("## SECTION 5 — Registry & Summary Files\n")

w("| File | Purpose |")
w("|------|---------|")
w("| `RESULTS_AUDIT.md` | Older hand-written audit. Some entries are outdated (wrong 300k label, missing new runs). Use CLARITY_REPORT.md instead. |")
w("| `FULL_EXPERIMENT_REGISTRY.md` / `.json` | Config-verified registry of the original 21 experiments. Does not include new runs (attack-trained, 200k, continue). |")
w("| `EXPERIMENTS_MAP.md` | Folder layout guide. Accurate but brief. |")
w("| `THESIS_ALL_EXPERIMENTS.csv` / `.md` | Full 28-run registry with BER and routing metrics. Use this for thesis tables. |")
w("| `THESIS_GROUPED_TABLES.md` | Previous grouped table drafts. |")
w("| `thesis_tables_grouped/*.csv` | Per-dimension grouped CSVs for building figures. |")
w("| `rq4_inference_benchmark.csv` | Inference cost benchmark (latency, memory). |")
w("| `rq4_param_breakdown.csv` | Parameter count breakdown. |")
w("| `RESULTS_AUDIT.csv` | Machine-readable version of RESULTS_AUDIT.md. |")
w("| `plots/experiment_summary.csv` | Summary used for scatter/bar plots. |")
w("")

# ── SECTION 6: key discrepancies to fix ──────────────────────────────────────
w("---")
w("## SECTION 6 — Discrepancies & Issues to Fix\n")

w("### Issue 1: `attack_summary_8exp_300k_ep15.csv` is mislabeled")
w("- **File says:** 8-expert model trained on 300k images")
w("- **Reality:** This is `unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03` (100k dataset) evaluated at ep15")
w("- **Evidence:** The experiment folder is under `coco100k/`; ep15 val BER=0.86% matches the file's identity BER ≈ 0.66%")
w("- **Fix:** Rename file to `attack_summary_8exp_run_b_ep15.csv` or note the correct label in the thesis")
w("")

w("### Issue 2: `attack_summary.csv` is a duplicate / ambiguous")
w("- **File:** `comparison_hidden_vs_moe/attack_summary.csv` (no model suffix)")
w("- **Reality:** Appears to be an early R0 evaluation run (different numbers from main_ep20)")
w("- **Fix:** For thesis, always use `attack_summary_main_ep20.csv` (more images, documented)")
w("")

w("### Issue 3: Attack-trained ep30 results not in thesis")
w("- **File:** `comparison_hidden_vs_moe/attack_summary_attack_v1_ep30.csv`")
w("- **Content:** Attack-trained model evaluated at ep30 — best attack results in the whole experiment")
w("  - Crop: +34.3pp vs HiDDeN (87.0% vs 52.7%)")
w("  - Gaussian: +19.1pp (99.4% vs 80.3%)")
w("  - Wins 6/9 attacks vs HiDDeN")
w("- **Current thesis:** Only uses ep65/ep68 (wins 5/9 with partial collapse)")
w("- **Action needed:** Decide whether to report ep30 as the primary attack-trained result (recommended)")
w("")

w("### Issue 4: `continyuemain.txt` and `deletelater.txt` are duplicates")
w("- Both log the exact same R0 continuation run (ep21–60 on coco100k)")
w("- `deletelater.txt` can be deleted")
w("")

w("### Issue 5: FULL_EXPERIMENT_REGISTRY.md is outdated (21 runs, now 29)")
w("- New runs since the registry was built: continue_ep60, attack_v1_ep68, 200k run, top2, dense, 8exp run B")
w("- Use THESIS_ALL_EXPERIMENTS.csv for the full 29-run picture")
w("")

w("### Issue 6: Some legacy_early folders have wrong dataset label")
w("- In FULL_EXPERIMENT_REGISTRY.md, rows 7,8,11,14,15,17,18,19 list `coco20k` as dataset")
w("  but are stored under `experiments/legacy_early/coco100k/` paths")
w("- The actual dataset used (from train log `train_folder`) is coco20k for those runs")
w("- The folder name is wrong — it should be `legacy_early/coco20k/` not `legacy_early/coco100k/`")
w("- Low priority (these are all legacy/failed runs)")
w("")

# Write file
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Written: {OUT}  ({OUT.stat().st_size} bytes)")
