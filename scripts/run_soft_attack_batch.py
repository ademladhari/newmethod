#!/usr/bin/env python3
"""Re-run HiDDeN vs MoE attack eval with soft router for thesis ablation runs."""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "scripts" / "compare_hidden_vs_moe_attacks.py"
DATA = ROOT / "data" / "coco100k"
EXP_U = ROOT / "results" / "experiments" / "unfrozen_moe" / "coco100k"
EXP_F = ROOT / "results" / "experiments" / "frozen_moe" / "coco100k"
ARCHIVE_CKPT = ROOT / "results" / "archive" / "moe_run" / "checkpoints"
EXTRACT_DIR = ROOT / "results" / "archive" / "extracted_checkpoints"


def extract_attack_ep65() -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    dest = EXTRACT_DIR / "moe_unfrozen_sym_t14_attack_v1--epoch-65.pyt"
    if dest.exists():
        return dest
    zpath = ROOT / "results" / "archive" / "zips" / "moe_stabilized_v2_milder_epoch50.zip"
    if not zpath.exists():
        raise SystemExit(f"Missing zip for attack ep65: {zpath}")
    with zipfile.ZipFile(zpath) as zf:
        member = next(n for n in zf.namelist() if "attack_v1--epoch-65.pyt" in n)
        zf.extract(member, EXTRACT_DIR)
    nested = next(EXTRACT_DIR.rglob("moe_unfrozen_sym_t14_attack_v1--epoch-65.pyt"))
    if nested != dest:
        dest.write_bytes(nested.read_bytes())
    return dest


RUNS = [
    {
        "label": "R0 ep20",
        "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
        "ckpt_name": "moe_unfrozen_sym_t14_v1--epoch-20.pyt",
        "out": ROOT / "results/comparison_hidden_vs_moe/attack_summary_main_ep20_soft_router.csv",
        "batches": 50,
    },
    {
        "label": "attack ep30",
        "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_attack_v1_ep68_2026-06-04",
        "ckpt_name": "moe_unfrozen_sym_t14_attack_v1--epoch-30.pyt",
        "out": ROOT / "results/comparison_hidden_vs_moe/attack_summary_attack_v1_ep30_soft_router.csv",
        "batches": 50,
    },
    {
        "label": "attack ep65",
        "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_attack_v1_ep68_2026-06-04",
        "ckpt": "extract_ep65",
        "out": ROOT / "results/ber_500_attacks/moe_semicollapsed_attack_ep65_attack_summary_soft_router.csv",
        "batches": 32,
    },
    {
        "label": "continue ep38",
        "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_continue_ep60_2026-06-01",
        "ckpt": ARCHIVE_CKPT / "moe_unfrozen_sym_t14_v1--epoch-38.pyt",
        "out": ROOT / "results/ber_500_attacks/moe_continue_ep38_attack_summary_soft_router.csv",
        "batches": 32,
    },
    {
        "label": "continue ep52",
        "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_continue_ep60_2026-06-01",
        "ckpt": ARCHIVE_CKPT / "moe_unfrozen_sym_t14_v1--epoch-52.pyt",
        "out": ROOT / "results/ber_500_attacks/moe_continue_ep52_attack_summary_soft_router.csv",
        "batches": 32,
    },
    {
        "label": "top2 ep30",
        "run": EXP_U / "4exp_k2/batch128/unfrozen_4exp_top2_b128_ep30_2026-06-03",
        "ckpt_name": "moe_unfrozen_4exp_sparse_top2_b128--epoch-30.pyt",
        "out": ROOT / "results/comparison_hidden_vs_moe/attack_summary_top2_ep20_soft_router.csv",
        "batches": 50,
    },
    {
        "label": "dense k=4 ep20",
        "run": EXP_U / "4exp_k4/batch128/unfrozen_4exp_dense_k4_b128_ep20_2026-06-02",
        "ckpt_name": "moe_unfrozen_4exp_sparse_b128--epoch-20.pyt",
        "out": ROOT / "results/comparison_hidden_vs_moe/attack_summary_dense_4exp_k4_ep20_soft_router.csv",
        "batches": 50,
    },
    {
        "label": "8exp A ep20",
        "run": EXP_U / "8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02",
        "ckpt_name": "moe_unfrozen_8exp_sparse_b128--epoch-20.pyt",
        "out": ROOT / "results/comparison_hidden_vs_moe/attack_summary_8exp_k1_ep20_soft_router.csv",
        "batches": 50,
    },
    {
        "label": "8exp B ep15",
        "run": EXP_U / "8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03",
        "ckpt_name": "moe_unfrozen_8exp_sparse_b128--epoch-15.pyt",
        "out": ROOT / "results/comparison_hidden_vs_moe/attack_summary_8exp_300k_ep15_soft_router.csv",
        "batches": 50,
    },
    {
        "label": "frozen ep30",
        "run": EXP_F / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04",
        "ckpt_name": "moe_sym_bal08_v1--epoch-30.pyt",
        "out": ROOT / "results/ber_500_attacks/attack_summary_frozen_ep30_soft_router.csv",
        "batches": 32,
    },
]


def resolve_ckpt(spec: dict) -> Path:
    if spec.get("ckpt") == "extract_ep65":
        return extract_attack_ep65()
    if "ckpt" in spec:
        return Path(spec["ckpt"])
    return spec["run"] / "checkpoints" / spec["ckpt_name"]


def main() -> int:
    if not PY.exists():
        print("Missing venv python:", PY, file=sys.stderr)
        return 1
    if not DATA.is_dir():
        print("Missing data dir:", DATA, file=sys.stderr)
        return 1

    failed = []
    for i, spec in enumerate(RUNS, 1):
        run = spec["run"]
        ckpt = resolve_ckpt(spec)
        out = spec["out"]
        if not run.is_dir():
            failed.append((spec["label"], f"missing run folder {run}"))
            continue
        if not ckpt.is_file():
            failed.append((spec["label"], f"missing checkpoint {ckpt}"))
            continue

        print(f"\n[{i}/{len(RUNS)}] {spec['label']}")
        print("  run:", run)
        print("  ckpt:", ckpt)
        print("  out:", out)

        cmd = [
            str(PY),
            str(SCRIPT),
            "--device",
            "cuda",
            "--data-dir",
            str(DATA),
            "--val-folder",
            "val",
            "--batch-size",
            "16",
            "--max-batches",
            str(spec["batches"]),
            "--attacks",
            "all",
            "--moe-soft-router",
            "--moe-run-folder",
            str(run),
            "--moe-checkpoint",
            str(ckpt),
            "--output",
            str(out),
        ]
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            failed.append((spec["label"], f"exit code {rc}"))

    print("\n" + "=" * 60)
    if failed:
        print("FAILED:")
        for label, reason in failed:
            print(f"  - {label}: {reason}")
        return 1
    print("All soft-router attack evals completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
