#!/usr/bin/env python3
"""Run WAVES benchmark (benchmark_attacks.py) for all MoE thesis variants at 1k images."""
from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP_U = ROOT / "results/experiments/unfrozen_moe/coco100k"
EXP_F = ROOT / "results/experiments/frozen_moe/coco100k"
ARCHIVE_CKPT = ROOT / "results/archive/moe_run/checkpoints"
EXTRACT_DIR = ROOT / "results/archive/extracted_checkpoints"


def _load_benchmark():
    path = ROOT / "benchmark"
    name = "repo_benchmark"
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def _extract_attack_ep65() -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    dest = EXTRACT_DIR / "moe_unfrozen_sym_t14_attack_v1--epoch-65.pyt"
    if dest.exists():
        return dest
    zpath = ROOT / "results/archive/zips/moe_stabilized_v2_milder_epoch50.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"Missing zip for attack ep65: {zpath}")
    with zipfile.ZipFile(zpath) as zf:
        member = next(n for n in zf.namelist() if "attack_v1--epoch-65.pyt" in n)
        zf.extract(member, EXTRACT_DIR)
    nested = next(EXTRACT_DIR.rglob("moe_unfrozen_sym_t14_attack_v1--epoch-65.pyt"))
    if nested != dest:
        dest.write_bytes(nested.read_bytes())
    return dest


def moe_variants() -> list[dict]:
    """All MoE rows from thesis ablation table + frozen 4-exp composite-best checkpoints."""
    return [
        {
            "name": "MoE-R0-soft",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
            "ckpt": EXP_U
            / "4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/checkpoints/moe_unfrozen_sym_t14_v1--epoch-20.pyt",
        },
        {
            "name": "MoE-attack-ep65",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_attack_v1_ep68_2026-06-04",
            "ckpt": _extract_attack_ep65(),
        },
        {
            "name": "MoE-continue-ep38",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_continue_ep60_2026-06-01",
            "ckpt": ARCHIVE_CKPT / "moe_unfrozen_sym_t14_v1--epoch-38.pyt",
        },
        {
            "name": "MoE-continue-ep52",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_continue_ep60_2026-06-01",
            "ckpt": ARCHIVE_CKPT / "moe_unfrozen_sym_t14_v1--epoch-52.pyt",
        },
        {
            "name": "MoE-continue-ep60",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_continue_ep60_2026-06-01",
            "ckpt": ARCHIVE_CKPT / "moe_unfrozen_sym_t14_v1--epoch-60.pyt",
        },
        {
            "name": "MoE-top2-ep30",
            "run": EXP_U / "4exp_k2/batch128/unfrozen_4exp_top2_b128_ep30_2026-06-03",
            "ckpt": EXP_U
            / "4exp_k2/batch128/unfrozen_4exp_top2_b128_ep30_2026-06-03/checkpoints/moe_unfrozen_4exp_sparse_top2_b128--epoch-30.pyt",
        },
        {
            "name": "MoE-dense-k4-ep20",
            "run": EXP_U / "4exp_k4/batch128/unfrozen_4exp_dense_k4_b128_ep20_2026-06-02",
            "ckpt": EXP_U
            / "4exp_k4/batch128/unfrozen_4exp_dense_k4_b128_ep20_2026-06-02/checkpoints/moe_unfrozen_4exp_sparse_b128--epoch-20.pyt",
        },
        {
            "name": "MoE-8exp-A-ep20",
            "run": EXP_U / "8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02",
            "ckpt": EXP_U
            / "8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-02/checkpoints/moe_unfrozen_8exp_sparse_b128--epoch-20.pyt",
        },
        {
            "name": "MoE-8exp-B-ep15",
            "run": EXP_U / "8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03",
            "ckpt": EXP_U
            / "8exp_k1/batch128/unfrozen_8exp_sparse_k1_b128_ep20_2026-06-03/checkpoints/moe_unfrozen_8exp_sparse_b128--epoch-15.pyt",
        },
        {
            "name": "MoE-frozen-4exp-ep16",
            "run": EXP_F / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01",
            "ckpt": EXP_F
            / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01/checkpoints/moe_sym_bal08_v1--epoch-16.pyt",
        },
        {
            "name": "MoE-frozen-4exp-ep10",
            "run": EXP_F / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04",
            "ckpt": EXP_F
            / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04/checkpoints/moe_sym_bal08_v1--epoch-10.pyt",
        },
    ]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description="WAVES benchmark for all MoE variants")
    p.add_argument("--images", type=Path, default=ROOT / "data/coco100k/val")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--output", type=Path, default=ROOT / "results/benchmark_moe_variants")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--include-hidden",
        action="store_true",
        help="Re-run HiDDeN ep177 (default: copy from results/benchmark if present)",
    )
    p.add_argument("--only", nargs="*", help="Subset of variant names (e.g. MoE-frozen-4exp-ep16)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose per-run CSV already exists in --output",
    )
    p.add_argument(
        "--skip-r0",
        action="store_true",
        help="Reuse MoE-R0-soft from results/benchmark/results_moe.csv",
    )
    p.add_argument(
        "--write-comparison",
        action="store_true",
        help="After run, write comparison_tables/ (SSL + HiDDeN vs each MoE)",
    )
    args = p.parse_args()

    bm = _load_benchmark()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    images = bm.list_images(args.images, args.n_images)

    all_rows: list[dict] = []
    existing_hidden = ROOT / "results/benchmark/results_hidden.csv"

    if args.include_hidden or not existing_hidden.is_file():
        print("\n=== HiDDeN ep177 ===")
        hidden_model, hidden_cfg = bm.load_hidden(bm.DEFAULT_HIDDEN_CKPT, bm.DEFAULT_HIDDEN_RUN, device)
        rows_h = bm.benchmark_model_waves(
            "HiDDeN-ep177",
            hidden_model,
            bm.decode_hidden,
            use_sigmoid=False,
            images=images,
            device=device,
            msg_len=hidden_cfg.message_length,
        )
        all_rows.extend(rows_h)
        bm.write_csv(args.output / "results_hidden.csv", rows_h)
        del hidden_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        rows_h = _read_csv_rows(existing_hidden)
        if rows_h and int(rows_h[0].get("n_images", 0)) == args.n_images:
            print(f"Using existing HiDDeN results from {existing_hidden}")
            all_rows.extend(rows_h)
        else:
            print(f"Warning: {existing_hidden} missing or n_images mismatch; skipping HiDDeN")

    variants = moe_variants()
    if args.only:
        allow = set(args.only)
        variants = [v for v in variants if v["name"] in allow]
    if args.skip_r0:
        r0_csv = ROOT / "results/benchmark/results_moe.csv"
        r0_rows = _read_csv_rows(r0_csv)
        if r0_rows and int(float(r0_rows[0].get("n_images", 0))) == args.n_images:
            print(f"Using existing MoE-R0-soft from {r0_csv}")
            all_rows.extend(r0_rows)
            variants = [v for v in variants if v["name"] != "MoE-R0-soft"]
        else:
            print(f"Warning: --skip-r0 but {r0_csv} missing or n_images mismatch")

    for spec in variants:
        safe = spec["name"].replace("/", "_")
        out_csv = args.output / f"results_{safe}.csv"
        if args.skip_existing and out_csv.is_file():
            print(f"SKIP {spec['name']} (exists: {out_csv})")
            all_rows.extend(_read_csv_rows(out_csv))
            continue
        run, ckpt = Path(spec["run"]), Path(spec["ckpt"])
        if not ckpt.is_file():
            print(f"SKIP {spec['name']}: missing checkpoint {ckpt}")
            continue
        print(f"\n=== {spec['name']} ===")
        print(f"  run:  {run}")
        print(f"  ckpt: {ckpt}")
        moe_model, moe_cfg = bm.load_moe(ckpt, run, device)
        rows = bm.benchmark_model_waves(
            spec["name"],
            moe_model,
            bm.decode_moe,
            use_sigmoid=False,
            images=images,
            device=device,
            msg_len=moe_cfg.message_length,
        )
        all_rows.extend(rows)
        bm.write_csv(out_csv, rows)
        del moe_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_rows:
        bm.write_csv(args.output / "results_combined_all.csv", all_rows)
        with open(args.output / "results_combined_all.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "attack_suite": "waves",
                    "n_images": len(images),
                    "variants": [v["name"] for v in variants],
                    "attacks": all_rows,
                },
                f,
                indent=2,
            )

        # Bit-acc pivot for headline attacks
        key_attacks = ("identity", "jpeg_q50", "crop", "gaussian", "combined")
        methods = sorted({r["method"] for r in all_rows})
        print(f"\n{'Attack':<14}", end="")
        for m in methods:
            print(f" {m[:18]:>18}", end="")
        print()
        print("-" * (14 + 19 * len(methods)))
        for atk in key_attacks:
            print(f"{atk:<14}", end="")
            for m in methods:
                row = next((r for r in all_rows if r["method"] == m and r["attack"] == atk), None)
                val = f"{float(row['bit_accuracy']):.3f}" if row else "n/a"
                print(f" {val:>18}", end="")
            print()

        if args.write_comparison:
            import subprocess

            cmd = [
                sys.executable,
                str(ROOT / "scripts/build_waves14_comparison_tables.py"),
                "--combined",
                str(args.output / "results_combined_all.csv"),
                "--output",
                str(args.output),
            ]
            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
