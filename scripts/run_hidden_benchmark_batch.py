#!/usr/bin/env python3
"""Run 14-attack hidden-suite benchmark (HiDDeN noise + 5 WAVES PIL) for all thesis MoE variants."""
from __future__ import annotations

import argparse
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
    name = "repo_benchmark_hidden_batch"
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
    with zipfile.ZipFile(zpath) as zf:
        member = next(n for n in zf.namelist() if "attack_v1--epoch-65.pyt" in n)
        zf.extract(member, EXTRACT_DIR)
    nested = next(EXTRACT_DIR.rglob("moe_unfrozen_sym_t14_attack_v1--epoch-65.pyt"))
    if nested != dest:
        dest.write_bytes(nested.read_bytes())
    return dest


def moe_variants() -> list[dict]:
    """Thesis batch-128 variants; continuation = ep52 only (best post-ep20 by attack wins)."""
    return [
        {
            "name": "MoE-R0-soft",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01",
            "ckpt": EXP_U
            / "4exp_k1/batch128/sym_t14_unfrozen_bal004warm10_ep20_2026-06-01/checkpoints/moe_unfrozen_sym_t14_v1--epoch-20.pyt",
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
            "name": "MoE-frozen-4exp-ep30",
            "run": EXP_F / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04",
            "ckpt": EXP_F
            / "4exp_k1/batch128/sym_bal08_jitter0_bal08warm5_temp14to10_ep30_2026-06-04/checkpoints/moe_sym_bal08_v1--epoch-30.pyt",
        },
        {
            "name": "MoE-attack-ep30",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_attack_v1_ep68_2026-06-04",
            "ckpt": EXP_U
            / "4exp_k1/batch128/sym_t14_unfrozen_attack_v1_ep68_2026-06-04/checkpoints/moe_unfrozen_sym_t14_attack_v1--epoch-30.pyt",
        },
        {
            "name": "MoE-continue-ep52",
            "run": EXP_U / "4exp_k1/batch128/sym_t14_unfrozen_continue_ep60_2026-06-01",
            "ckpt": ARCHIVE_CKPT / "moe_unfrozen_sym_t14_v1--epoch-52.pyt",
        },
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Hidden-suite 14-attack benchmark for MoE variants")
    p.add_argument("--images", type=Path, default=ROOT / "data/coco100k/val")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--output", type=Path, default=ROOT / "results/benchmark_hidden_variants")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--only", nargs="*", help="Subset of variant names")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--skip-hidden", action="store_true", help="Reuse results/benchmark_hidden/results_hidden.csv")
    args = p.parse_args()

    bm = _load_benchmark()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    images = bm.list_images(args.images, args.n_images)
    noise_registry = bm.hidden_noise_registry(device)

    all_rows: list[dict] = []
    hidden_src = ROOT / "results/benchmark_hidden/results_hidden.csv"

    if args.skip_hidden and hidden_src.is_file():
        import csv

        with hidden_src.open(newline="", encoding="utf-8") as f:
            rows_h = list(csv.DictReader(f))
        print(f"Using HiDDeN from {hidden_src}")
        all_rows.extend(rows_h)
    else:
        print("\n=== HiDDeN ep177 ===")
        hidden_model, hidden_cfg = bm.load_hidden(bm.DEFAULT_HIDDEN_CKPT, bm.DEFAULT_HIDDEN_RUN, device)
        rows_h = bm.benchmark_model_hidden(
            "HiDDeN-ep177",
            hidden_model,
            bm.decode_hidden,
            use_sigmoid=False,
            images=images,
            device=device,
            msg_len=hidden_cfg.message_length,
            noise_registry=noise_registry,
        )
        all_rows.extend(rows_h)
        bm.write_csv(args.output / "results_hidden.csv", rows_h)
        del hidden_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    variants = moe_variants()
    if args.only:
        allow = set(args.only)
        variants = [v for v in variants if v["name"] in allow]

    for spec in variants:
        safe = spec["name"].replace("/", "_")
        out_csv = args.output / f"results_{safe}.csv"
        if args.skip_existing and out_csv.is_file():
            import csv

            with out_csv.open(newline="", encoding="utf-8") as f:
                all_rows.extend(list(csv.DictReader(f)))
            print(f"SKIP {spec['name']} (exists)")
            continue
        run, ckpt = Path(spec["run"]), Path(spec["ckpt"])
        if not ckpt.is_file():
            print(f"SKIP {spec['name']}: missing {ckpt}")
            continue
        print(f"\n=== {spec['name']} ===")
        moe_model, moe_cfg = bm.load_moe(ckpt, run, device)
        rows = bm.benchmark_model_hidden(
            spec["name"],
            moe_model,
            bm.decode_moe,
            use_sigmoid=False,
            images=images,
            device=device,
            msg_len=moe_cfg.message_length,
            noise_registry=noise_registry,
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
                {"attack_suite": "hidden", "n_images": len(images), "attacks": all_rows},
                f,
                indent=2,
            )
        bm._print_win_loss(all_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
