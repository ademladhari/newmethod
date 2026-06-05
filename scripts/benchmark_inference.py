#!/usr/bin/env python3
"""
RQ4 Inference Cost Benchmark — HiDDeN epoch-177 vs MoE R0 (unfrozen, 4 exp, top-1).

Measures decode latency, encode latency, peak GPU/CPU memory, and parameter counts
at batch sizes 1, 8, and 32.  Writes results/rq4_inference_benchmark.csv.

Usage (from repo root):
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --device cpu --repeats 100
    python scripts/benchmark_inference.py --batch-sizes 1,8,32,64
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HIDDEN_DIR = ROOT / "hidden"
MOE_DIR = ROOT / "hidden_moe_unfrozen"

DEFAULT_HIDDEN_CKPT = ROOT / "hiddenepcoh177" / "my_hidden_experiment--epoch-177.pyt"
DEFAULT_HIDDEN_OPTIONS = ROOT / "hidden" / "runs" / "my_hidden_experiment 2026.05.17--21-47-43" / "options-and-config.pickle"
DEFAULT_MOE_RUN = (
    ROOT
    / "results"
    / "experiments"
    / "unfrozen_moe"
    / "coco100k"
    / "4exp_k1"
    / "batch128"
    / "sym_t14_unfrozen_bal004warm10_ep20_2026-06-01"
)
DEFAULT_MOE_CKPT = DEFAULT_MOE_RUN / "checkpoints" / "moe_unfrozen_sym_t14_v1--epoch-20.pyt"
DEFAULT_OUT_CSV = ROOT / "results" / "rq4_inference_benchmark.csv"

IMG_H, IMG_W = 128, 128
MESSAGE_LENGTH = 30


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hidden_baseline(checkpoint_path: Path, options_path: Path, device: torch.device):
    sys.path.insert(0, str(HIDDEN_DIR))
    import utils as hidden_utils
    from model.hidden import Hidden
    from noise_layers.noiser import Noiser

    _, hidden_config, noise_config = hidden_utils.load_options(str(options_path))
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()
    return model, hidden_config


def load_moe_model(run_folder: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(MOE_DIR))
    import utils as moe_utils
    from model.hidden_moe import HiddenMoE

    _, hidden_config, _ = moe_utils.load_options(str(run_folder / "options-and-config.pickle"))
    model = HiddenMoE(hidden_config, device, tb_logger=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_from_checkpoint(checkpoint, load_optimizers=False)
    model.set_epoch(checkpoint["epoch"])
    model.encoder_decoder.eval()
    return model, hidden_config


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def hidden_param_breakdown(model) -> dict:
    enc_dec = model.encoder_decoder
    return {
        "encoder": count_params(enc_dec.encoder),
        "decoder": count_params(enc_dec.decoder),
        "discriminator": count_params(model.discriminator),
        "total_enc_dec": count_params(enc_dec),
    }


def moe_param_breakdown(model) -> dict:
    enc_dec = model.encoder_decoder
    if hasattr(enc_dec, "module"):
        enc_dec = enc_dec.module
    moe_layer = enc_dec.decoder.moe_layer
    return {
        "encoder": count_params(enc_dec.encoder),
        "decoder_feature_layers": count_params(enc_dec.decoder.feature_layers),
        "moe_shared_extractor": count_params(moe_layer.shared_extractor),
        "moe_router": count_params(moe_layer.router),
        "moe_experts_all": count_params(moe_layer.experts),
        "moe_experts_active_1": count_params(moe_layer.experts[0]),
        "decoder_total": count_params(enc_dec.decoder),
        "discriminator": count_params(model.discriminator),
        "total_enc_dec": count_params(enc_dec),
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def time_fn(fn, warmup: int, repeats: int, device: torch.device) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for fn() over `repeats` timed calls after `warmup`."""
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

def benchmark_hidden(model, batch_size: int, device: torch.device, warmup: int, repeats: int) -> dict:
    enc_dec = model.encoder_decoder
    images = torch.randn(batch_size, 3, IMG_H, IMG_W, device=device)
    messages = torch.randint(0, 2, (batch_size, MESSAGE_LENGTH), dtype=torch.float32, device=device)

    # Pre-encode so decoder timing is isolated
    with torch.no_grad():
        encoded = enc_dec.encoder(images, messages)

    def _encode():
        with torch.no_grad():
            enc_dec.encoder(images, messages)

    def _decode():
        with torch.no_grad():
            enc_dec.decoder(encoded)

    def _full():
        with torch.no_grad():
            enc = enc_dec.encoder(images, messages)
            enc_dec.decoder(enc)

    enc_mean, enc_std = time_fn(_encode, warmup, repeats, device)
    dec_mean, dec_std = time_fn(_decode, warmup, repeats, device)
    full_mean, full_std = time_fn(_full, warmup, repeats, device)

    _reset_peak_memory(device)
    with torch.no_grad():
        for _ in range(5):
            enc = enc_dec.encoder(images, messages)
            enc_dec.decoder(enc)
    _sync(device)
    mem_mb = _peak_memory_mb(device)

    return {
        "model": "HiDDeN",
        "batch_size": batch_size,
        "encode_ms_mean": round(enc_mean, 4),
        "encode_ms_std": round(enc_std, 4),
        "decode_ms_mean": round(dec_mean, 4),
        "decode_ms_std": round(dec_std, 4),
        "full_ms_mean": round(full_mean, 4),
        "full_ms_std": round(full_std, 4),
        "peak_mem_mb": round(mem_mb, 2),
        "encode_ms_per_image": round(enc_mean / batch_size, 4),
        "decode_ms_per_image": round(dec_mean / batch_size, 4),
        "full_ms_per_image": round(full_mean / batch_size, 4),
    }


def benchmark_moe(model, batch_size: int, device: torch.device, warmup: int, repeats: int) -> dict:
    enc_dec = model.encoder_decoder
    if hasattr(enc_dec, "module"):
        enc_dec_module = enc_dec.module
    else:
        enc_dec_module = enc_dec

    images = torch.randn(batch_size, 3, IMG_H, IMG_W, device=device)
    messages = torch.randint(0, 2, (batch_size, MESSAGE_LENGTH), dtype=torch.float32, device=device)

    with torch.no_grad():
        encoded = enc_dec_module.encoder(images, messages)

    def _encode():
        with torch.no_grad():
            enc_dec_module.encoder(images, messages)

    def _decode():
        with torch.no_grad():
            enc_dec_module.decoder(encoded)

    def _full():
        with torch.no_grad():
            enc = enc_dec_module.encoder(images, messages)
            enc_dec_module.decoder(enc)

    enc_mean, enc_std = time_fn(_encode, warmup, repeats, device)
    dec_mean, dec_std = time_fn(_decode, warmup, repeats, device)
    full_mean, full_std = time_fn(_full, warmup, repeats, device)

    _reset_peak_memory(device)
    with torch.no_grad():
        for _ in range(5):
            enc = enc_dec_module.encoder(images, messages)
            enc_dec_module.decoder(enc)
    _sync(device)
    mem_mb = _peak_memory_mb(device)

    return {
        "model": "MoE",
        "batch_size": batch_size,
        "encode_ms_mean": round(enc_mean, 4),
        "encode_ms_std": round(enc_std, 4),
        "decode_ms_mean": round(dec_mean, 4),
        "decode_ms_std": round(dec_std, 4),
        "full_ms_mean": round(full_mean, 4),
        "full_ms_std": round(full_std, 4),
        "peak_mem_mb": round(mem_mb, 2),
        "encode_ms_per_image": round(enc_mean / batch_size, 4),
        "decode_ms_per_image": round(dec_mean / batch_size, 4),
        "full_ms_per_image": round(full_mean / batch_size, 4),
    }


# ---------------------------------------------------------------------------
# Parameter summary CSV
# ---------------------------------------------------------------------------

def write_param_csv(hidden_params: dict, moe_params: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, v in hidden_params.items():
        rows.append({"model": "HiDDeN", "component": k, "parameters": v})
    for k, v in moe_params.items():
        rows.append({"model": "MoE", "component": k, "parameters": v})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "component", "parameters"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Parameter breakdown → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ4 inference cost benchmark.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", default="1,8,32", help="Comma-separated batch sizes.")
    parser.add_argument("--warmup", default=10, type=int, help="Warmup passes before timing.")
    parser.add_argument("--repeats", default=50, type=int, help="Timed passes per configuration.")
    parser.add_argument("--hidden-ckpt", default=str(DEFAULT_HIDDEN_CKPT))
    parser.add_argument("--hidden-options", default=str(DEFAULT_HIDDEN_OPTIONS))
    parser.add_argument("--moe-run", default=str(DEFAULT_MOE_RUN))
    parser.add_argument("--moe-ckpt", default=str(DEFAULT_MOE_CKPT))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    args = parser.parse_args()

    device = torch.device(args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    out_csv = Path(args.out_csv)

    print(f"\n{'='*60}")
    print(f"RQ4 Inference Benchmark")
    print(f"Device : {device}")
    print(f"Batches: {batch_sizes}")
    print(f"Warmup : {args.warmup}  Repeats: {args.repeats}")
    print(f"{'='*60}\n")

    # Load models
    print("Loading HiDDeN baseline...")
    hidden_model, hidden_config = load_hidden_baseline(
        Path(args.hidden_ckpt), Path(args.hidden_options), device
    )
    print("Loading MoE R0 model...")
    moe_model, moe_config = load_moe_model(
        Path(args.moe_run), Path(args.moe_ckpt), device
    )

    # Parameter counts
    hidden_params = hidden_param_breakdown(hidden_model)
    moe_params = moe_param_breakdown(moe_model)

    print("\n--- Parameter counts ---")
    print(f"{'Component':<35} {'HiDDeN':>12} {'MoE':>12}")
    print("-" * 60)
    all_keys = sorted(set(hidden_params) | set(moe_params))
    for k in all_keys:
        h = hidden_params.get(k, "-")
        m = moe_params.get(k, "-")
        h_str = f"{h:,}" if isinstance(h, int) else str(h)
        m_str = f"{m:,}" if isinstance(m, int) else str(m)
        print(f"  {k:<33} {h_str:>12} {m_str:>12}")

    param_csv_path = out_csv.parent / "rq4_param_breakdown.csv"
    write_param_csv(hidden_params, moe_params, param_csv_path)

    # Latency benchmarks
    all_rows = []
    for bs in batch_sizes:
        print(f"\n--- Batch size {bs} ---")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        row_h = benchmark_hidden(hidden_model, bs, device, args.warmup, args.repeats)
        row_m = benchmark_moe(moe_model, bs, device, args.warmup, args.repeats)

        for row in (row_h, row_m):
            all_rows.append(row)
            print(
                f"  {row['model']:<8}  "
                f"enc={row['encode_ms_mean']:.2f}ms  "
                f"dec={row['decode_ms_mean']:.2f}ms  "
                f"full={row['full_ms_mean']:.2f}ms  "
                f"mem={row['peak_mem_mb']:.1f}MB"
            )

        dec_ratio = row_m["decode_ms_mean"] / max(row_h["decode_ms_mean"], 1e-9)
        full_ratio = row_m["full_ms_mean"] / max(row_h["full_ms_mean"], 1e-9)
        print(f"  Ratio  dec={dec_ratio:.3f}×  full={full_ratio:.3f}×")

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults written → {out_csv}")
    print("\nRegenerate RQ4 figure with:")
    print("  python scripts/plot_thesis_rq_figures.py")


if __name__ == "__main__":
    main()
