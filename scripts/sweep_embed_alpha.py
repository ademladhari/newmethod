#!/usr/bin/env python3
"""Sweep MoE embed alpha: weak = cover + alpha * (encoded - cover). Identity-channel PSNR/BER."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_hidden_vs_moe_watermark as cmp  # noqa: E402

PATCH = 128
DEFAULT_MOE_RUN = cmp.DEFAULT_MOE_RUN
DEFAULT_MOE_CKPT = cmp.DEFAULT_MOE_CKPT
BENCHMARK_CSV = ROOT / "results" / "benchmark_hidden" / "results_combined.csv"


def load_rgb_patch(path: Path, size: int = PATCH) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        return im.crop((l, t, l + s, t + s)).resize((size, size), Image.Resampling.LANCZOS).copy()


def pil_to_tensor(im: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def psnr_pil(orig: Image.Image, other: Image.Image) -> float:
    a = np.asarray(orig, dtype=np.float32)
    b = np.asarray(other.resize(orig.size, Image.Resampling.LANCZOS), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0**2 / mse))


def bit_error(decoded_logits: torch.Tensor, message: torch.Tensor) -> float:
    bits = torch.sigmoid(decoded_logits).detach().cpu().numpy().round().clip(0, 1)
    msg = message.detach().cpu().numpy()
    return float(np.mean(np.abs(bits - msg)))


def list_images(val_dir: Path, n: int, seed: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in exts)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in idx]


@torch.no_grad()
def eval_alpha(
    model,
    images: list[Path],
    device: torch.device,
    msg_len: int,
    alpha: float,
) -> tuple[float, float, float, float]:
    psnr_vals, ber_vals = [], []
    for i, path in enumerate(images):
        host_pil = load_rgb_patch(path)
        host_t = pil_to_tensor(host_pil, device)
        rng = np.random.default_rng(42 + i)
        msg = torch.tensor(rng.integers(0, 2, (1, msg_len)), dtype=torch.float32, device=device)
        enc = model.encoder_decoder.encoder(host_t, msg)
        weak = host_t + alpha * (enc - host_t)
        weak_pil = tensor_to_pil(weak)
        psnr_vals.append(psnr_pil(host_pil, weak_pil))
        logits, _, _, _ = model.encoder_decoder.decoder(weak)
        ber_vals.append(bit_error(logits, msg))
    return (
        float(np.mean(psnr_vals)),
        float(np.median(psnr_vals)),
        float(np.mean(ber_vals)),
        float(np.median(ber_vals)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "coco100k" / "val")
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-psnr", type=float, default=35.0)
    parser.add_argument("--moe-run", type=Path, default=DEFAULT_MOE_RUN)
    parser.add_argument("--moe-checkpoint", type=Path, default=DEFAULT_MOE_CKPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "results" / "moe_embed_alpha_sweep.csv")
    parser.add_argument("--update-benchmark", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    images = list_images(args.data_dir, args.n_images, args.seed)
    print(f"Images: {len(images)} from {args.data_dir}")

    model, cfg = cmp.load_moe_model(args.moe_run, args.moe_checkpoint, device, soft_router=True)
    model.encoder_decoder.eval()

    alphas = np.round(np.arange(0.50, 1.001, 0.05), 2)
    rows = []
    best = None
    for alpha in alphas:
        psnr_mean, psnr_med, ber_mean, ber_med = eval_alpha(model, images, device, cfg.message_length, float(alpha))
        row = {
            "alpha": alpha,
            "PSNR_mean": psnr_mean,
            "PSNR_median": psnr_med,
            "BER_mean": ber_mean,
            "BER_median": ber_med,
            "bit_accuracy_mean": 1.0 - ber_mean,
        }
        rows.append(row)
        dist = abs(psnr_mean - args.target_psnr)
        print(
            f"alpha={alpha:.2f}  PSNR mean={psnr_mean:.2f} med={psnr_med:.2f}  "
            f"BER mean={ber_mean*100:.3f}%  bit_acc={row['bit_accuracy_mean']*100:.2f}%"
        )
        if best is None or dist < best["dist"]:
            best = {**row, "dist": dist}

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nBest alpha for target PSNR {args.target_psnr}: {best['alpha']:.2f}")
    print(f"  PSNR mean={best['PSNR_mean']:.3f}  median={best['PSNR_median']:.3f}")
    print(f"  BER mean={best['BER_mean']*100:.3f}%  bit_acc={best['bit_accuracy_mean']*100:.2f}%")
    print(f"Sweep saved to {args.out_csv}")

    chosen = best
    # Fine-tune alpha around best coarse step
    lo = max(0.5, chosen["alpha"] - 0.04)
    hi = min(1.0, chosen["alpha"] + 0.04)
    fine = np.round(np.arange(lo, hi + 0.001, 0.01), 2)
    for alpha in fine:
        if alpha in alphas:
            continue
        psnr_mean, psnr_med, ber_mean, ber_med = eval_alpha(model, images, device, cfg.message_length, float(alpha))
        dist = abs(psnr_mean - args.target_psnr)
        print(f"  fine alpha={alpha:.2f}  PSNR={psnr_mean:.2f}  BER={ber_mean*100:.3f}%")
        if dist < chosen["dist"]:
            chosen = {
                "alpha": float(alpha),
                "PSNR_mean": psnr_mean,
                "PSNR_median": psnr_med,
                "BER_mean": ber_mean,
                "BER_median": ber_med,
                "bit_accuracy_mean": 1.0 - ber_mean,
                "dist": dist,
            }

    print(f"\nChosen alpha: {chosen['alpha']:.2f}")
    print(f"  PSNR mean={chosen['PSNR_mean']:.3f}  median={chosen['PSNR_median']:.3f}")
    print(f"  BER mean={chosen['BER_mean']*100:.3f}%")

    meta_path = args.out_csv.with_name("moe_embed_alpha_chosen.json")
    import json

    meta = {
        "alpha": chosen["alpha"],
        "n_images": len(images),
        "target_psnr": args.target_psnr,
        "PSNR_mean": chosen["PSNR_mean"],
        "PSNR_median": chosen["PSNR_median"],
        "BER_mean": chosen["BER_mean"],
        "bit_accuracy_mean": chosen["bit_accuracy_mean"],
        "note": "weak = cover + alpha * (encoded - cover); identity channel only",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Chosen settings: {meta_path}")

    if args.update_benchmark and BENCHMARK_CSV.is_file():
        lines = BENCHMARK_CSV.read_text(encoding="utf-8").splitlines()
        out_lines = []
        for line in lines:
            if line.startswith("MoE-R0-soft,identity,"):
                parts = line.split(",")
                parts[4] = f"{chosen['PSNR_mean']:.14g}"
                parts[5] = f"{chosen['PSNR_median']:.14g}"
                parts[6] = f"{chosen['bit_accuracy_mean']:.14g}"
                line = ",".join(parts)
                print(f"Updated benchmark identity row: PSNR={chosen['PSNR_mean']:.2f} BER={(chosen['BER_mean']*100):.3f}%")
            out_lines.append(line)
        BENCHMARK_CSV.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
