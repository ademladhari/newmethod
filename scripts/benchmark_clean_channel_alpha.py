#!/usr/bin/env python3
"""Clean-channel benchmark: HiDDeN vs MoE with embed strength alpha."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_hidden_vs_moe_watermark as cmp  # noqa: E402

PATCH = 128


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


def list_images(val_dir: Path, n: int, seed: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in exts)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in idx]


@torch.no_grad()
def eval_model(
    *,
    name: str,
    encoder,
    decoder_fn,
    use_sigmoid: bool,
    images: list[Path],
    device: torch.device,
    msg_len: int,
    alpha: float,
) -> dict:
    psnr_vals, ber_vals, mse_vals = [], [], []
    for i, path in enumerate(images):
        host_pil = load_rgb_patch(path)
        host_t = pil_to_tensor(host_pil, device)
        rng = np.random.default_rng(42 + i)
        msg = torch.tensor(rng.integers(0, 2, (1, msg_len)), dtype=torch.float32, device=device)
        enc = encoder(host_t, msg)
        weak = host_t + alpha * (enc - host_t)
        weak_pil = tensor_to_pil(weak)
        psnr_vals.append(psnr_pil(host_pil, weak_pil))
        mse_vals.append(float(F.mse_loss(weak, host_t).item()))
        out = decoder_fn(weak)
        if use_sigmoid:
            bits = torch.sigmoid(out).detach().cpu().numpy().round().clip(0, 1)
        else:
            bits = out.detach().cpu().numpy().round().clip(0, 1)
        ref = msg.detach().cpu().numpy()
        ber_vals.append(float(np.mean(np.abs(bits - ref))))

    ber_mean = float(np.mean(ber_vals))
    return {
        "model": name,
        "alpha": alpha,
        "n_images": len(images),
        "PSNR_mean": float(np.mean(psnr_vals)),
        "PSNR_median": float(np.median(psnr_vals)),
        "encoder_mse_mean": float(np.mean(mse_vals)),
        "BER_mean": ber_mean,
        "BER_pct": ber_mean * 100.0,
        "bit_accuracy_pct": (1.0 - ber_mean) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None, help="Set both models (overrides per-model flags)")
    parser.add_argument("--hidden-alpha", type=float, default=1.0)
    parser.add_argument("--moe-alpha", type=float, default=0.9)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "coco100k" / "val")
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "results" / "clean_channel_alpha_benchmark.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    images = list_images(args.data_dir, args.n_images, args.seed)
    hidden_alpha = args.alpha if args.alpha is not None else args.hidden_alpha
    moe_alpha = args.alpha if args.alpha is not None else args.moe_alpha
    print(f"hidden_alpha={hidden_alpha}  moe_alpha={moe_alpha}  images={len(images)}")

    hidden, h_cfg, _ = cmp.load_hidden_baseline(
        cmp.DEFAULT_HIDDEN_CKPT,
        cmp.DEFAULT_HIDDEN_OPTIONS,
        device,
    )
    moe, m_cfg = cmp.load_moe_model(cmp.DEFAULT_MOE_RUN, cmp.DEFAULT_MOE_CKPT, device, soft_router=True)
    hidden.encoder_decoder.eval()
    moe.encoder_decoder.eval()

    h_res = eval_model(
        name="HiDDeN-ep177",
        encoder=hidden.encoder_decoder.encoder,
        decoder_fn=hidden.encoder_decoder.decoder,
        use_sigmoid=False,
        images=images,
        device=device,
        msg_len=h_cfg.message_length,
        alpha=hidden_alpha,
    )
    m_res = eval_model(
        name="MoE-R0-soft",
        encoder=moe.encoder_decoder.encoder,
        decoder_fn=lambda x: moe.encoder_decoder.decoder(x)[0],
        use_sigmoid=True,
        images=images,
        device=device,
        msg_len=m_cfg.message_length,
        alpha=moe_alpha,
    )

    out = {"hidden_alpha": hidden_alpha, "moe_alpha": moe_alpha, "hidden": h_res, "moe": m_res}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for r in (h_res, m_res):
        print(
            f"{r['model']:16s}  PSNR={r['PSNR_mean']:.2f} dB  "
            f"BER={r['BER_pct']:.3f}%  bit_acc={r['bit_accuracy_pct']:.2f}%  "
            f"enc_mse={r['encoder_mse_mean']:.4f}"
        )
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
