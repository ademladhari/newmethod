#!/usr/bin/env python3
"""
Embed + decode watermarks on N images: HiDDeN epoch-177 vs MoE unfrozen (best run).

Model runs on 128x128 patches (architecture limit). Saved PNGs are full display size
by tiling patches across the image (default) or upscaling a single center crop.

Usage (from repo root):
  python scripts/compare_hidden_vs_moe_watermark.py --data-dir data/coco100k --num-images 10
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
HIDDEN_DIR = ROOT / "hidden"
MOE_DIR = ROOT / "hidden_moe_unfrozen"

DEFAULT_HIDDEN_RUN = ROOT / "hidden" / "runs" / "my_hidden_experiment 2026.05.17--21-47-43"
DEFAULT_HIDDEN_CKPT = ROOT / "hiddenepcoh177" / "my_hidden_experiment--epoch-177.pyt"
DEFAULT_HIDDEN_OPTIONS = DEFAULT_HIDDEN_RUN / "options-and-config.pickle"
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
DEFAULT_OUT = ROOT / "results" / "comparison_hidden_vs_moe"

TENSOR_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _import_hidden_utils():
    sys.path.insert(0, str(HIDDEN_DIR))
    import utils as hidden_utils  # noqa: E402
    from model.hidden import Hidden  # noqa: E402
    from noise_layers.noiser import Noiser  # noqa: E402

    return hidden_utils, Hidden, Noiser


def _import_moe_utils():
    sys.path.insert(0, str(MOE_DIR))
    import utils as moe_utils  # noqa: E402
    from model.hidden import Hidden  # noqa: E402
    from model.hidden_moe import HiddenMoE  # noqa: E402
    from noise_layers.noiser import Noiser  # noqa: E402

    return moe_utils, Hidden, HiddenMoE, Noiser


def load_hidden_baseline(checkpoint_path: Path, options_path: Path, device: torch.device):
    hidden_utils, Hidden, Noiser = _import_hidden_utils()
    _, hidden_config, noise_config = hidden_utils.load_options(str(options_path))
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()
    model.discriminator.eval()
    return model, hidden_config, hidden_utils


def load_moe_model(moe_run: Path, checkpoint_path: Path, device: torch.device):
    moe_utils, _Hidden, HiddenMoE, Noiser = _import_moe_utils()
    options_file = moe_run / "options-and-config.pickle"
    _, hidden_config, _ = moe_utils.load_options(str(options_file))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HiddenMoE(hidden_config, device, tb_logger=None)
    model.load_from_checkpoint(checkpoint, load_optimizers=False)
    model.set_epoch(checkpoint.get("epoch", 20))
    model.encoder_decoder.eval()
    model.discriminator.eval()
    return model, hidden_config


def list_images(val_dir: Path, num_images: int, seed: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in exts)
    if len(paths) < num_images:
        raise SystemExit(f"Need {num_images} images in {val_dir}, found {len(paths)}")
    return random.Random(seed).sample(paths, num_images)


def load_display_image(path: Path, max_side: int) -> Image.Image:
    """Resize so longest edge = max_side (keep aspect ratio)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def pil_to_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    t = transforms.ToTensor()(pil_img)
    return TENSOR_NORM(t).unsqueeze(0).to(device)


def center_crop_tensor(pil_img: Image.Image, patch_h: int, patch_w: int, device: torch.device) -> torch.Tensor:
    return pil_to_tensor(transforms.CenterCrop((patch_h, patch_w))(pil_img), device)


def encoded_tensor_to_uint8(encoded: torch.Tensor) -> np.ndarray:
    t = (encoded.detach().cpu().clamp(-1, 1) + 1) * 127.5
    return t.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)


def watermark_tiled(
    encoder,
    pil_img: Image.Image,
    message: torch.Tensor,
    patch_h: int,
    patch_w: int,
    device: torch.device,
) -> Image.Image:
    """Watermark full image with non-overlapping patch_h x patch_w tiles."""
    arr = np.array(pil_img)
    h0, w0 = arr.shape[:2]
    pad_h = (patch_h - h0 % patch_h) % patch_h
    pad_w = (patch_w - w0 % patch_w) % patch_w
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    hp, wp = arr.shape[:2]

    with torch.no_grad():
        for y in range(0, hp, patch_h):
            for x in range(0, wp, patch_w):
                tile = Image.fromarray(arr[y : y + patch_h, x : x + patch_w])
                tile_t = pil_to_tensor(tile, device)
                enc = encoder(tile_t, message)
                out[y : y + patch_h, x : x + patch_w] = encoded_tensor_to_uint8(enc)

    return Image.fromarray(out[:h0, :w0])


def bit_error(decoded, message, use_sigmoid: bool) -> float:
    if use_sigmoid:
        decoded = torch.sigmoid(decoded)
    bits = decoded.detach().cpu().numpy().round().clip(0, 1)
    msg = message.detach().cpu().numpy()
    return float(np.mean(np.abs(bits - msg)))


def psnr(cover: torch.Tensor, wm: torch.Tensor) -> float:
    mse = F.mse_loss(wm, cover).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(4.0 / mse)


def encode_decode_hidden(model, image: torch.Tensor, message: torch.Tensor):
    enc = model.encoder_decoder.encoder(image, message)
    dec = model.encoder_decoder.decoder(enc)
    return enc, dec


def validate_hidden(model, image: torch.Tensor, message: torch.Tensor):
    """Official HiDDeN path: encoder → noiser → decoder."""
    losses, (_, _, decoded) = model.validate_on_batch([image, message])
    return float(losses["bitwise-error  "]), decoded


def encode_decode_moe(model, image: torch.Tensor, message: torch.Tensor):
    enc = model.encoder_decoder.encoder(image, message)
    dec_logits, _, _, _ = model.encoder_decoder.decoder(enc)
    return enc, dec_logits


def validate_moe(model, image: torch.Tensor, message: torch.Tensor):
    losses, (_, _, decoded, _, _, _) = model.validate_on_batch([image, message], apply_training_noise=False)
    return float(losses["bitwise-error  "]), decoded


def upscale_pil(pil_img: Image.Image, scale: int) -> Image.Image:
    w, h = pil_img.size
    return pil_img.resize((w * scale, h * scale), Image.Resampling.NEAREST)


def pil_to_grid_tensor(pil_img: Image.Image, thumb: int = 512) -> torch.Tensor:
    """Letterbox into thumb×thumb for montages with mixed aspect ratios."""
    pil_img = pil_img.copy()
    pil_img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (thumb, thumb), (128, 128, 128))
    w, h = pil_img.size
    canvas.paste(pil_img, ((thumb - w) // 2, (thumb - h) // 2))
    return transforms.ToTensor()(canvas)


def main():
    parser = argparse.ArgumentParser(description="Compare HiDDeN-177 vs MoE watermarking on N images.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "coco100k")
    parser.add_argument("--val-folder", type=str, default="val")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--message-seed", type=int, default=12345)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Longest edge of saved images (full resolution up to this size).",
    )
    parser.add_argument(
        "--center-crop-only",
        action="store_true",
        help="Save only 128x128 center crop (old behavior).",
    )
    parser.add_argument("--hidden-run-folder", type=Path, default=DEFAULT_HIDDEN_RUN)
    parser.add_argument("--hidden-checkpoint", type=Path, default=DEFAULT_HIDDEN_CKPT)
    parser.add_argument("--hidden-options", type=Path, default=None, help="Default: <hidden-run-folder>/options-and-config.pickle")
    parser.add_argument("--moe-run-folder", type=Path, default=DEFAULT_MOE_RUN)
    parser.add_argument("--moe-checkpoint", type=Path, default=DEFAULT_MOE_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Skip per-image PNGs; only summary.csv is written.")
    args = parser.parse_args()

    val_dir = args.data_dir / args.val_folder
    if not val_dir.is_dir():
        raise SystemExit(f"Missing val folder: {val_dir}")

    device = torch.device(args.device)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if not args.no_save_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Display max side:", args.max_side, "| tiled:", not args.center_crop_only)
    hidden_options = args.hidden_options or (args.hidden_run_folder / "options-and-config.pickle")
    print("Loading HiDDeN baseline:", args.hidden_checkpoint)
    print("HiDDeN options:", hidden_options)
    hidden_model, hidden_cfg, _ = load_hidden_baseline(
        args.hidden_checkpoint, hidden_options, device
    )
    print("Loading MoE:", args.moe_checkpoint)
    moe_model, moe_cfg = load_moe_model(args.moe_run_folder, args.moe_checkpoint, device)

    msg_len = hidden_cfg.message_length
    patch_h, patch_w = hidden_cfg.H, hidden_cfg.W
    encoder_h = hidden_model.encoder_decoder.encoder
    encoder_m = moe_model.encoder_decoder.encoder

    image_paths = list_images(val_dir, args.num_images, args.seed)
    rng = np.random.RandomState(args.message_seed)

    rows = []
    grid_rows = []

    for idx, path in enumerate(image_paths):
        display = load_display_image(path, args.max_side)
        dw, dh = display.size
        message = torch.tensor(rng.choice([0, 1], (1, msg_len)), dtype=torch.float32, device=device)

        # Metrics on center 128×128 (training resolution)
        center_t = center_crop_tensor(display, patch_h, patch_w, device)
        enc_h_center, dec_h_direct = encode_decode_hidden(hidden_model, center_t, message)
        enc_m_center, dec_m_direct = encode_decode_moe(moe_model, center_t, message)
        ber_h_validate, _ = validate_hidden(hidden_model, center_t, message)
        ber_m_validate, _ = validate_moe(moe_model, center_t, message)
        ber_h_direct = bit_error(dec_h_direct, message, use_sigmoid=False)
        ber_m_direct = bit_error(dec_m_direct, message, use_sigmoid=True)

        if args.center_crop_only:
            enc_h, _ = encode_decode_hidden(hidden_model, center_t, message)
            enc_m, _ = encode_decode_moe(moe_model, center_t, message)
            cover_pil = transforms.CenterCrop((patch_h, patch_w))(display)
            hidden_pil = Image.fromarray(encoded_tensor_to_uint8(enc_h))
            moe_pil = Image.fromarray(encoded_tensor_to_uint8(enc_m))
            psnr_h = psnr(center_t, enc_h)
            psnr_m = psnr(center_t, enc_m)
        else:
            cover_pil = display
            hidden_pil = watermark_tiled(encoder_h, display, message, patch_h, patch_w, device)
            moe_pil = watermark_tiled(encoder_m, display, message, patch_h, patch_w, device)
            # PSNR on full image in tensor space (downsampled if huge for speed)
            max_psnr_side = 512
            cover_small = cover_pil.copy()
            if max(cover_small.size) > max_psnr_side:
                s = max_psnr_side / max(cover_small.size)
                cover_small = cover_small.resize(
                    (int(cover_small.width * s), int(cover_small.height * s)),
                    Image.Resampling.LANCZOS,
                )
                hidden_small = hidden_pil.resize(cover_small.size, Image.Resampling.LANCZOS)
                moe_small = moe_pil.resize(cover_small.size, Image.Resampling.LANCZOS)
            else:
                hidden_small, moe_small = hidden_pil, moe_pil
            cover_t = pil_to_tensor(cover_small, device)
            psnr_h = psnr(cover_t, pil_to_tensor(hidden_small, device))
            psnr_m = psnr(cover_t, pil_to_tensor(moe_small, device))

        row = {
            "index": idx,
            "source_file": path.name,
            "save_width": dw,
            "save_height": dh,
            "hidden_ber_validate": ber_h_validate,
            "hidden_ber_direct": ber_h_direct,
            "moe_ber_validate": ber_m_validate,
            "moe_ber_direct": ber_m_direct,
            "hidden_psnr_full": psnr_h,
            "moe_psnr_full": psnr_m,
        }
        rows.append(row)
        print(
            f"[{idx+1}/{args.num_images}] {path.name} ({dw}x{dh}) | "
            f"HiDDeN BER validate={ber_h_validate:.4f} direct={ber_h_direct:.4f} | "
            f"MoE BER validate={ber_m_validate:.4f} direct={ber_m_direct:.4f} | "
            f"PSNR HiDDeN={psnr_h:.2f} MoE={psnr_m:.2f}"
        )

        if not args.no_save_images:
            stem = f"{idx:02d}_{path.stem}"
            cover_pil.save(img_dir / f"{stem}_cover.png")
            hidden_pil.save(img_dir / f"{stem}_hidden_wm.png")
            moe_pil.save(img_dir / f"{stem}_moe_wm.png")
            # Center 128×128 upscaled — shows true model quality without tile seams
            cover_center = transforms.CenterCrop((patch_h, patch_w))(display)
            hidden_center = Image.fromarray(encoded_tensor_to_uint8(enc_h_center))
            moe_center = Image.fromarray(encoded_tensor_to_uint8(enc_m_center))
            scale = max(1, args.max_side // max(patch_h, patch_w))
            upscale_pil(cover_center, scale).save(img_dir / f"{stem}_cover_center{scale}x.png")
            upscale_pil(hidden_center, scale).save(img_dir / f"{stem}_hidden_wm_center{scale}x.png")
            upscale_pil(moe_center, scale).save(img_dir / f"{stem}_moe_wm_center{scale}x.png")

            trip = torch.stack(
                [
                    pil_to_grid_tensor(cover_pil),
                    pil_to_grid_tensor(hidden_pil),
                    pil_to_grid_tensor(moe_pil),
                ]
            )
            vutils.save_image(trip, img_dir / f"{stem}_triptych.png", nrow=3)
            grid_rows.append(trip)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if not args.no_save_images and grid_rows:
        vutils.save_image(torch.cat(grid_rows, dim=0), out_dir / "comparison_grid.png", nrow=3)

    def _summarize_ber(name: str, key: str) -> None:
        bers = [r[key] for r in rows]
        perfect = sum(1 for b in bers if b == 0.0)
        print(f"\n=== {name} @ center 128×128 (validate = encoder→noiser→decoder) ===")
        print(f"  Perfect decode (BER=0): {perfect}/{len(rows)} images")
        print(f"  Mean bit-error rate:    {np.mean(bers):.4f}  (= avg fraction of wrong bits, NOT '% failed')")
        print(f"  Example: BER 0.10 on 30-bit message = 3 bits wrong, 27 bits correct")

    _summarize_ber("HiDDeN", "hidden_ber_validate")
    _summarize_ber("MoE", "moe_ber_validate")
    print(f"\nSaved summary to {out_dir / 'summary.csv'}")
    if args.no_save_images:
        print("(Per-image PNGs skipped: --no-save-images)")
    else:
        print(f"Saved per-image PNGs under {img_dir}")
        print("For visuals without tile seams, open *_hidden_wm_center8x.png (not *_hidden_wm.png).")


if __name__ == "__main__":
    main()
