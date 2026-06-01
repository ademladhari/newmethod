#!/usr/bin/env python3
"""
Evaluate HiDDeN vs MoE under channel attacks (JPEG, crop, resize, etc.).

Embed on 128×128 val crops, apply attack on the watermarked image, decode.

Usage (from repo root):
  python scripts/compare_hidden_vs_moe_attacks.py --data-dir data/coco100k
  python scripts/compare_hidden_vs_moe_attacks.py --attacks jpeg,crop,combined --max-batches 20
  python scripts/compare_hidden_vs_moe_attacks.py --attacks all --hidden-only
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HIDDEN_DIR = ROOT / "hidden"
MOE_DIR = ROOT / "hidden_moe_unfrozen"

DEFAULT_HIDDEN_RUN = ROOT / "hidden" / "runs" / "my_hidden_experiment 2026.05.17--21-47-43"
DEFAULT_HIDDEN_CKPT = ROOT / "hiddenepcoh177" / "my_hidden_experiment--epoch-177.pyt"
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
DEFAULT_OUT = ROOT / "results" / "comparison_hidden_vs_moe" / "attack_summary.csv"


def _import_compare():
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_hidden_vs_moe_watermark as cmp  # noqa: E402

    return cmp


def _import_hidden_noise(device: torch.device):
    sys.path.insert(0, str(HIDDEN_DIR))
    from noise_layers.crop import Crop  # noqa: E402
    from noise_layers.cropout import Cropout  # noqa: E402
    from noise_layers.dropout import Dropout  # noqa: E402
    from noise_layers.identity import Identity  # noqa: E402
    from noise_layers.jpeg_compression import JpegCompression  # noqa: E402
    from noise_layers.quantization import Quantization  # noqa: E402
    from noise_layers.resize import Resize  # noqa: E402
    from moe.moe_noise import GaussianNoise  # noqa: E402

    return {
        "identity": [Identity()],
        "jpeg": [JpegCompression(device)],
        "quant": [Quantization(device)],
        "crop": [Crop((0.55, 0.6), (0.55, 0.6))],
        "cropout": [Cropout((0.55, 0.6), (0.55, 0.6))],
        "dropout": [Dropout((0.55, 0.6))],
        "resize": [Resize((0.5, 1.1))],
        "gaussian": [GaussianNoise((0.01, 0.05))],
        "combined": [
            Resize((0.5, 1.1)),
            JpegCompression(device),
            Crop((0.55, 0.6), (0.55, 0.6)),
            Dropout((0.55, 0.6)),
            GaussianNoise((0.01, 0.05)),
        ],
    }


def apply_attack(layers, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
    """Apply noise layers; resize watermarked tensor back to cover size after spatial attacks."""
    target_hw = (cover.shape[2], cover.shape[3])
    data = [encoded, cover]
    for layer in layers:
        layer.eval()
        data = layer(data)
        if data[0].shape[2:] != target_hw:
            data[0] = F.interpolate(
                data[0], size=target_hw, mode="bilinear", align_corners=False
            )
        # Layers like dropout blend with cover at full resolution
        data[1] = cover
    return data[0]


def bit_error_rate(decoded: torch.Tensor, message: torch.Tensor, use_sigmoid: bool) -> float:
    if use_sigmoid:
        decoded = torch.sigmoid(decoded)
    bits = decoded.detach().cpu().numpy().round().clip(0, 1)
    msg = message.detach().cpu().numpy()
    return float(np.mean(np.abs(bits - msg)))


def decode_hidden(model, noised: torch.Tensor) -> torch.Tensor:
    return model.encoder_decoder.decoder(noised)


def decode_moe(model, noised: torch.Tensor) -> torch.Tensor:
    logits, _, _, _ = model.encoder_decoder.decoder(noised)
    return logits


@torch.no_grad()
def evaluate_attack(
    attack_name: str,
    layers: list,
    hidden_model,
    moe_model,
    loader: DataLoader,
    message_length: int,
    device: torch.device,
    hidden_only: bool,
    max_batches: int | None,
) -> dict:
    hidden_bers: list[float] = []
    moe_bers: list[float] = []
    batches = 0

    for images, _ in loader:
        if max_batches is not None and batches >= max_batches:
            break
        batches += 1
        images = images.to(device)
        message = torch.tensor(
            np.random.choice([0, 1], (images.shape[0], message_length)),
            dtype=torch.float32,
            device=device,
        )

        enc_h = hidden_model.encoder_decoder.encoder(images, message)
        noised_h = apply_attack(layers, enc_h, images)
        dec_h = decode_hidden(hidden_model, noised_h)
        hidden_bers.append(bit_error_rate(dec_h, message, use_sigmoid=False))

        if not hidden_only:
            enc_m = moe_model.encoder_decoder.encoder(images, message)
            noised_m = apply_attack(layers, enc_m, images)
            dec_m = decode_moe(moe_model, noised_m)
            moe_bers.append(bit_error_rate(dec_m, message, use_sigmoid=True))

    row = {
        "attack": attack_name,
        "batches": batches,
        "images": batches * loader.batch_size,
    }
    row["hidden_ber"] = float(np.mean(hidden_bers)) if hidden_bers else float("nan")
    row["hidden_bit_acc"] = 1.0 - row["hidden_ber"]
    if not hidden_only:
        row["moe_ber"] = float(np.mean(moe_bers)) if moe_bers else float("nan")
        row["moe_bit_acc"] = 1.0 - row["moe_ber"]
        row["moe_minus_hidden_acc"] = row["moe_bit_acc"] - row["hidden_bit_acc"]
    return row


def build_val_loader(data_dir: Path, val_folder: str, batch_size: int, hidden_config, num_workers: int):
    sys.path.insert(0, str(HIDDEN_DIR))
    import utils as hidden_utils  # noqa: E402
    from options import TrainingOptions  # noqa: E402

    eval_options = TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=1,
        train_folder=str(data_dir / "train"),
        validation_folder=str(data_dir / val_folder),
        runs_folder=".",
        start_epoch=1,
        experiment_name="attack_eval",
    )
    _, val_loader = hidden_utils.get_data_loaders(hidden_config, eval_options)
    return val_loader


def parse_attacks(attacks_arg: str, device: torch.device) -> dict[str, list]:
    registry = _import_hidden_noise(device)
    if attacks_arg.strip().lower() == "all":
        return registry
    names = [a.strip().lower() for a in attacks_arg.split(",") if a.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Unknown attack(s): {unknown}. Choose from: {', '.join(registry)}")
    return {n: registry[n] for n in names}


def main():
    parser = argparse.ArgumentParser(description="HiDDeN vs MoE under image attacks.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "coco100k")
    parser.add_argument("--val-folder", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=50, help="Val batches per attack (50×16=800 images).")
    parser.add_argument(
        "--attacks",
        type=str,
        default="all",
        help="Comma-separated: identity,jpeg,quant,crop,cropout,dropout,resize,gaussian,combined — or 'all'.",
    )
    parser.add_argument("--hidden-run-folder", type=Path, default=DEFAULT_HIDDEN_RUN)
    parser.add_argument("--hidden-checkpoint", type=Path, default=DEFAULT_HIDDEN_CKPT)
    parser.add_argument("--hidden-options", type=Path, default=None)
    parser.add_argument("--moe-run-folder", type=Path, default=DEFAULT_MOE_RUN)
    parser.add_argument("--moe-checkpoint", type=Path, default=DEFAULT_MOE_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hidden-only", action="store_true", help="Skip MoE (baseline only).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    cmp = _import_compare()
    hidden_options = args.hidden_options or (args.hidden_run_folder / "options-and-config.pickle")

    print("Device:", device)
    print("HiDDeN checkpoint:", args.hidden_checkpoint)
    hidden_model, hidden_cfg, _ = cmp.load_hidden_baseline(
        args.hidden_checkpoint, hidden_options, device
    )

    moe_model = None
    if not args.hidden_only:
        print("MoE checkpoint:", args.moe_checkpoint)
        moe_model, _ = cmp.load_moe_model(args.moe_run_folder, args.moe_checkpoint, device)

    val_dir = args.data_dir / args.val_folder
    if not val_dir.is_dir():
        raise SystemExit(f"Missing: {val_dir}")

    val_loader = build_val_loader(
        args.data_dir, args.val_folder, args.batch_size, hidden_cfg, args.num_workers
    )
    attacks = parse_attacks(args.attacks, device)

    print(f"\nVal: {val_dir} | batch_size={args.batch_size} | max_batches={args.max_batches}")
    print("Attacks:", ", ".join(attacks.keys()))
    print("\nBER = fraction of wrong bits (lower is better). BitAcc = 1 - BER.\n")

    rows = []
    for attack_name, layers in attacks.items():
        print(f"Running attack: {attack_name} ...", flush=True)
        row = evaluate_attack(
            attack_name,
            layers,
            hidden_model,
            moe_model,
            val_loader,
            hidden_cfg.message_length,
            device,
            args.hidden_only,
            args.max_batches,
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.hidden_only:
        print(f"{'Attack':<12} | {'HiDDeN BER':>11} | {'HiDDeN BitAcc':>14}")
        print("-" * 44)
        for r in rows:
            print(f"{r['attack']:<12} | {r['hidden_ber']:>11.4f} | {r['hidden_bit_acc']:>14.4f}")
    else:
        print(f"{'Attack':<12} | {'HiDDeN BER':>11} | {'MoE BER':>9} | {'Δ BitAcc':>10}")
        print("-" * 50)
        for r in rows:
            print(
                f"{r['attack']:<12} | {r['hidden_ber']:>11.4f} | {r['moe_ber']:>9.4f} | {r['moe_minus_hidden_acc']:>+10.4f}"
            )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
