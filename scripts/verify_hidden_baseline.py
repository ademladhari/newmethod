"""Verify HiDDeN epoch-177 load + decode paths match official validate_on_batch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
HIDDEN_DIR = ROOT / "hidden"
CKPT_EXPORT = ROOT / "hiddenepcoh177" / "my_hidden_experiment--epoch-177.pyt"
RUN_FOLDER = HIDDEN_DIR / "runs" / "my_hidden_experiment 2026.05.17--21-47-43"
VAL_IMAGE = ROOT / "data" / "coco100k" / "val" / "000000107554.jpg"


def main():
    sys.path.insert(0, str(HIDDEN_DIR))
    import utils
    from model.hidden import Hidden
    from noise_layers.noiser import Noiser
    from noise_layers.identity import Identity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options_file = RUN_FOLDER / "options-and-config.pickle"
    train_options, hidden_config, noise_config = utils.load_options(str(options_file))

    print("=== Config ===")
    print("H, W:", hidden_config.H, hidden_config.W, "message_length:", hidden_config.message_length)
    print("noise layers in pickle:", noise_config)

    ckpt_export = torch.load(CKPT_EXPORT, map_location=device)
    ckpt_run_path = str(sorted((RUN_FOLDER / "checkpoints").glob("*.pyt"))[-1])
    ckpt_run = torch.load(ckpt_run_path, map_location=device)

    def state_l1(a, b):
        return sum((a[k].float() - b[k].float()).abs().sum().item() for k in a if k in b)

    diff = state_l1(ckpt_export["enc-dec-model"], ckpt_run["enc-dec-model"])
    print("\n=== Checkpoint match ===")
    print("export:", CKPT_EXPORT.name, "epoch", ckpt_export.get("epoch"))
    print("run:   ", Path(ckpt_run_path).name, "epoch", ckpt_run.get("epoch"))
    print("enc-dec L1 diff (should be 0):", diff)

  # load image
    tf = transforms.Compose(
        [
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = tf(Image.open(VAL_IMAGE).convert("RGB")).unsqueeze(0).to(device)
    message = torch.tensor(
        np.random.RandomState(12345).choice([0, 1], (1, hidden_config.message_length)),
        dtype=torch.float32,
        device=device,
    )

    def ber(decoded, msg):
        bits = decoded.detach().cpu().numpy().round().clip(0, 1)
        return float(np.mean(np.abs(bits - msg.cpu().numpy())))

    for label, ckpt in [("export_ckpt", ckpt_export), ("run_ckpt", ckpt_run)]:
        noiser = Noiser(noise_config, device)
        model = Hidden(hidden_config, device, noiser, tb_logger=None)
        utils.model_from_checkpoint(model, ckpt)
        model.encoder_decoder.eval()

        with torch.no_grad():
            enc = model.encoder_decoder.encoder(image, message)
            dec_direct = model.encoder_decoder.decoder(enc)
            _, noised, dec_full = model.encoder_decoder(image, message)
            losses, (_, _, dec_val) = model.validate_on_batch([image, message])

        print(f"\n=== {label} ===")
        print("BER direct encode->decode (no noiser):     ", ber(dec_direct, message))
        print("BER encoder_decoder forward (random noise):", ber(dec_full, message))
        print("BER validate_on_batch:                    ", ber(dec_val, message))
        print("validate bitwise-error field:             ", losses["bitwise-error  "])

    # identity-only noiser
    print("\n=== run_ckpt + Identity-only noiser ===")
    noiser_id = Noiser([Identity()], device)
    model = Hidden(hidden_config, device, noiser_id, tb_logger=None)
    utils.model_from_checkpoint(model, ckpt_run)
    model.encoder_decoder.eval()
    with torch.no_grad():
        losses, (enc, noised, dec) = model.validate_on_batch([image, message])
    print("BER validate_on_batch (identity only):", ber(dec, message))
    print("bitwise-error field:", losses["bitwise-error  "])


if __name__ == "__main__":
    main()
