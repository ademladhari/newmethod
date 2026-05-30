import argparse
import os

import numpy as np
import torch

import utils
from model.hidden import Hidden
from model.hidden_moe import HiddenMoE
from moe.moe_noise import GaussianNoise
from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.noiser import Noiser
from noise_layers.resize import Resize
from options import TrainingOptions


def _load_model_from_run_folder(run_folder: str, device: torch.device, is_moe: bool):
    options_file = os.path.join(run_folder, "options-and-config.pickle")
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    checkpoint, checkpoint_file = utils.load_last_checkpoint(os.path.join(run_folder, "checkpoints"))

    if is_moe:
        model = HiddenMoE(hidden_config, device, tb_logger=None)
    else:
        model = Hidden(hidden_config, device, Noiser([], device), tb_logger=None)

    if is_moe:
        model.load_from_checkpoint(checkpoint)
    else:
        utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()
    model.discriminator.eval()
    return model, hidden_config, checkpoint_file


def _build_attacks(device: torch.device):
    identity = Identity()
    jpeg = JpegCompression(device)
    crop = Crop((0.55, 0.6), (0.55, 0.6))
    dropout = Dropout((0.55, 0.6))
    resize = Resize((0.5, 1.1))
    gaussian = GaussianNoise((0.01, 0.05))

    return {
        "identity": [identity],
        "jpeg": [jpeg],
        "crop": [crop],
        "dropout": [dropout],
        "resize": [resize],
        "gaussian": [gaussian],
        "combined": [resize, jpeg, crop, dropout, gaussian],
    }


def _apply_attack(attack_layers, encoded_images, cover_images):
    data = [encoded_images.clone(), cover_images]
    for layer in attack_layers:
        data = layer(data)
    return data[0]


def _bit_accuracy(decoded, message, use_sigmoid=False):
    if use_sigmoid:
        decoded = torch.sigmoid(decoded)
    decoded_bits = decoded.detach().cpu().numpy().round().clip(0, 1)
    message_bits = message.detach().cpu().numpy()
    bit_err = np.mean(np.abs(decoded_bits - message_bits))
    return 1.0 - float(bit_err)


def _empty_router_stats(num_experts: int):
    return {
        "selected_count": np.zeros(num_experts, dtype=np.float64),
        "selected_weight_sum": np.zeros(num_experts, dtype=np.float64),
        "prob_sum": np.zeros(num_experts, dtype=np.float64),
        "sample_count": 0,
    }


def _update_router_stats(router_stats, router_probs, topk_indices, topk_weights):
    probs_np = router_probs.detach().cpu().numpy()
    indices_np = topk_indices.detach().cpu().numpy()
    weights_np = topk_weights.detach().cpu().numpy()
    batch_size = probs_np.shape[0]
    router_stats["prob_sum"] += probs_np.sum(axis=0)
    router_stats["sample_count"] += batch_size
    for row_idx in range(indices_np.shape[0]):
        for col_idx in range(indices_np.shape[1]):
            expert_idx = int(indices_np[row_idx, col_idx])
            router_stats["selected_count"][expert_idx] += 1.0
            router_stats["selected_weight_sum"][expert_idx] += float(weights_np[row_idx, col_idx])


def _finalize_router_stats(router_stats, top_k: int):
    sample_count = max(1, int(router_stats["sample_count"]))
    total_selections = max(1.0, float(sample_count * top_k))
    selected_rate = router_stats["selected_count"] / total_selections
    mean_selected_weight = router_stats["selected_weight_sum"] / np.maximum(router_stats["selected_count"], 1e-8)
    mean_prob = router_stats["prob_sum"] / float(sample_count)
    max_use = float(np.max(selected_rate))
    max_sel = max_use * float(top_k)
    return {
        "expert_load": selected_rate.tolist(),
        "expert_prob_mean": mean_prob.tolist(),
        "expert_selected_weight_mean": mean_selected_weight.tolist(),
        "expert_max_use": max_use,
        "expert_max_sel": max_sel,
    }


def evaluate_attack(attack_layers, baseline_model, moe_model, data_loader, message_length, device):
    baseline_acc = []
    moe_acc = []
    num_experts = moe_model._encoder_decoder_module().decoder.moe_layer.num_experts
    top_k = moe_model._encoder_decoder_module().decoder.moe_layer.top_k
    router_stats = _empty_router_stats(num_experts)

    with torch.no_grad():
        for image, _ in data_loader:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

            encoded_baseline = baseline_model.encoder_decoder.encoder(image, message)
            encoded_moe = moe_model.encoder_decoder.encoder(image, message)

            noised_baseline = _apply_attack(attack_layers, encoded_baseline, image)
            noised_moe = _apply_attack(attack_layers, encoded_moe, image)

            decoded_baseline = baseline_model.encoder_decoder.decoder(noised_baseline)
            decoded_moe, router_probs, topk_indices, router_logits = moe_model.encoder_decoder.decoder(noised_moe)
            topk_weights = torch.gather(router_probs, dim=1, index=topk_indices)
            topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)

            baseline_acc.append(_bit_accuracy(decoded_baseline, message, use_sigmoid=False))
            moe_acc.append(_bit_accuracy(decoded_moe, message, use_sigmoid=True))
            _update_router_stats(router_stats, router_probs, topk_indices, topk_weights=topk_weights)

    finalized_router_stats = _finalize_router_stats(router_stats, top_k=top_k)
    return float(np.mean(baseline_acc)), float(np.mean(moe_acc)), finalized_router_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline HiDDeN vs MoE model under each attack.")
    parser.add_argument("--baseline-run-folder", required=True, type=str, help="Baseline run folder.")
    parser.add_argument("--moe-run-folder", required=True, type=str, help="MoE run folder.")
    parser.add_argument("--data-dir", required=True, type=str, help="Dataset root with train/ and val/.")
    parser.add_argument("--batch-size", default=16, type=int, help="Evaluation batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    baseline_model, baseline_config, baseline_ckpt = _load_model_from_run_folder(
        args.baseline_run_folder, device, is_moe=False
    )
    moe_model, moe_config, moe_ckpt = _load_model_from_run_folder(args.moe_run_folder, device, is_moe=True)

    eval_options = TrainingOptions(
        batch_size=args.batch_size,
        number_of_epochs=1,
        train_folder=os.path.join(args.data_dir, "train"),
        validation_folder=os.path.join(args.data_dir, "val"),
        runs_folder=".",
        start_epoch=1,
        experiment_name="eval",
    )
    _, val_loader = utils.get_data_loaders(baseline_config, eval_options)

    attacks = _build_attacks(device)

    print("Loaded baseline checkpoint:", baseline_ckpt)
    print("Loaded MoE checkpoint:", moe_ckpt)
    print("")
    print("Attack      | Baseline BitAcc | MoE BitAcc | Delta")
    print("------------|-----------------|------------|------")

    for attack_name, attack_layers in attacks.items():
        baseline_acc, moe_acc, router_stats = evaluate_attack(
            attack_layers=attack_layers,
            baseline_model=baseline_model,
            moe_model=moe_model,
            data_loader=val_loader,
            message_length=baseline_config.message_length,
            device=device,
        )
        delta = moe_acc - baseline_acc
        print(
            "{:<11} | {:>15.4f} | {:>10.4f} | {:+.4f}".format(
                attack_name, baseline_acc, moe_acc, delta
            )
        )
        print(
            "  MoE routing: max_use={:.4f} max_sel={:.4f} expert_load={}".format(
                router_stats["expert_max_use"],
                router_stats["expert_max_sel"],
                ", ".join("{:.4f}".format(v) for v in router_stats["expert_load"]),
            )
        )
        print(
            "  MoE probs  : mean_prob={}".format(
                ", ".join("{:.4f}".format(v) for v in router_stats["expert_prob_mean"])
            )
        )
        print(
            "  MoE weight : mean_selected_weight={}".format(
                ", ".join("{:.4f}".format(v) for v in router_stats["expert_selected_weight_mean"])
            )
        )


if __name__ == "__main__":
    main()

