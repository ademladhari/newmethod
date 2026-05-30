import logging
import os

import numpy as np
import torch
import torch.nn as nn

from model.discriminator import Discriminator
from model.encoder import Encoder
from moe.moe_decoder import MoEDecoder
from moe.moe_noise import MoENoiseLayer
from options_moe import HiDDenMoEConfiguration
from vgg_loss import VGGLoss


class EncoderDecoderMoE(nn.Module):
    def __init__(self, config: HiDDenMoEConfiguration, noiser: MoENoiseLayer):
        super(EncoderDecoderMoE, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.decoder = MoEDecoder(config)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_message, router_probs, topk_indices, router_logits = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message, router_probs, topk_indices, router_logits


class HiddenMoE:
    def __init__(self, configuration: HiDDenMoEConfiguration, device: torch.device, tb_logger):
        super(HiddenMoE, self).__init__()

        self.encoder_decoder = EncoderDecoderMoE(configuration, MoENoiseLayer(device=device)).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.config = configuration
        self.device = device
        self.use_amp = bool(getattr(configuration, "enable_fp16", False) and device.type == "cuda")

        self.balance_loss_start_weight = getattr(configuration, "balance_loss_start_weight", 0.01)
        self.balance_loss_warmup_epochs = max(1, int(getattr(configuration, "balance_loss_warmup_epochs", 20)))
        self.router_z_loss_weight = getattr(configuration, "router_z_loss_weight", 0.001)
        self.router_temperature_start = getattr(configuration, "router_temperature_start", 1.0)
        self.router_temperature_end = getattr(configuration, "router_temperature_end", 1.0)
        self.router_grad_clip_norm = max(0.0, float(getattr(configuration, "router_grad_clip_norm", 1.0)))
        self.expert_weight_decay = max(0.0, float(getattr(configuration, "expert_weight_decay", 1e-4)))
        self.freeze_hidden_backbone = bool(getattr(configuration, "freeze_hidden_backbone", False))
        self.freeze_discriminator = bool(getattr(configuration, "freeze_discriminator", False))
        self.init_hidden_checkpoint = str(getattr(configuration, "init_hidden_checkpoint", ""))
        self.router_fp32 = bool(getattr(configuration, "router_fp32", True))
        self.scaler_enc_dec = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler_discrim = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.optimizer_enc_dec = None
        self.optimizer_discrim = None
        self._build_optimizers()

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.cover_label = 1
        self.encoded_label = 0
        self.current_balance_loss_weight = configuration.balance_loss_weight

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules["final_layer"]
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name("grads/encoder_out"))
            expert_linear = self.encoder_decoder.decoder.moe_layer.experts[0].linear
            expert_linear.weight.register_hook(tb_logger.grad_hook_by_name("grads/moe_expert0_out"))
            discrim_final = self.discriminator._modules["linear"]
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name("grads/discrim_out"))

    @staticmethod
    def _count_trainable_params(module: nn.Module):
        return sum(param.numel() for param in module.parameters() if param.requires_grad)

    def _build_optimizers(self):
        expert_params = list(self.encoder_decoder.decoder.moe_layer.experts.parameters())
        expert_param_ids = {id(param) for param in expert_params}
        non_expert_params = [
            param for param in self.encoder_decoder.parameters() if id(param) not in expert_param_ids and param.requires_grad
        ]
        trainable_expert_params = [param for param in expert_params if param.requires_grad]
        optimizer_groups = []
        if non_expert_params:
            optimizer_groups.append({"params": non_expert_params})
        if trainable_expert_params:
            optimizer_groups.append({"params": trainable_expert_params, "weight_decay": self.expert_weight_decay})
        if not optimizer_groups:
            raise ValueError("No trainable encoder-decoder parameters remain after freezing.")
        self.optimizer_enc_dec = torch.optim.Adam(optimizer_groups)
        trainable_discriminator_params = [param for param in self.discriminator.parameters() if param.requires_grad]
        if trainable_discriminator_params:
            self.optimizer_discrim = torch.optim.Adam(trainable_discriminator_params)
        else:
            self.optimizer_discrim = None

    def _encoder_decoder_module(self) -> EncoderDecoderMoE:
        if isinstance(self.encoder_decoder, nn.DataParallel):
            return self.encoder_decoder.module
        return self.encoder_decoder

    def _discriminator_module(self) -> Discriminator:
        if isinstance(self.discriminator, nn.DataParallel):
            return self.discriminator.module
        return self.discriminator

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def enable_multi_gpu(self) -> int:
        """Wrap encoder-decoder and discriminator in DataParallel when multiple GPUs exist."""
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            self.encoder_decoder = nn.DataParallel(self.encoder_decoder)
            self.discriminator = nn.DataParallel(self.discriminator)
            logging.info("Using {} GPUs with DataParallel.".format(gpu_count))
        return gpu_count

    @staticmethod
    def _normalize_state_dict(state_dict):
        if any(key.startswith("module.") for key in state_dict.keys()):
            return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        return state_dict

    def _log_trainable_param_summary(self):
        encoder_decoder = self._encoder_decoder_module()
        summary = {
            "encoder": self._count_trainable_params(encoder_decoder.encoder),
            "decoder_feature_layers": self._count_trainable_params(encoder_decoder.decoder.feature_layers),
            "moe_shared_extractor": self._count_trainable_params(encoder_decoder.decoder.moe_layer.shared_extractor),
            "moe_router": self._count_trainable_params(encoder_decoder.decoder.moe_layer.router),
            "moe_experts": self._count_trainable_params(encoder_decoder.decoder.moe_layer.experts),
            "discriminator": self._count_trainable_params(self._discriminator_module()),
        }
        logging.info("Trainable parameter counts: {}".format(summary))

    def apply_freeze_settings(self):
        encoder_decoder = self._encoder_decoder_module()
        frozen_modules = []
        if self.freeze_hidden_backbone:
            for param in encoder_decoder.encoder.parameters():
                param.requires_grad = False
            for param in encoder_decoder.decoder.feature_layers.parameters():
                param.requires_grad = False
            frozen_modules.extend(["encoder", "decoder.feature_layers"])
        if self.freeze_discriminator:
            for param in self._discriminator_module().parameters():
                param.requires_grad = False
            frozen_modules.append("discriminator")
        if frozen_modules:
            logging.info("Frozen modules applied: {}".format(", ".join(frozen_modules)))
        else:
            logging.info("Frozen modules applied: none")
        self._build_optimizers()
        self._log_trainable_param_summary()

    def initialize_from_hidden_checkpoint(self, hidden_checkpoint: dict, load_discriminator: bool = True):
        encoder_decoder = self._encoder_decoder_module()
        checkpoint_enc_dec = self._normalize_state_dict(hidden_checkpoint["enc-dec-model"])
        model_state = encoder_decoder.state_dict()
        copied_keys = []
        skipped_keys = []

        for key, value in checkpoint_enc_dec.items():
            if key.startswith("encoder.") and key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                copied_keys.append(key)
            elif key.startswith("encoder."):
                skipped_keys.append(key)

        for key, value in checkpoint_enc_dec.items():
            if not key.startswith("decoder.layers."):
                continue
            remainder = key[len("decoder.layers.") :]
            if not remainder or not remainder[0].isdigit():
                continue
            layer_idx_text, *rest = remainder.split(".", 1)
            layer_idx = int(layer_idx_text)
            if layer_idx > 6 or not rest:
                continue
            mapped_key = "decoder.feature_layers.{}.{}".format(layer_idx, rest[0])
            if mapped_key in model_state and model_state[mapped_key].shape == value.shape:
                model_state[mapped_key] = value
                copied_keys.append("{} -> {}".format(key, mapped_key))
            else:
                skipped_keys.append("{} -> {}".format(key, mapped_key))

        encoder_decoder.load_state_dict(model_state, strict=False)

        copied_discriminator = 0
        skipped_discriminator = 0
        if load_discriminator:
            checkpoint_discrim = self._normalize_state_dict(hidden_checkpoint["discrim-model"])
            discrim_state = self._discriminator_module().state_dict()
            for key, value in checkpoint_discrim.items():
                if key in discrim_state and discrim_state[key].shape == value.shape:
                    discrim_state[key] = value
                    copied_discriminator += 1
                else:
                    skipped_discriminator += 1
            self._discriminator_module().load_state_dict(discrim_state, strict=False)

        logging.info(
            "Warm-start transfer summary | copied={} skipped={} discrim_copied={} discrim_skipped={}".format(
                len(copied_keys), len(skipped_keys), copied_discriminator, skipped_discriminator
            )
        )
        if skipped_keys:
            preview = skipped_keys[:8]
            logging.info("Warm-start skipped key preview: {}".format(preview))

    def load_from_checkpoint(self, checkpoint):
        enc_dec_state = self._normalize_state_dict(checkpoint["enc-dec-model"])
        discrim_state = self._normalize_state_dict(checkpoint["discrim-model"])
        self._encoder_decoder_module().load_state_dict(enc_dec_state)
        self._discriminator_module().load_state_dict(discrim_state)
        if self.optimizer_enc_dec is not None and checkpoint.get("enc-dec-optim") is not None:
            self.optimizer_enc_dec.load_state_dict(checkpoint["enc-dec-optim"])
        if self.optimizer_discrim is not None and checkpoint.get("discrim-optim") is not None:
            self.optimizer_discrim.load_state_dict(checkpoint["discrim-optim"])

    def save_checkpoint(self, experiment_name: str, epoch: int, checkpoint_folder: str):
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        checkpoint_filename = os.path.join(checkpoint_folder, "{}--epoch-{}.pyt".format(experiment_name, epoch))
        logging.info("Saving checkpoint to {}".format(checkpoint_filename))
        checkpoint = {
            "enc-dec-model": self._encoder_decoder_module().state_dict(),
            "enc-dec-optim": self.optimizer_enc_dec.state_dict() if self.optimizer_enc_dec is not None else None,
            "discrim-model": self._discriminator_module().state_dict(),
            "discrim-optim": self.optimizer_discrim.state_dict() if self.optimizer_discrim is not None else None,
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_filename)
        logging.info("Saving checkpoint done.")

    def set_epoch(self, epoch: int):
        encoder_decoder = self._encoder_decoder_module()
        encoder_decoder.noiser.set_training_schedule(epoch)
        progress = max(0.0, min(1.0, float(epoch - 1) / float(self.balance_loss_warmup_epochs)))
        self.current_balance_loss_weight = self.balance_loss_start_weight + (
            self.config.balance_loss_weight - self.balance_loss_start_weight
        ) * progress

        total_epochs = max(float(getattr(self.config, "number_of_epochs", 200)), 1.0)
        progress = max(0.0, min(1.0, float(epoch - 1) / total_epochs))
        current_temperature = self.router_temperature_start + (
            self.router_temperature_end - self.router_temperature_start
        ) * progress
        encoder_decoder.decoder.moe_layer.router.set_temperature(current_temperature)

    def _calc_balance_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor):
        num_experts = router_probs.shape[1]
        importance = torch.mean(router_probs, dim=0)
        topk_one_hot = torch.nn.functional.one_hot(topk_indices, num_classes=num_experts).float()
        load = torch.mean(topk_one_hot.reshape(-1, num_experts), dim=0)
        balance_loss = num_experts * torch.sum(importance * load)
        return balance_loss, load

    @staticmethod
    def _calc_router_z_loss(router_logits: torch.Tensor):
        z = torch.logsumexp(router_logits, dim=-1)
        return torch.mean(z ** 2)

    @staticmethod
    def _calc_router_entropy(router_probs: torch.Tensor):
        entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1)
        return torch.mean(entropy)

    def _calc_expert_weight_similarity(self):
        experts = self._encoder_decoder_module().decoder.moe_layer.experts
        if len(experts) < 2:
            return torch.tensor(0.0, device=self.device)
        flat_weights = []
        for expert in experts:
            flat_weights.append(expert.linear.weight.reshape(-1))
        weight_matrix = torch.stack(flat_weights, dim=0)
        normalized = torch.nn.functional.normalize(weight_matrix, p=2, dim=1, eps=1e-8)
        cosine_matrix = torch.mm(normalized, normalized.t())
        mask = torch.triu(torch.ones_like(cosine_matrix), diagonal=1).bool()
        return cosine_matrix[mask].mean()

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            if self.optimizer_discrim is not None:
                self.optimizer_discrim.zero_grad(set_to_none=True)
                with self._autocast_context():
                    d_on_cover = self.discriminator(images)
                    d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
                if self.use_amp:
                    self.scaler_discrim.scale(d_loss_on_cover).backward()
                else:
                    d_loss_on_cover.backward()
            else:
                with torch.no_grad():
                    with self._autocast_context():
                        d_on_cover = self.discriminator(images)
                        d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            noiser = self._encoder_decoder_module().noiser
            noiser.pick_batch_expert()
            try:
                with self._autocast_context():
                    encoded_images, noised_images, decoded_messages, router_probs, topk_indices, router_logits = self.encoder_decoder(
                        images, messages
                    )
            finally:
                noiser.clear_batch_expert()

            if self.optimizer_discrim is not None:
                with self._autocast_context():
                    d_on_encoded = self.discriminator(encoded_images.detach())
                    d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())
                if self.use_amp:
                    self.scaler_discrim.scale(d_loss_on_encoded).backward()
                    self.scaler_discrim.step(self.optimizer_discrim)
                    self.scaler_discrim.update()
                else:
                    d_loss_on_encoded.backward()
                    self.optimizer_discrim.step()
            else:
                with torch.no_grad():
                    with self._autocast_context():
                        d_on_encoded = self.discriminator(encoded_images.detach())
                        d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            self.optimizer_enc_dec.zero_grad(set_to_none=True)
            with self._autocast_context():
                d_on_encoded_for_enc = self.discriminator(encoded_images)
                g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

                if self.vgg_loss is None:
                    g_loss_enc = self.mse_loss(encoded_images, images)
                else:
                    vgg_on_cov = self.vgg_loss(images)
                    vgg_on_enc = self.vgg_loss(encoded_images)
                    g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

                g_loss_dec = self.bce_with_logits_loss(decoded_messages, messages.float())
                g_loss_bal, expert_load = self._calc_balance_loss(router_probs, topk_indices)
                g_loss_z = self._calc_router_z_loss(router_logits)
                router_entropy = self._calc_router_entropy(router_probs)
                expert_weight_similarity = self._calc_expert_weight_similarity()
                g_loss = (
                    self.config.adversarial_loss * g_loss_adv
                    + self.config.encoder_loss * g_loss_enc
                    + self.config.decoder_loss * g_loss_dec
                    + self.current_balance_loss_weight * g_loss_bal
                    + self.router_z_loss_weight * g_loss_z
                )
            if self.use_amp:
                self.scaler_enc_dec.scale(g_loss).backward()
            else:
                g_loss.backward()
            if self.router_grad_clip_norm > 0:
                if self.use_amp:
                    self.scaler_enc_dec.unscale_(self.optimizer_enc_dec)
                router_params = self._encoder_decoder_module().decoder.moe_layer.router.parameters()
                router_grad_norm = torch.nn.utils.clip_grad_norm_(router_params, self.router_grad_clip_norm)
                router_grad_norm_value = float(router_grad_norm.detach().cpu().item())
            else:
                router_grad_norm_value = 0.0
            if self.use_amp:
                self.scaler_enc_dec.step(self.optimizer_enc_dec)
                self.scaler_enc_dec.update()
            else:
                self.optimizer_enc_dec.step()

        decoded_rounded = torch.sigmoid(decoded_messages).detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
            batch_size * messages.shape[1]
        )

        losses = {
            "loss           ": g_loss.item(),
            "encoder_mse    ": g_loss_enc.item(),
            "dec_bce        ": g_loss_dec.item(),
            "balance_loss   ": g_loss_bal.item(),
            "router_z_loss  ": g_loss_z.item(),
            "router_entropy ": router_entropy.item(),
            "router_grad_norm": router_grad_norm_value,
            "expert_w_cos  ": float(expert_weight_similarity.detach().cpu().item()),
            "balance_alpha ": self.current_balance_loss_weight,
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.item(),
            "discr_cover_bce": d_loss_on_cover.item(),
            "discr_encod_bce": d_loss_on_encoded.item(),
            "expert_max_use ": float(torch.max(expert_load).detach().cpu().item()),
            "expert_max_sel ": float(torch.max(expert_load).detach().cpu().item() * topk_indices.shape[1]),
        }
        for idx, load_value in enumerate(expert_load.detach().cpu().tolist()):
            losses["expert_load_{} ".format(idx)] = float(load_value)
        return losses, (encoded_images, noised_images, decoded_messages, router_probs, topk_indices, router_logits)

    def validate_on_batch(self, batch: list):
        if self.tb_logger is not None:
            encoder_decoder = self._encoder_decoder_module()
            discriminator = self._discriminator_module()
            encoder_final = encoder_decoder.encoder._modules["final_layer"]
            self.tb_logger.add_tensor("weights/encoder_out", encoder_final.weight)
            expert_linear = encoder_decoder.decoder.moe_layer.experts[0].linear
            self.tb_logger.add_tensor("weights/moe_expert0_out", expert_linear.weight)
            discrim_final = discriminator._modules["linear"]
            self.tb_logger.add_tensor("weights/discrim_out", discrim_final.weight)

        images, messages = batch
        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            with self._autocast_context():
                d_on_cover = self.discriminator(images)
                d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            noiser = self._encoder_decoder_module().noiser
            noiser.pick_batch_expert()
            try:
                with self._autocast_context():
                    encoded_images, noised_images, decoded_messages, router_probs, topk_indices, router_logits = self.encoder_decoder(
                        images, messages
                    )
            finally:
                noiser.clear_batch_expert()

            with self._autocast_context():
                d_on_encoded = self.discriminator(encoded_images)
                d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())
                d_on_encoded_for_enc = self.discriminator(encoded_images)
                g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

                if self.vgg_loss is None:
                    g_loss_enc = self.mse_loss(encoded_images, images)
                else:
                    vgg_on_cov = self.vgg_loss(images)
                    vgg_on_enc = self.vgg_loss(encoded_images)
                    g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

                g_loss_dec = self.bce_with_logits_loss(decoded_messages, messages.float())
                g_loss_bal, expert_load = self._calc_balance_loss(router_probs, topk_indices)
                g_loss_z = self._calc_router_z_loss(router_logits)
                router_entropy = self._calc_router_entropy(router_probs)
                expert_weight_similarity = self._calc_expert_weight_similarity()
                g_loss = (
                    self.config.adversarial_loss * g_loss_adv
                    + self.config.encoder_loss * g_loss_enc
                    + self.config.decoder_loss * g_loss_dec
                    + self.current_balance_loss_weight * g_loss_bal
                    + self.router_z_loss_weight * g_loss_z
                )

        decoded_rounded = torch.sigmoid(decoded_messages).detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
            batch_size * messages.shape[1]
        )
        losses = {
            "loss           ": g_loss.item(),
            "encoder_mse    ": g_loss_enc.item(),
            "dec_bce        ": g_loss_dec.item(),
            "balance_loss   ": g_loss_bal.item(),
            "router_z_loss  ": g_loss_z.item(),
            "router_entropy ": router_entropy.item(),
            "expert_w_cos  ": float(expert_weight_similarity.detach().cpu().item()),
            "balance_alpha ": self.current_balance_loss_weight,
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.item(),
            "discr_cover_bce": d_loss_on_cover.item(),
            "discr_encod_bce": d_loss_on_encoded.item(),
            "expert_max_use ": float(torch.max(expert_load).detach().cpu().item()),
            "expert_max_sel ": float(torch.max(expert_load).detach().cpu().item() * topk_indices.shape[1]),
        }
        for idx, load_value in enumerate(expert_load.detach().cpu().tolist()):
            losses["expert_load_{} ".format(idx)] = float(load_value)
        return losses, (encoded_images, noised_images, decoded_messages, router_probs, topk_indices, router_logits)

    def to_stirng(self):
        return "{}\n{}".format(str(self.encoder_decoder), str(self.discriminator))

