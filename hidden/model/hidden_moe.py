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
        decoded_message, router_probs = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message, router_probs


class HiddenMoE:
    def __init__(self, configuration: HiDDenMoEConfiguration, device: torch.device, tb_logger):
        super(HiddenMoE, self).__init__()

        self.encoder_decoder = EncoderDecoderMoE(configuration, MoENoiseLayer(device=device)).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

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

    def set_epoch(self, epoch: int):
        self.encoder_decoder.noiser.set_training_schedule(epoch)
        if epoch <= 20:
            self.current_balance_loss_weight = 0.001
        elif epoch <= 80:
            progress = float(epoch - 20) / 60.0
            self.current_balance_loss_weight = 0.001 + (self.config.balance_loss_weight - 0.001) * max(
                0.0, min(1.0, progress)
            )
        else:
            self.current_balance_loss_weight = self.config.balance_loss_weight

    def _calc_balance_loss(self, router_probs: torch.Tensor):
        expert_usage = torch.mean(router_probs, dim=0)
        uniform = torch.full_like(expert_usage, 1.0 / router_probs.shape[1])
        balance_loss = self.mse_loss(expert_usage, uniform)
        return balance_loss, expert_usage

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            self.optimizer_discrim.zero_grad()
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()

            encoded_images, noised_images, decoded_messages, router_probs = self.encoder_decoder(images, messages)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())
            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            self.optimizer_enc_dec.zero_grad()
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.bce_with_logits_loss(decoded_messages, messages.float())
            g_loss_bal, expert_usage = self._calc_balance_loss(router_probs)
            g_loss = (
                self.config.adversarial_loss * g_loss_adv
                + self.config.encoder_loss * g_loss_enc
                + self.config.decoder_loss * g_loss_dec
                + self.current_balance_loss_weight * g_loss_bal
            )
            g_loss.backward()
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
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.item(),
            "discr_cover_bce": d_loss_on_cover.item(),
            "discr_encod_bce": d_loss_on_encoded.item(),
            "expert_max_use ": float(torch.max(expert_usage).detach().cpu().item()),
        }
        return losses, (encoded_images, noised_images, decoded_messages, router_probs)

    def validate_on_batch(self, batch: list):
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules["final_layer"]
            self.tb_logger.add_tensor("weights/encoder_out", encoder_final.weight)
            expert_linear = self.encoder_decoder.decoder.moe_layer.experts[0].linear
            self.tb_logger.add_tensor("weights/moe_expert0_out", expert_linear.weight)
            discrim_final = self.discriminator._modules["linear"]
            self.tb_logger.add_tensor("weights/discrim_out", discrim_final.weight)

        images, messages = batch
        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            encoded_images, noised_images, decoded_messages, router_probs = self.encoder_decoder(images, messages)
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
            g_loss_bal, expert_usage = self._calc_balance_loss(router_probs)
            g_loss = (
                self.config.adversarial_loss * g_loss_adv
                + self.config.encoder_loss * g_loss_enc
                + self.config.decoder_loss * g_loss_dec
                + self.current_balance_loss_weight * g_loss_bal
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
            "bitwise-error  ": bitwise_avg_err,
            "adversarial_bce": g_loss_adv.item(),
            "discr_cover_bce": d_loss_on_cover.item(),
            "discr_encod_bce": d_loss_on_encoded.item(),
            "expert_max_use ": float(torch.max(expert_usage).detach().cpu().item()),
        }
        return losses, (encoded_images, noised_images, decoded_messages, router_probs)

    def to_stirng(self):
        return "{}\n{}".format(str(self.encoder_decoder), str(self.discriminator))

