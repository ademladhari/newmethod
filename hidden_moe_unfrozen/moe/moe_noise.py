import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize


class GaussianNoise(nn.Module):
    def __init__(self, sigma_range=(0.01, 0.05)):
        super(GaussianNoise, self).__init__()
        self.sigma_min = sigma_range[0]
        self.sigma_max = sigma_range[1]

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(noised_image) * sigma
        noised_image = torch.clamp(noised_image + noise, min=-1.0, max=1.0)
        return [noised_image, noised_and_cover[1]]


class DeviceAwareJpegCompression(nn.Module):
    """
    Lazily instantiates one JPEG module per device for DataParallel safety.
    """

    def __init__(self, yuv_keep_weights=(25, 9, 9)):
        super(DeviceAwareJpegCompression, self).__init__()
        self.yuv_keep_weights = yuv_keep_weights
        self._jpeg_by_device = nn.ModuleDict()

    @staticmethod
    def _device_key(device: torch.device):
        if device.type == "cuda":
            return "cuda_{}".format(device.index if device.index is not None else 0)
        return "cpu"

    def _get_or_create_jpeg(self, device: torch.device):
        key = self._device_key(device)
        if key not in self._jpeg_by_device:
            self._jpeg_by_device[key] = JpegCompression(device, self.yuv_keep_weights)
        return self._jpeg_by_device[key]

    def forward(self, noised_and_cover):
        current_device = noised_and_cover[0].device
        jpeg_module = self._get_or_create_jpeg(current_device)
        return jpeg_module(noised_and_cover)


class MoENoiseLayer(nn.Module):
    """
    Distortion-expert noise layer.
    During eval defaults to identity; during training samples one expert.
    """

    def __init__(
        self,
        device,
        crop_ratio=(0.55, 0.6),
        dropout_ratio=(0.55, 0.6),
        resize_ratio=(0.5, 1.1),
        sigma_range=(0.01, 0.05),
    ):
        super(MoENoiseLayer, self).__init__()
        self.identity = Identity()
        self.expert_names = [
            "identity",
            "jpeg",
            "crop",
            "dropout",
            "resize",
            "gaussian",
        ]
        self.experts = [
            self.identity,
            DeviceAwareJpegCompression(),
            Crop(crop_ratio, crop_ratio),
            Dropout(dropout_ratio),
            Resize(resize_ratio),
            GaussianNoise(sigma_range),
        ]
        self.attack_weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._selected_expert_idx = 0
        self._locked_expert_idx = None

    def set_attack_weights(self, attack_weights):
        if len(attack_weights) != len(self.experts):
            raise ValueError("attack_weights must match number of experts.")
        weights = np.asarray(attack_weights, dtype=np.float64)
        if np.sum(weights) <= 0:
            raise ValueError("attack_weights must sum to a positive value.")
        self.attack_weights = weights / np.sum(weights)

    def set_training_schedule(self, epoch: int):
        if epoch <= 20:
            self.set_attack_weights([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return

        if epoch <= 80:
            progress = float(epoch - 20) / 60.0
            non_identity = max(0.0, min(1.0, progress))
            each = non_identity / 5.0
            self.set_attack_weights([1.0 - non_identity, each, each, each, each, each])
            return

        self.set_attack_weights([1.0 / 6.0] * 6)

    def get_last_expert_name(self):
        return self.expert_names[self._selected_expert_idx]

    def pick_batch_expert(self):
        """Pick one expert for the full batch (required for DataParallel multi-GPU)."""
        self._locked_expert_idx = int(
            np.random.choice(np.arange(len(self.experts)), p=self.attack_weights)
        )

    def clear_batch_expert(self):
        self._locked_expert_idx = None

    def _restore_spatial_size(self, noised_and_cover, target_size):
        """Resize noised image back to cover/encoded spatial size (required for multi-GPU gather)."""
        if noised_and_cover[0].shape[2:] != target_size:
            noised_and_cover[0] = F.interpolate(
                noised_and_cover[0],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return noised_and_cover

    def forward(self, encoded_and_cover, apply_attack=None):
        if apply_attack is None:
            apply_attack = self.training
        if not apply_attack:
            self._selected_expert_idx = 0
            return self.identity(encoded_and_cover)

        # Use cover image size — experts may crop/resize encoded in-place.
        target_size = encoded_and_cover[1].shape[2:]

        if self._locked_expert_idx is not None:
            self._selected_expert_idx = self._locked_expert_idx
        else:
            self._selected_expert_idx = int(
                np.random.choice(np.arange(len(self.experts)), p=self.attack_weights)
            )

        noised_and_cover = self.experts[self._selected_expert_idx](encoded_and_cover)
        return self._restore_spatial_size(noised_and_cover, target_size)

