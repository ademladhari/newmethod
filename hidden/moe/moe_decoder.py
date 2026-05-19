import torch.nn as nn

from model.conv_bn_relu import ConvBNRelu
from options_moe import HiDDenMoEConfiguration
from moe.moe_layer import MoELayer


class MoEDecoder(nn.Module):
    """
    Drop-in decoder replacement that exposes router probabilities.
    """

    def __init__(self, config: HiDDenMoEConfiguration):
        super(MoEDecoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))
        self.feature_layers = nn.Sequential(*layers)

        self.moe_layer = MoELayer(
            in_channels=self.channels,
            message_length=config.message_length,
            num_experts=config.num_experts,
            top_k=config.top_k,
            shared_channels=self.channels,
            expert_channels=self.channels,
        )

    def forward(self, image_with_wm):
        features = self.feature_layers(image_with_wm)
        decoded_message, router_probs, topk_indices = self.moe_layer(features)
        return decoded_message, router_probs, topk_indices

