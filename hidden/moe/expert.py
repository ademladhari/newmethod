import torch.nn as nn

from model.conv_bn_relu import ConvBNRelu


class Expert(nn.Module):
    """
    Single MoE expert that decodes watermark bits from shared feature maps.
    """

    def __init__(self, in_channels: int, expert_channels: int, message_length: int):
        super(Expert, self).__init__()
        self.layers = nn.Sequential(
            ConvBNRelu(in_channels, expert_channels),
            ConvBNRelu(expert_channels, expert_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Linear(expert_channels, message_length)

    def forward(self, feature_map):
        x = self.layers(feature_map)
        x.squeeze_(3).squeeze_(2)
        return self.linear(x)

