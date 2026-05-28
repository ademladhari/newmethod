import torch.nn as nn

from model.conv_bn_relu import ConvBNRelu


class Expert(nn.Module):
    """
    Single MoE expert that decodes watermark bits from shared feature maps.
    """

    def __init__(self, in_channels: int, expert_channels: int, message_length: int, dropout: float = 0.1):
        super(Expert, self).__init__()
        dropout = max(0.0, min(1.0, float(dropout)))
        self.layers = nn.Sequential(
            ConvBNRelu(in_channels, expert_channels),
            ConvBNRelu(expert_channels, expert_channels),
            nn.Dropout2d(p=dropout),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Linear(expert_channels, message_length)

    def forward(self, feature_map):
        x = self.layers(feature_map)
        x = x.squeeze(3).squeeze(2)
        return self.linear(x)

