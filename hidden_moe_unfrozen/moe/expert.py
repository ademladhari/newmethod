import torch.nn as nn

from model.conv_bn_relu import ConvBNRelu


def _conv_norm_relu(in_channels: int, out_channels: int, use_group_norm: bool):
    if use_group_norm:
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )
    return ConvBNRelu(in_channels, out_channels)


class Expert(nn.Module):
    """
    Single MoE expert that decodes watermark bits from shared feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        expert_channels: int,
        message_length: int,
        dropout: float = 0.1,
        use_group_norm: bool = False,
    ):
        super(Expert, self).__init__()
        dropout = max(0.0, min(1.0, float(dropout)))
        self.layers = nn.Sequential(
            _conv_norm_relu(in_channels, expert_channels, use_group_norm),
            _conv_norm_relu(expert_channels, expert_channels, use_group_norm),
            nn.Dropout2d(p=dropout),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Linear(expert_channels, message_length)

    def forward(self, feature_map):
        x = self.layers(feature_map)
        x = x.squeeze(3).squeeze(2)
        return self.linear(x)

