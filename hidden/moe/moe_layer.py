import torch
import torch.nn as nn

from model.conv_bn_relu import ConvBNRelu
from moe.expert import Expert
from moe.router import Router


class MoELayer(nn.Module):
    """
    Shared feature extractor + top-k routed experts.
    """

    def __init__(
        self,
        in_channels: int,
        message_length: int,
        num_experts: int = 8,
        top_k: int = 2,
        shared_channels: int = 64,
        expert_channels: int = 64,
        router_jitter_noise: float = 0.01,
        router_temperature: float = 1.0,
    ):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.shared_extractor = nn.Sequential(
            ConvBNRelu(in_channels, shared_channels),
            ConvBNRelu(shared_channels, shared_channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.router = Router(
            shared_channels,
            num_experts,
            top_k,
            jitter_noise=router_jitter_noise,
            temperature=router_temperature,
        )
        self.experts = nn.ModuleList(
            [
                Expert(
                    in_channels=shared_channels,
                    expert_channels=expert_channels,
                    message_length=message_length,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        shared_features = self.shared_extractor(x)
        pooled = self.pool(shared_features).flatten(start_dim=1)
        topk_indices, topk_weights, full_probs, router_logits = self.router(pooled)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(shared_features))

        stacked_outputs = torch.stack(expert_outputs, dim=1)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, stacked_outputs.shape[-1])
        selected_outputs = torch.gather(stacked_outputs, dim=1, index=gather_index)
        combined = torch.sum(selected_outputs * topk_weights.unsqueeze(-1), dim=1)
        return combined, full_probs, topk_indices, router_logits

