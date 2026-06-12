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
        router_input_dropout: float = 0.1,
        router_temperature: float = 1.0,
        router_force_fp32: bool = True,
        expert_dropout: float = 0.1,
        expert_init_offset_scale: float = 1e-3,
        expert_use_group_norm: bool = False,
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
            input_dropout=router_input_dropout,
            temperature=router_temperature,
            force_fp32=router_force_fp32,
        )
        self.experts = nn.ModuleList(
            [
                Expert(
                    in_channels=shared_channels,
                    expert_channels=expert_channels,
                    message_length=message_length,
                    dropout=expert_dropout,
                    use_group_norm=expert_use_group_norm,
                )
                for _ in range(num_experts)
            ]
        )
        self._apply_distinct_expert_init(expert_init_offset_scale)
        # Inference-only diagnostic: blend all experts by router softmax (see set_eval_soft_router).
        self.eval_soft_router = False

    def set_eval_soft_router(self, enabled: bool):
        """When True at eval with top_k=1, use full softmax over all experts instead of hard top-1."""
        self.eval_soft_router = bool(enabled)

    def _apply_distinct_expert_init(self, offset_scale: float):
        if offset_scale <= 0:
            return
        center = (self.num_experts - 1) * 0.5
        with torch.no_grad():
            for idx, expert in enumerate(self.experts):
                scale = (idx - center) * offset_scale
                for param in expert.parameters():
                    if param.ndim > 1:
                        param.add_(torch.randn_like(param) * scale)

    def forward(self, x):
        shared_features = self.shared_extractor(x)
        pooled = self.pool(shared_features).flatten(start_dim=1)
        topk_indices, topk_weights, full_probs, router_logits = self.router(pooled)

        expert_outputs = [expert(shared_features) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        if not self.training and self.top_k == 1 and not self.eval_soft_router:
            # True sparse dispatch at inference: only the selected expert runs per image.
            expert_idx = topk_indices[:, 0]  # [B]
            message_length = self.experts[0].linear.out_features
            combined = torch.zeros(
                x.shape[0], message_length, device=x.device, dtype=shared_features.dtype
            )
            for i, expert in enumerate(self.experts):
                mask = expert_idx == i
                if mask.any():
                    combined[mask] = expert_outputs[i][mask]
            combined = combined * topk_weights[:, 0].to(combined.dtype).unsqueeze(-1)
        elif not self.training and self.top_k == 1 and self.eval_soft_router:
            # Diagnostic: weighted sum over all experts (full router softmax, no hard pick).
            weights = full_probs.unsqueeze(-1).to(stacked_outputs.dtype)
            combined = torch.sum(stacked_outputs * weights, dim=1)
        else:
            # Training or top_k > 1: top-k weighted blend (all experts run in training).
            gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, stacked_outputs.shape[-1])
            selected_outputs = torch.gather(stacked_outputs, dim=1, index=gather_index)
            combined = torch.sum(selected_outputs * topk_weights.to(selected_outputs.dtype).unsqueeze(-1), dim=1)

        return combined, full_probs, topk_indices, router_logits

