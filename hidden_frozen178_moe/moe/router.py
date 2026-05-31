import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Routes each sample to top-k experts and exposes full expert probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        jitter_noise: float = 0.01,
        input_dropout: float = 0.1,
        temperature: float = 1.0,
        force_fp32: bool = False,
    ):
        super(Router, self).__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if top_k > num_experts:
            raise ValueError("top_k cannot be larger than num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.temperature = temperature
        self.force_fp32 = bool(force_fp32)
        self.input_dropout = nn.Dropout(p=max(0.0, min(1.0, input_dropout)))
        hidden_dim = max(input_dim // 2, 32)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def set_temperature(self, temperature: float):
        self.temperature = max(float(temperature), 1e-4)

    @staticmethod
    def _select_topk(scaled_logits: torch.Tensor, full_probs: torch.Tensor, top_k: int):
        """Select experts by logits; normalize weights from softmax of selected logits."""
        _, topk_indices = torch.topk(scaled_logits, k=top_k, dim=1)
        selected_logits = torch.gather(scaled_logits, dim=1, index=topk_indices)
        topk_weights = F.softmax(selected_logits, dim=1)
        return topk_indices, topk_weights, full_probs, scaled_logits

    def forward(self, features_flat: torch.Tensor):
        if self.force_fp32 and torch.is_autocast_enabled():
            with torch.autocast(device_type=features_flat.device.type, enabled=False):
                dropped_features = self.input_dropout(features_flat.float())
                logits = self.mlp(dropped_features).float()
                if self.training and self.jitter_noise > 0:
                    logits = logits + torch.randn_like(logits) * self.jitter_noise
                scaled_logits = logits / max(self.temperature, 1e-4)
                full_probs = F.softmax(scaled_logits, dim=1)
                return self._select_topk(scaled_logits, full_probs, self.top_k)

        dropped_features = self.input_dropout(features_flat)
        logits = self.mlp(dropped_features).float()
        if self.training and self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise
        scaled_logits = logits / max(self.temperature, 1e-4)
        full_probs = F.softmax(scaled_logits, dim=1)
        return self._select_topk(scaled_logits, full_probs, self.top_k)

