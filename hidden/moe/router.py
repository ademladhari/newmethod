import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Routes each sample to top-k experts and exposes full expert probabilities.
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int):
        super(Router, self).__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if top_k > num_experts:
            raise ValueError("top_k cannot be larger than num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = max(input_dim // 2, 32)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, features_flat: torch.Tensor):
        logits = self.mlp(features_flat)
        full_probs = F.softmax(logits, dim=1)
        topk_weights, topk_indices = torch.topk(full_probs, k=self.top_k, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)
        return topk_indices, topk_weights, full_probs

