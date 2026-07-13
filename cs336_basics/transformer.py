import math

import torch
from einops import einsum, rearrange

# -- linear layer


class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        if not dtype:
            dtype = torch.bfloat16
        if not device:
            device = "cuda"

        std = math.sqrt(1 / in_features + out_features)

        self.W = torch.empty(out_features, in_features, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        return x @ self.W.T


class MyEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.E = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.E, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor):
        return self.E[token_ids]


