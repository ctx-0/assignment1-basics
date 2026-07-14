import math

import torch
from torch import nn
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


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.W = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        # rms = torch.sqrt(torch.mean(x*x, dim=1, keepdim=True) + self.eps)
        # x_normed = x / rms

        in_dtype = x.dtype
        x = x.to(torch.float32)

        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_normed = x * inv_rms

        res = self.W * x_normed
        return res.to(in_dtype)


# SiLU
def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


# GLU
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # d_ff = int(d_model * (8 / 3))
        d_ff = 64 * math.ceil(8 * d_model / (3 * 64))

        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.W3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor):
        hidden = SiLU(self.W1(x)) * self.W3(x)
        return self.W2(hidden)
