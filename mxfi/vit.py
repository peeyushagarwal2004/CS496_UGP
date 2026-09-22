"""A small vision transformer for 32x32 inputs.

The study so far covers two convolutional networks. A transformer is the third
architecture family the project needs, and it is the one where F18 makes a
sharp prediction: as capacity grows the failure rate becomes dominated by
non-finite propagation, so a format carrying NaN codes (e4m3) should do worse
relative to its bit width than one carrying none (e3m2).

Transformers also differ structurally in ways that could plausibly change the
answer, which is what makes the test worth running:

* every weight matrix is a ``Linear``, so MX blocks run along the model
  dimension rather than a convolution's reduction axis;
* LayerNorm sits between blocks and renormalises activations, which may mask
  part of a corrupted weight's effect the way BatchNorm did before folding;
* attention passes through a softmax, which is bounded and may absorb damage,
  but also mixes every token with every other, which may spread it.

ImageNet is not reachable from the machines available, so this is a CIFAR-10
model trained from scratch rather than DeiT. It keeps the DeiT-Tiny shape
(192 channels, 3 heads) at a depth and patch size suited to 32x32 images.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ViT", "vit_small_cifar", "PatchEmbed", "Attention", "Block"]


class PatchEmbed(nn.Module):
    """Split the image into patches and project each to the model dimension."""

    def __init__(self, img_size: int = 32, patch: int = 4, in_ch: int = 3,
                 dim: int = 192):
        super().__init__()
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)      # (B, N, D)


class Attention(nn.Module):
    """Multi-head self-attention with explicit qkv/proj Linear layers.

    Written out rather than using ``nn.MultiheadAttention`` so that the
    projections are ordinary ``Linear`` modules the MX quantiser can see.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                # each (B, H, N, d)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Vision transformer for small images, classification from a CLS token."""

    def __init__(self, img_size: int = 32, patch: int = 4, dim: int = 192,
                 depth: int = 6, heads: int = 3, num_classes: int = 10):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, 3, dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


def vit_small_cifar(num_classes: int = 10) -> ViT:
    """DeiT-Tiny width and head count, depth 6, 4x4 patches: ~2.7M parameters."""
    return ViT(img_size=32, patch=4, dim=192, depth=6, heads=3,
               num_classes=num_classes)
