# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorFusionBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 64, gate_init: float = 0.0) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.prior_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, patch_tokens: torch.Tensor, prior_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.shape != prior_tokens.shape:
            raise ValueError(
                f"patch_tokens and prior_tokens must have the same shape, got "
                f"{patch_tokens.shape} vs {prior_tokens.shape}"
            )
        fused = torch.cat(
            [
                self.token_norm(patch_tokens),
                self.prior_norm(prior_tokens.to(dtype=patch_tokens.dtype)),
            ],
            dim=-1,
        )
        return patch_tokens + self.gate * self.mlp(fused)


class SummaryTokenFusionBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 64, gate_init: float = 0.0) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.value = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, patch_tokens: torch.Tensor, summary_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3 or summary_tokens.ndim != 3:
            raise ValueError(
                f"Expected 3D patch/summary tensors, got {patch_tokens.shape} and {summary_tokens.shape}"
            )
        if patch_tokens.shape[0] != summary_tokens.shape[0]:
            raise ValueError(
                f"patch_tokens and summary_tokens batch dims must match, got "
                f"{patch_tokens.shape} vs {summary_tokens.shape}"
            )

        token_features = self.token_norm(patch_tokens)
        summary_features = self.summary_norm(summary_tokens.to(dtype=patch_tokens.dtype))
        query = self.query(token_features)
        key = self.key(summary_features)
        value = self.value(summary_features)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(max(1, key.shape[-1])),
            dim=-1,
        )
        context = torch.matmul(attention, value)
        return patch_tokens + self.gate * self.out(context)


class HumanPriorAdapter(nn.Module):
    """
    Projects image-space human prior maps to patch-token space and exposes
    lightweight fusion blocks for the input stage and every transformer layer.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        summary_in_channels: int = 0,
        hidden_dim: int = 64,
        gate_init: float = 0.0,
        multi_scale_factors: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.in_channels = in_channels
        self.summary_in_channels = summary_in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        normalized_scale_factors = []
        for factor in (multi_scale_factors or (1,)):
            factor = max(1, int(factor))
            if factor not in normalized_scale_factors:
                normalized_scale_factors.append(factor)
        if 1 not in normalized_scale_factors:
            normalized_scale_factors = [1, *normalized_scale_factors]
        self.multi_scale_factors = tuple(normalized_scale_factors)
        if len(self.multi_scale_factors) > 1:
            self.register_buffer(
                "scale_factors_tensor",
                torch.tensor(self.multi_scale_factors, dtype=torch.int64),
                persistent=True,
            )
        else:
            self.register_buffer("scale_factors_tensor", None, persistent=False)

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )
        self.scale_projs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
            )
            for _ in self.multi_scale_factors[1:]
        )
        if len(self.multi_scale_factors) > 1:
            self.scale_logits = nn.Parameter(torch.zeros((len(self.multi_scale_factors),), dtype=torch.float32))
        else:
            self.register_parameter("scale_logits", None)
        self.input_fusion = PriorFusionBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, gate_init=gate_init)
        self.frame_fusions = nn.ModuleList(
            PriorFusionBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, gate_init=gate_init)
            for _ in range(depth)
        )
        self.global_fusions = nn.ModuleList(
            PriorFusionBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, gate_init=gate_init)
            for _ in range(depth)
        )
        self.summary_proj = (
            nn.Sequential(
                nn.Linear(summary_in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            if summary_in_channels > 0
            else None
        )
        self.global_summary_fusions = nn.ModuleList(
            SummaryTokenFusionBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, gate_init=gate_init)
            for _ in range(depth)
        )

    def project_prior_maps(self, prior_maps: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            prior_maps: Tensor of shape [B, S, C, H, W].
            target_hw: Target patch-grid spatial size as (H_tokens, W_tokens).

        Returns:
            Tensor of shape [B * S, H_tokens * W_tokens, embed_dim].
        """
        if prior_maps.ndim != 5:
            raise ValueError(f"Expected prior_maps with 5 dims [B, S, C, H, W], got {prior_maps.shape}")

        batch_size, seq_len, channels, _, _ = prior_maps.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} prior channels, got {channels}. "
                "Please align model.human_prior_channels with the exported prior maps."
            )

        prior_maps = prior_maps.reshape(batch_size * seq_len, channels, prior_maps.shape[-2], prior_maps.shape[-1])
        branch_features = []
        for branch_idx, scale_factor in enumerate(self.multi_scale_factors):
            if branch_idx == 0:
                scaled_maps = F.interpolate(
                    prior_maps,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                branch_feature = self.proj(scaled_maps)
            else:
                scaled_hw = (
                    max(1, int(round(target_hw[0] / float(scale_factor)))),
                    max(1, int(round(target_hw[1] / float(scale_factor)))),
                )
                scaled_maps = F.interpolate(
                    prior_maps,
                    size=scaled_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                branch_feature = self.scale_projs[branch_idx - 1](scaled_maps)
                if scaled_hw != target_hw:
                    branch_feature = F.interpolate(
                        branch_feature,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
            branch_features.append(branch_feature)

        if len(branch_features) == 1:
            fused_features = branch_features[0]
        else:
            scale_weights = torch.softmax(self.scale_logits[: len(branch_features)], dim=0).to(
                dtype=branch_features[0].dtype,
                device=branch_features[0].device,
            )
            fused_features = torch.zeros_like(branch_features[0])
            for weight, feature in zip(scale_weights, branch_features):
                fused_features = fused_features + weight * feature

        prior_tokens = fused_features.flatten(2).transpose(1, 2).contiguous()
        return prior_tokens

    def fuse_input_tokens(self, patch_tokens: torch.Tensor, prior_tokens: torch.Tensor) -> torch.Tensor:
        return self.input_fusion(patch_tokens, prior_tokens)

    def fuse_frame_tokens(self, patch_tokens: torch.Tensor, prior_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.frame_fusions[layer_idx](patch_tokens, prior_tokens)

    def fuse_global_tokens(self, patch_tokens: torch.Tensor, prior_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.global_fusions[layer_idx](patch_tokens, prior_tokens)

    def project_summary_tokens(self, summary_tokens: torch.Tensor) -> torch.Tensor:
        if self.summary_proj is None:
            raise ValueError("summary_tokens were provided, but summary_in_channels=0 for HumanPriorAdapter.")
        if summary_tokens.ndim != 4:
            raise ValueError(
                f"Expected summary_tokens with 4 dims [B, S, T, C], got {summary_tokens.shape}"
            )

        batch_size, seq_len, token_count, channels = summary_tokens.shape
        if channels != self.summary_in_channels:
            raise ValueError(
                f"Expected {self.summary_in_channels} summary token channels, got {channels}. "
                "Please align model.human_prior_summary_channels with the exported summary tokens."
            )
        projected = self.summary_proj(summary_tokens.reshape(batch_size * seq_len, token_count, channels))
        return projected.contiguous()

    def fuse_global_summary_tokens(
        self,
        patch_tokens: torch.Tensor,
        summary_tokens: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.global_summary_fusions[layer_idx](patch_tokens, summary_tokens)

    def forward(self, prior_maps: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        return self.project_prior_maps(prior_maps=prior_maps, target_hw=target_hw)
