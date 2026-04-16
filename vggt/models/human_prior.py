# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanPriorAdapter(nn.Module):
    """
    Projects optional image-space human prior maps to the patch-token space.

    The module keeps the integration lightweight:
    - resize prior maps to the patch grid
    - project them to the token embedding dimension
    - fuse them into the patch tokens with a learnable gate
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_dim: int = 64,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, prior_maps: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
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
        prior_maps = F.interpolate(
            prior_maps,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        prior_tokens = self.proj(prior_maps)
        prior_tokens = prior_tokens.flatten(2).transpose(1, 2).contiguous()
        return prior_tokens * self.gate
