from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, dim_out // 8 or 1)), num_channels=dim_out),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, dim_out // 8 or 1)), num_channels=dim_out),
            nn.GELU(),
        )
        self.skip = None
        if stride != 1 or dim_in != dim_out:
            self.skip = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        return self.block(x) + residual


class UpBlock(nn.Module):
    def __init__(self, dim_in: int, dim_skip: int, dim_out: int):
        super().__init__()
        self.fuse = ConvBlock(dim_in + dim_skip, dim_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class DetailNormalRefiner(nn.Module):
    """
    Image-aligned residual refiner on top of coarse prior normals.

    Inputs:
    - rgb: [B, 3, H, W]
    - coarse_normal: [B, 3, H, W]
    - human_mask: [B, 1, H, W] or [B, H, W]
    """

    def __init__(
        self,
        in_channels: int = 7,
        base_dim: int = 32,
        residual_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)

        self.stem = ConvBlock(in_channels, base_dim)
        self.down1 = ConvBlock(base_dim, base_dim * 2, stride=2)
        self.down2 = ConvBlock(base_dim * 2, base_dim * 4, stride=2)
        self.bottleneck = ConvBlock(base_dim * 4, base_dim * 4)
        self.up1 = UpBlock(base_dim * 4, base_dim * 2, base_dim * 2)
        self.up2 = UpBlock(base_dim * 2, base_dim, base_dim)
        self.residual_head = nn.Sequential(
            ConvBlock(base_dim, base_dim),
            nn.Conv2d(base_dim, 3, kernel_size=1),
        )

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Expected mask [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
        return (mask > 0.5).float()

    @staticmethod
    def _normalize_normal(normal: torch.Tensor) -> torch.Tensor:
        if normal.ndim != 4 or normal.shape[1] != 3:
            raise ValueError(f"Expected normal [B, 3, H, W], got {tuple(normal.shape)}")
        return F.normalize(normal.float(), p=2, dim=1, eps=1e-6)

    def forward(
        self,
        rgb: torch.Tensor,
        coarse_normal: torch.Tensor,
        human_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected rgb [B, 3, H, W], got {tuple(rgb.shape)}")
        rgb = rgb.float()
        mask = self._normalize_mask(human_mask)
        coarse = self._normalize_normal(coarse_normal)

        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        x = torch.cat([rgb, coarse, mask], dim=1)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = self.bottleneck(s2)
        x = self.up1(x, s1)
        x = self.up2(x, s0)

        residual = torch.tanh(self.residual_head(x)) * self.residual_scale
        refined = F.normalize(coarse + residual * mask, p=2, dim=1, eps=1e-6)
        refined = refined * mask + coarse * (1.0 - mask)

        return {
            "refined_normal": refined,
            "normal_residual": residual,
            "coarse_normal": coarse,
            "human_mask": mask,
        }
