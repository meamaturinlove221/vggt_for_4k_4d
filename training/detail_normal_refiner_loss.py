from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_channel_first_normal(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(tensor.shape)}")
    if tensor.shape[1] == 3:
        return F.normalize(tensor.float(), p=2, dim=1, eps=1e-6)
    if tensor.shape[-1] == 3:
        return F.normalize(tensor.permute(0, 3, 1, 2).float(), p=2, dim=1, eps=1e-6)
    raise ValueError(f"Expected normal tensor with 3 channels, got {tuple(tensor.shape)}")


def _to_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
    return (mask > 0.5).float()


def make_boundary_weight(mask: torch.Tensor, kernel_size: int = 7, boundary_boost: float = 2.5) -> torch.Tensor:
    mask = _to_mask(mask)
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = ((dilated - eroded) > 0.0).float()
    return 1.0 + boundary * max(0.0, float(boundary_boost) - 1.0)


def cosine_normal_loss(
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = _to_channel_first_normal(pred_normal)
    target = _to_channel_first_normal(target_normal)
    valid = _to_mask(valid_mask)
    if weight is None:
        weight = torch.ones_like(valid)
    else:
        weight = weight.float()
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)
    effective = valid * weight
    denom = effective.sum().clamp(min=1.0)
    dot = torch.sum(pred * target, dim=1, keepdim=True)
    loss = (1.0 - dot) * effective
    return loss.sum() / denom


def mask_restricted_l1_loss(
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = _to_channel_first_normal(pred_normal)
    target = _to_channel_first_normal(target_normal)
    valid = _to_mask(valid_mask)
    if weight is None:
        weight = torch.ones_like(valid)
    else:
        weight = weight.float()
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)
    effective = valid * weight
    denom = effective.sum().clamp(min=1.0)
    loss = torch.abs(pred - target).sum(dim=1, keepdim=True) * effective
    return loss.sum() / denom


def _spatial_gradient(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = tensor[..., :, 1:] - tensor[..., :, :-1]
    grad_y = tensor[..., 1:, :] - tensor[..., :-1, :]
    return grad_x, grad_y


def edge_aware_normal_loss(
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    rgb: torch.Tensor,
    weight: torch.Tensor | None = None,
    edge_sensitivity: float = 6.0,
) -> torch.Tensor:
    pred = _to_channel_first_normal(pred_normal)
    target = _to_channel_first_normal(target_normal)
    valid = _to_mask(valid_mask)
    rgb = rgb.float()
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb [B, 3, H, W], got {tuple(rgb.shape)}")
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    if weight is None:
        weight = torch.ones_like(valid)
    else:
        weight = weight.float()
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)

    pred_grad_x, pred_grad_y = _spatial_gradient(pred)
    target_grad_x, target_grad_y = _spatial_gradient(target)
    rgb_grad_x, rgb_grad_y = _spatial_gradient(rgb)
    valid_x, valid_y = _spatial_gradient(valid)
    valid_x = (valid_x.abs() < 1e-6).float() * valid[..., :, 1:] * valid[..., :, :-1]
    valid_y = (valid_y.abs() < 1e-6).float() * valid[..., 1:, :] * valid[..., :-1, :]

    image_weight_x = torch.exp(-edge_sensitivity * rgb_grad_x.abs().mean(dim=1, keepdim=True))
    image_weight_y = torch.exp(-edge_sensitivity * rgb_grad_y.abs().mean(dim=1, keepdim=True))

    weight_x = weight[..., :, 1:] * weight[..., :, :-1] * image_weight_x * valid_x
    weight_y = weight[..., 1:, :] * weight[..., :-1, :] * image_weight_y * valid_y

    denom = weight_x.sum() + weight_y.sum()
    denom = denom.clamp(min=1.0)
    loss_x = (pred_grad_x - target_grad_x).abs().sum(dim=1, keepdim=True) * weight_x
    loss_y = (pred_grad_y - target_grad_y).abs().sum(dim=1, keepdim=True) * weight_y
    return (loss_x.sum() + loss_y.sum()) / denom


def compute_detail_normal_refiner_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    cosine_weight: float = 1.0,
    edge_weight: float = 0.25,
    mask_weight: float = 0.10,
    boundary_boost: float = 2.5,
) -> dict[str, torch.Tensor]:
    refined_normal = predictions["refined_normal"]
    teacher_normal = batch["teacher_normal"]
    teacher_mask = batch.get("teacher_mask", batch["human_mask"])
    rgb = batch["rgb"]
    boundary_weight = make_boundary_weight(batch["human_mask"], boundary_boost=boundary_boost)

    loss_cos = cosine_normal_loss(refined_normal, teacher_normal, teacher_mask, weight=boundary_weight)
    loss_edge = edge_aware_normal_loss(refined_normal, teacher_normal, teacher_mask, rgb, weight=boundary_weight)
    loss_mask = mask_restricted_l1_loss(refined_normal, teacher_normal, teacher_mask, weight=boundary_weight)
    total = cosine_weight * loss_cos + edge_weight * loss_edge + mask_weight * loss_mask

    metrics = {
        "loss_detail_normal_total": total,
        "loss_detail_normal_cosine": loss_cos,
        "loss_detail_normal_edge": loss_edge,
        "loss_detail_normal_mask_restricted": loss_mask,
        "metric_boundary_weight_mean": boundary_weight.mean(),
    }

    if "hairline_mask" in batch:
        metrics["metric_hairline_cosine"] = cosine_normal_loss(
            refined_normal,
            teacher_normal,
            batch["hairline_mask"],
            weight=boundary_weight,
        )
    if "ear_band_mask" in batch:
        metrics["metric_ear_band_cosine"] = cosine_normal_loss(
            refined_normal,
            teacher_normal,
            batch["ear_band_mask"],
            weight=boundary_weight,
        )
    return metrics
