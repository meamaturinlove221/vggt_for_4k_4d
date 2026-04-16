# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Dict, Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
        - Optionally, normalization metadata used to transform auxiliary priors
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    normalization_aux: Dict[str, torch.Tensor] = {
        "reference_rotation": extrinsics[:, 0, :3, :3],
        "reference_translation": extrinsics[:, 0, :3, 3],
        "avg_scale": torch.ones(B, device=device, dtype=extrinsics.dtype),
    }


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        normalization_aux["avg_scale"] = avg_scale


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        outputs = (new_extrinsics[:, :, :3], cam_points, new_world_points, depths)
        if return_aux:
            return (*outputs, normalization_aux)
        return outputs

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)


    outputs = (new_extrinsics, new_cam_points, new_world_points, new_depths)
    if return_aux:
        return (*outputs, normalization_aux)
    return outputs


def normalize_world_points_with_reference(
    world_points: torch.Tensor,
    reference_rotation: torch.Tensor,
    reference_translation: torch.Tensor,
    avg_scale: Optional[torch.Tensor] = None,
    name: str = "world_points",
) -> torch.Tensor:
    """
    Applies the same world-to-reference-camera transform used for the GT points.
    """
    check_valid_tensor(world_points, name)

    # Match normalize_camera_extrinsics_and_points_batch(): world points are
    # [B, S, H, W, 3], so only two singleton dims are needed on the rotation
    # for broadcasted matmul. Adding a third one incorrectly creates a 6D tensor.
    rotation = reference_rotation.transpose(-1, -2).unsqueeze(1).unsqueeze(2)
    translation = reference_translation.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    normalized_world_points = (world_points @ rotation) + translation

    if avg_scale is not None:
        normalized_world_points = normalized_world_points / avg_scale.view(-1, 1, 1, 1, 1)

    return check_and_fix_inf_nan(normalized_world_points, f"normalized_{name}", hard_max=None)


def normalize_depths_with_scale(
    depths: torch.Tensor,
    avg_scale: torch.Tensor,
    name: str = "depths",
) -> torch.Tensor:
    """
    Applies the same per-sample scale normalization used for GT depths.
    """
    check_valid_tensor(depths, name)
    normalized_depths = depths / avg_scale.view(-1, *([1] * (depths.ndim - 1)))
    return check_and_fix_inf_nan(normalized_depths, f"normalized_{name}", hard_max=None)


def normalize_cam_points_with_scale(
    cam_points: torch.Tensor,
    avg_scale: torch.Tensor,
    name: str = "cam_points",
) -> torch.Tensor:
    """
    Applies the same per-sample scale normalization used for GT camera-space points.
    """
    check_valid_tensor(cam_points, name)
    normalized_cam_points = cam_points / avg_scale.view(-1, *([1] * (cam_points.ndim - 1)))
    return check_and_fix_inf_nan(normalized_cam_points, f"normalized_{name}", hard_max=None)



