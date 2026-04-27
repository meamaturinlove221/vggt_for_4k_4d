from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.detail_normal_refiner import DetailNormalRefiner  # noqa: E402


COARSE_NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")
SUMMARY_NORMAL_CHANNELS = ("smplx_summary_cam_nx", "smplx_summary_cam_ny", "smplx_summary_cam_nz")
SUMMARY_POSITION_CHANNELS = (
    "smplx_summary_posed_cam_x",
    "smplx_summary_posed_cam_y",
    "smplx_summary_posed_cam_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a self-contained training case with ROI refined normals. "
            "This updates case inputs.prior_maps and targets.prior_normals so the main trainer can see "
            "stronger normal conditioning and stronger normal supervision."
        )
    )
    parser.add_argument("--dataset-npz", required=True, help="ROI dataset export with roi_box_xyxy/view_index metadata")
    parser.add_argument("--case-dir", required=True, help="Source training case directory")
    parser.add_argument("--output-case-dir", required=True, help="Patched copied case directory")
    parser.add_argument("--normal-source", choices=("refined", "teacher"), default="refined")
    parser.add_argument("--checkpoint", help="Refiner checkpoint required when --normal-source refined")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--patch-mask-source",
        choices=("coarse_valid", "human_mask", "coarse_or_human", "teacher_mask"),
        default="coarse_or_human",
    )
    parser.add_argument(
        "--summary-update",
        choices=("none", "mean_view_delta", "projected_token_sample"),
        default="none",
        help="Optional heuristic update for prior_summary_tokens normal channels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resize_float_hw(array: np.ndarray, out_h: int, out_w: int, *, is_mask: bool = False) -> np.ndarray:
    arr = np.asarray(array)
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR

    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        pil = pil.resize((int(out_w), int(out_h)), mode)
        resized = np.asarray(pil, dtype=np.float32)
        return (resized > 0.5) if is_mask else resized

    if arr.ndim == 3 and arr.shape[-1] >= 1:
        channels = []
        for channel_idx in range(arr.shape[-1]):
            pil = Image.fromarray(arr[..., channel_idx].astype(np.float32), mode="F")
            pil = pil.resize((int(out_w), int(out_h)), mode)
            channel = np.asarray(pil, dtype=np.float32)
            if is_mask:
                channel = (channel > 0.5).astype(np.float32)
            channels.append(channel)
        stacked = np.stack(channels, axis=-1)
        if is_mask:
            return stacked > 0.5
        return stacked.astype(np.float32)

    raise ValueError(f"Unsupported shape for resize: {arr.shape}")


def _normalize_normals(normals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normalized = normals / np.clip(norms, 1e-6, None)
    if mask is not None:
        normalized = normalized.copy()
        normalized[~np.asarray(mask, dtype=bool)] = 0.0
    return normalized.astype(np.float32)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "rgb",
            "human_mask",
            "coarse_prior_normal",
            "coarse_prior_valid_mask",
            "teacher_normal",
            "teacher_mask",
            "roi_box_xyxy",
            "view_index",
            "view_name",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise KeyError(f"Dataset NPZ is missing required keys: {missing}")
        return {key: np.array(payload[key]) for key in payload.files}


def _predict_refined_normals(
    *,
    checkpoint_path: Path,
    device: torch.device,
    dataset_payload: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = DetailNormalRefiner(
        base_dim=int(ckpt_args.get("base_dim", 32)),
        residual_scale=float(ckpt_args.get("residual_scale", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rgb = torch.from_numpy(dataset_payload["rgb"]).permute(0, 3, 1, 2).float() / 255.0
    human_mask = torch.from_numpy(dataset_payload["human_mask"].astype(np.float32))[:, None]
    coarse = torch.from_numpy(dataset_payload["coarse_prior_normal"]).permute(0, 3, 1, 2).float()

    refined_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(rgb.shape[0]), int(batch_size)):
            end = min(int(rgb.shape[0]), start + int(batch_size))
            preds = model(
                rgb=rgb[start:end].to(device),
                coarse_normal=coarse[start:end].to(device),
                human_mask=human_mask[start:end].to(device),
            )
            refined = preds["refined_normal"].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            refined_batches.append(refined)
    return np.concatenate(refined_batches, axis=0)


def _copy_case_tree(case_dir: Path, output_case_dir: Path, overwrite: bool) -> None:
    if output_case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output case dir already exists: {output_case_dir}")
        shutil.rmtree(output_case_dir)
    shutil.copytree(case_dir, output_case_dir)


def _load_case_payloads(case_dir: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    case_manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    with np.load(case_dir / "inputs.npz", allow_pickle=False) as inputs_payload:
        inputs = {key: np.array(inputs_payload[key]) for key in inputs_payload.files}
    with np.load(case_dir / "targets.npz", allow_pickle=False) as targets_payload:
        targets = {key: np.array(targets_payload[key]) for key in targets_payload.files}
    return case_manifest, inputs, targets


def _lookup_indices(channel_names: list[str], target_names: tuple[str, ...]) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(channel_names)}
    return [int(lookup[name]) for name in target_names]


def _project_cam_points_to_pixels(cam_points: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam_points = np.asarray(cam_points, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if cam_points.ndim != 2 or cam_points.shape[1] != 3:
        raise ValueError(f"Expected cam_points [N, 3], got {cam_points.shape}")
    z = cam_points[:, 2]
    valid = np.isfinite(cam_points).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    uv = np.zeros((cam_points.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        points_valid = cam_points[valid]
        uvw = (intrinsic @ points_valid.T).T
        uv_valid = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
        uv[valid] = uv_valid.astype(np.float32)
        valid[valid] = np.isfinite(uv_valid).all(axis=1)
    return uv, valid


def _select_patch_mask(dataset_payload: dict[str, np.ndarray], sample_idx: int, patch_mask_source: str) -> np.ndarray:
    coarse_valid = np.asarray(dataset_payload["coarse_prior_valid_mask"][sample_idx], dtype=bool)
    human_mask = np.asarray(dataset_payload["human_mask"][sample_idx], dtype=bool)
    teacher_mask = np.asarray(dataset_payload["teacher_mask"][sample_idx], dtype=bool)
    if patch_mask_source == "coarse_valid":
        return coarse_valid
    if patch_mask_source == "human_mask":
        return human_mask
    if patch_mask_source == "teacher_mask":
        return teacher_mask
    return coarse_valid | human_mask | teacher_mask


def _patch_case(
    *,
    case_manifest: dict,
    inputs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    dataset_payload: dict[str, np.ndarray],
    patch_normals: np.ndarray,
    patch_mask_source: str,
    summary_update: str,
) -> dict[str, object]:
    prior_maps = np.array(inputs["prior_maps"], copy=True).astype(np.float32)
    prior_summary_tokens = np.array(inputs["prior_summary_tokens"], copy=True).astype(np.float32)
    prior_normals = np.array(targets["prior_normals"], copy=True).astype(np.float32)
    intrinsics = np.asarray(targets["intrinsics"], dtype=np.float32)

    input_channel_names = [str(name) for name in case_manifest["prior_input_meta"]["channel_names"]]
    summary_channel_names = [str(name) for name in case_manifest["prior_input_meta"]["summary_channel_names"]]
    dense_normal_indices = _lookup_indices(input_channel_names, COARSE_NORMAL_CHANNELS)
    summary_normal_indices = _lookup_indices(summary_channel_names, SUMMARY_NORMAL_CHANNELS)
    summary_position_indices = (
        _lookup_indices(summary_channel_names, SUMMARY_POSITION_CHANNELS)
        if summary_update == "projected_token_sample"
        else []
    )

    roi_boxes = np.asarray(dataset_payload["roi_box_xyxy"], dtype=np.int32)
    view_indices = np.asarray(dataset_payload["view_index"], dtype=np.int32)
    view_names = np.asarray(dataset_payload["view_name"])
    coarse = np.asarray(dataset_payload["coarse_prior_normal"], dtype=np.float32)

    patch_records: list[dict[str, object]] = []
    for sample_idx in range(patch_normals.shape[0]):
        view_idx = int(view_indices[sample_idx])
        x0, y0, x1, y1 = [int(v) for v in roi_boxes[sample_idx].tolist()]
        roi_h = max(1, y1 - y0)
        roi_w = max(1, x1 - x0)

        patch_mask = _select_patch_mask(dataset_payload, sample_idx, patch_mask_source)
        refined_roi = _resize_float_hw(patch_normals[sample_idx], roi_h, roi_w, is_mask=False).astype(np.float32)
        refined_mask = _resize_float_hw(patch_mask.astype(np.float32), roi_h, roi_w, is_mask=True)
        if refined_mask.ndim == 3:
            refined_mask = refined_mask[..., 0]
        refined_roi = _normalize_normals(refined_roi, refined_mask)

        coarse_roi = _resize_float_hw(coarse[sample_idx], roi_h, roi_w, is_mask=False).astype(np.float32)
        coarse_roi = _normalize_normals(coarse_roi, refined_mask)

        input_patch = prior_maps[view_idx, dense_normal_indices, y0:y1, x0:x1].transpose(1, 2, 0).astype(np.float32)
        target_patch = prior_normals[view_idx, y0:y1, x0:x1].astype(np.float32)
        input_patch[refined_mask] = refined_roi[refined_mask]
        target_patch[refined_mask] = refined_roi[refined_mask]
        prior_maps[view_idx, dense_normal_indices, y0:y1, x0:x1] = input_patch.transpose(2, 0, 1)
        prior_normals[view_idx, y0:y1, x0:x1] = target_patch

        summary_delta = np.zeros(3, dtype=np.float32)
        if np.any(refined_mask):
            summary_delta = refined_roi[refined_mask].mean(axis=0) - coarse_roi[refined_mask].mean(axis=0)
        summary_updated_tokens = 0
        if summary_update == "mean_view_delta":
            token_view = prior_summary_tokens[view_idx]
            token_view[:, summary_normal_indices] += summary_delta[None, :]
            prior_summary_tokens[view_idx] = token_view
            summary_updated_tokens = int(token_view.shape[0])
        elif summary_update == "projected_token_sample":
            token_view = prior_summary_tokens[view_idx]
            token_positions = token_view[:, summary_position_indices]
            uv, valid = _project_cam_points_to_pixels(token_positions, intrinsics[view_idx])
            inside = (
                valid
                & (uv[:, 0] >= float(x0))
                & (uv[:, 0] < float(x1))
                & (uv[:, 1] >= float(y0))
                & (uv[:, 1] < float(y1))
            )
            if np.any(inside):
                local_x = np.clip(np.round(uv[inside, 0] - float(x0)).astype(np.int32), 0, roi_w - 1)
                local_y = np.clip(np.round(uv[inside, 1] - float(y0)).astype(np.int32), 0, roi_h - 1)
                token_valid = refined_mask[local_y, local_x]
                if np.any(token_valid):
                    sampled_normals = refined_roi[local_y[token_valid], local_x[token_valid]]
                    inside_indices = np.flatnonzero(inside)[token_valid]
                    token_view[inside_indices[:, None], summary_normal_indices] = sampled_normals.astype(np.float32)
                    prior_summary_tokens[view_idx] = token_view
                    summary_updated_tokens = int(token_valid.sum())

        patch_records.append(
            {
                "sample_index": int(sample_idx),
                "view_index": view_idx,
                "view_name": str(view_names[sample_idx]),
                "roi_box_xyxy": [x0, y0, x1, y1],
                "changed_pixels": int(np.asarray(refined_mask, dtype=bool).sum()),
                "summary_delta_mean": summary_delta.tolist(),
                "summary_updated_tokens": int(summary_updated_tokens),
            }
        )

    inputs["prior_maps"] = prior_maps.astype(np.float32)
    inputs["prior_summary_tokens"] = prior_summary_tokens.astype(np.float32)
    targets["prior_normals"] = prior_normals.astype(np.float32)
    return {
        "patch_records": patch_records,
        "dense_normal_indices": dense_normal_indices,
        "summary_normal_indices": summary_normal_indices,
        "summary_position_indices": summary_position_indices,
    }


def main() -> int:
    args = parse_args()
    dataset_npz = Path(args.dataset_npz).expanduser().resolve()
    case_dir = Path(args.case_dir).expanduser().resolve()
    output_case_dir = Path(args.output_case_dir).expanduser().resolve()
    device = _resolve_device(args.device)

    if args.normal_source == "refined" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --normal-source refined")

    dataset_payload = _load_dataset(dataset_npz)
    if args.normal_source == "teacher":
        patch_normals = np.asarray(dataset_payload["teacher_normal"], dtype=np.float32)
    else:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        patch_normals = _predict_refined_normals(
            checkpoint_path=checkpoint_path,
            device=device,
            dataset_payload=dataset_payload,
            batch_size=int(args.batch_size),
        )

    _copy_case_tree(case_dir, output_case_dir, overwrite=bool(args.overwrite))
    case_manifest, inputs, targets = _load_case_payloads(output_case_dir)
    patch_meta = _patch_case(
        case_manifest=case_manifest,
        inputs=inputs,
        targets=targets,
        dataset_payload=dataset_payload,
        patch_normals=patch_normals,
        patch_mask_source=str(args.patch_mask_source),
        summary_update=str(args.summary_update),
    )

    np.savez_compressed(output_case_dir / "inputs.npz", **inputs)
    np.savez_compressed(output_case_dir / "targets.npz", **targets)

    case_manifest["normal_patch_meta"] = {
        "normal_source": str(args.normal_source),
        "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint).expanduser().resolve()),
        "dataset_npz": str(dataset_npz),
        "patch_mask_source": str(args.patch_mask_source),
        "summary_update": str(args.summary_update),
        "notes": [
            "inputs.prior_maps dense normal channels were patched in ROI regions",
            "targets.prior_normals were patched in matching ROI regions",
            "summary token normal channels are only updated when summary_update != none",
        ],
    }
    (output_case_dir / "case_manifest.json").write_text(json.dumps(case_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "message": "training case patched with refined ROI normals",
        "dataset_npz": str(dataset_npz),
        "case_dir": str(case_dir),
        "output_case_dir": str(output_case_dir),
        "normal_source": str(args.normal_source),
        "device": str(device),
        "patch_mask_source": str(args.patch_mask_source),
        "summary_update": str(args.summary_update),
        "num_samples": int(patch_normals.shape[0]),
        "dense_normal_indices": patch_meta["dense_normal_indices"],
        "summary_normal_indices": patch_meta["summary_normal_indices"],
        "summary_position_indices": patch_meta["summary_position_indices"],
        "patch_records": patch_meta["patch_records"],
    }
    (output_case_dir / "normal_patch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
