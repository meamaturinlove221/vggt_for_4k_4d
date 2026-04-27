from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
for root in (REPO_ROOT, TRAINING_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools.dna_4k4d import (  # noqa: E402
    SUBSET_NAME,
    build_context,
    materialize_rgb_cams_smc,
    normalize_camera_id,
    require_h5py,
)
from tools.export_4k4d_human_prior import (  # noqa: E402
    infer_image_hw,
    load_keypoints_2d,
    load_keypoints_3d,
    load_mask,
    load_smplx_params,
    pad_resize_map,
    render_keypoint_heatmap,
    resolve_annotations_smc,
)
from tools.smplx_numpy import (  # noqa: E402
    DEFAULT_BODY_PART_COUNT,
    DEFAULT_BODY_PART_EMBED_DIM,
    DEFAULT_VERTEX_ID_EMBED_DIM,
    build_smplx_vertex_features,
    build_surface_cluster_ids,
    compute_vertex_normals,
    compute_pose_aligned_vertex_features,
    forward_smplx_mesh,
    pool_vertex_features,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)


POINT_SOURCES = ("world_points", "depth_unprojection")

# Current V2 is already built around pose-aligned, vertex-derived dense surface priors,
# and now includes richer identity semantics needed by the upgraded condition branch.
CURRENT_V2_SURFACE_CAPABILITIES = [
    "pose_aligned_smplx_driver",
    "vertex_derived_dense_surface_maps",
    "vertex_id_embedding",
    "body_part_embedding",
    "skinning_weight_summary",
    "per_view_projected_surface_condition",
    "summary_tokens_from_vertex_pooling",
    "layerwise_human_prior_adapter_ready",
    "multi_scale_surface_map_projection_ready",
    "output_side_depth_point_supervision",
]

NEXT_STAGE_SURFACE_ENHANCEMENTS = [
    "occlusion-aware dynamic surface features",
    "learned body-part / vertex identity codes",
    "stronger pose-noise schedules",
    "deeper multi-scale fusion policies",
]

NEXT_STAGE_ENHANCEMENT_GOALS = [
    "进一步增强身份语义表达",
    "继续提高对 pose 误差的鲁棒性",
    "让人体条件在不同层级位置上更充分地进入 backbone",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one self-contained 4K4D pseudo-training case bundle.")
    parser.add_argument("--scene-dir", required=True, help="Scene directory exported by tools/export_4k4d_scene.py.")
    parser.add_argument("--predictions-npz", required=True, help="VGGT predictions.npz aligned with the scene view order.")
    parser.add_argument("--output-dir", required=True, help="Self-contained case output directory.")
    parser.add_argument(
        "--external-prior-bundle",
        help=(
            "Path to an imported external prior bundle manifest or directory produced by "
            "tools/import_external_smplx_params.py. When set, this overrides scene-local prior regeneration."
        ),
    )
    parser.add_argument("--dataset-root", help="4K4D dataset root. If omitted, uses dataset_root from scene_manifest.json.")
    parser.add_argument("--subset-name", default=SUBSET_NAME, help=f"Subset name. Default: {SUBSET_NAME}")
    parser.add_argument(
        "--point-source",
        choices=POINT_SOURCES,
        default="depth_unprojection",
        help="Pseudo 3D source: use teacher world_points or depth+camera unprojection.",
    )
    parser.add_argument("--mask-only", action="store_true", help="Use silhouette-only prior maps.")
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian sigma for 2D keypoint heatmaps.")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Minimum confidence for 2D keypoint heatmaps.")
    parser.add_argument(
        "--geometry-prior-source",
        choices=("auto", "smplx_mesh", "keypoints3d"),
        default="auto",
        help="Geometry prior source. 'auto' prefers SMPL-X mesh when model files are available.",
    )
    parser.add_argument("--smplx-model-dir", help="Directory containing SMPL-X model files such as SMPLX_NEUTRAL.npz.")
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--mesh-fill-knn", type=int, default=4, help="KNN used to densify SMPL-X mesh priors inside the silhouette.")
    parser.add_argument("--summary-token-count", type=int, default=16, help="Number of pooled SMPL-X human summary tokens per view.")
    parser.add_argument("--vertex-id-dim", type=int, default=DEFAULT_VERTEX_ID_EMBED_DIM, help="Dimensionality of deterministic vertex identity embeddings.")
    parser.add_argument("--body-part-dim", type=int, default=DEFAULT_BODY_PART_EMBED_DIM, help="Dimensionality of deterministic body-part embeddings.")
    parser.add_argument("--body-part-count", type=int, default=DEFAULT_BODY_PART_COUNT, help="Number of coarse body-part groups derived from SMPL-X joint influence.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def closed_form_inverse_se3_numpy(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_map_to_point_map_numpy(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    world_points = []
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics_cam)
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1).astype(np.float32)
        intrinsic = intrinsics_cam[frame_idx].astype(np.float32)

        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]

        x_cam = (u - cu) * depth / fu
        y_cam = (v - cv) * depth / fv
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)

        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))

    return np.stack(world_points, axis=0)


def depth_to_cam_coords_points_numpy(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def load_scene_manifest(scene_dir: Path) -> dict:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found under {scene_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_scene_local_prior_bundle_path(scene_dir: Path, scene_manifest: dict) -> Path | None:
    candidates: list[Path] = []
    prior_maps_file = scene_manifest.get("prior_maps_file")
    if prior_maps_file:
        prior_path = Path(str(prior_maps_file)).expanduser()
        candidates.append(prior_path if prior_path.is_absolute() else scene_dir / prior_path)
    candidates.append(scene_dir / "prior_maps.npz")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            normalized = candidate.resolve(strict=False)
        except OSError:
            normalized = candidate
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            return normalized
    return None


def resolve_preprocess_variant_name(scene_manifest: dict) -> str | None:
    variant = scene_manifest.get("preprocess_variant")
    if isinstance(variant, str) and variant:
        return variant

    summary = scene_manifest.get("preprocess_variant_summary")
    if isinstance(summary, dict):
        variant = summary.get("variant")
        if isinstance(variant, str) and variant:
            return variant

    for view in scene_manifest.get("exported_views", []):
        variant = view.get("preprocess_variant")
        if isinstance(variant, str) and variant:
            return variant
    return None


def _coerce_string_list(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(item) for item in values]
    array = np.asarray(values)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.tolist()]


def load_scene_local_prior_bundle(
    scene_dir: Path,
    scene_manifest: dict,
    *,
    target_size: int,
    expected_view_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, object], Path]:
    prior_bundle_path = resolve_scene_local_prior_bundle_path(scene_dir, scene_manifest)
    preprocess_variant = resolve_preprocess_variant_name(scene_manifest)
    if prior_bundle_path is None:
        if preprocess_variant is not None:
            raise FileNotFoundError(
                f"Scene {scene_dir} declares preprocess variant '{preprocess_variant}' but has no scene-local prior bundle."
            )
        raise FileNotFoundError(f"No scene-local prior bundle found under {scene_dir}")

    with np.load(prior_bundle_path, allow_pickle=False) as prior_payload:
        if "prior_maps" not in prior_payload or "prior_mask" not in prior_payload:
            raise KeyError(f"{prior_bundle_path} must contain `prior_maps` and `prior_mask`.")

        prior_maps = np.asarray(prior_payload["prior_maps"], dtype=np.float32)
        prior_mask = np.asarray(prior_payload["prior_mask"], dtype=bool)
        prior_summary_tokens = None
        if "prior_summary_tokens" in prior_payload:
            summary_tokens = np.asarray(prior_payload["prior_summary_tokens"], dtype=np.float32)
            if summary_tokens.size > 0 and summary_tokens.ndim >= 2 and summary_tokens.shape[1] > 0:
                prior_summary_tokens = summary_tokens

        channel_names = _coerce_string_list(
            prior_payload["prior_channels"] if "prior_channels" in prior_payload else scene_manifest.get("prior_channels")
        )
        summary_channel_names = _coerce_string_list(
            prior_payload["prior_summary_channels"]
            if "prior_summary_channels" in prior_payload
            else scene_manifest.get("prior_summary_channels")
        )

    if prior_maps.ndim != 4:
        raise ValueError(f"Expected scene-local prior maps [V, C, H, W], got {prior_maps.shape} from {prior_bundle_path}")
    if prior_maps.shape[0] != int(expected_view_count):
        raise ValueError(
            f"Scene-local prior maps view count {prior_maps.shape[0]} does not match manifest view count {expected_view_count}."
        )
    if prior_maps.shape[2:] != (int(target_size), int(target_size)):
        raise ValueError(
            f"Scene-local prior maps spatial size {prior_maps.shape[2:]} does not match prediction target {(target_size, target_size)}."
        )
    if prior_mask.shape != (prior_maps.shape[0], prior_maps.shape[2], prior_maps.shape[3]):
        raise ValueError(
            f"Scene-local prior mask shape {prior_mask.shape} does not match prior maps {prior_maps.shape}."
        )
    if prior_summary_tokens is not None and prior_summary_tokens.shape[0] != int(expected_view_count):
        raise ValueError(
            "Scene-local prior summary token view count "
            f"{prior_summary_tokens.shape[0]} does not match manifest view count {expected_view_count}."
        )

    if not channel_names:
        channel_names = [f"prior_channel_{idx:03d}" for idx in range(prior_maps.shape[1])]
    elif len(channel_names) != int(prior_maps.shape[1]):
        raise ValueError(
            f"Scene-local prior channel count {len(channel_names)} does not match prior map channels {prior_maps.shape[1]}."
        )

    if prior_summary_tokens is not None:
        if not summary_channel_names:
            summary_channel_names = [f"prior_summary_channel_{idx:03d}" for idx in range(prior_summary_tokens.shape[-1])]
        elif len(summary_channel_names) != int(prior_summary_tokens.shape[-1]):
            raise ValueError(
                "Scene-local prior summary channel count "
                f"{len(summary_channel_names)} does not match summary token channels {prior_summary_tokens.shape[-1]}."
            )
    else:
        summary_channel_names = []

    prior_input_meta = dict(scene_manifest.get("prior_input_meta", {}))
    prior_input_meta["channel_names"] = channel_names
    prior_input_meta["summary_channel_names"] = summary_channel_names
    prior_input_meta["channel_groups"] = _build_channel_group_meta(channel_names, summary_channel_names)
    prior_input_meta["scene_local_prior_bundle"] = {
        "path": str(prior_bundle_path),
        "preprocess_variant": preprocess_variant,
        "source": "scene_local_prior_bundle",
    }
    return prior_maps, prior_mask, prior_summary_tokens, prior_input_meta, prior_bundle_path


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _resolve_external_prior_bundle_manifest(bundle_ref: Path) -> Path:
    bundle_path = bundle_ref.expanduser()
    if bundle_path.is_dir():
        bundle_path = bundle_path / "external_prior_bundle_manifest.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"External prior bundle manifest not found: {bundle_path}")
    return bundle_path.resolve()


def load_external_prior_bundle(bundle_ref: Path, scene_manifest: dict) -> dict[str, object]:
    manifest_path = _resolve_external_prior_bundle_manifest(bundle_ref)
    bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    smplx_output = bundle_manifest.get("smplx_output") or "normalized_smplx_params.npz"
    smplx_path = Path(str(smplx_output)).expanduser()
    if not smplx_path.is_absolute():
        smplx_path = manifest_path.parent / smplx_path
    if not smplx_path.is_file():
        raise FileNotFoundError(f"External prior bundle SMPL-X payload not found: {smplx_path}")
    smplx_params = _load_npz_payload(smplx_path)

    expected_camera_ids = [
        normalize_camera_id(view["camera_id"])
        for view in scene_manifest.get("exported_views", [])
    ]
    camera_ids: list[str] = []
    camera_params: dict[str, dict[str, np.ndarray]] = {}
    camera_path: Path | None = None
    camera_output = bundle_manifest.get("camera_output")
    if camera_output:
        camera_path = Path(str(camera_output)).expanduser()
        if not camera_path.is_absolute():
            camera_path = manifest_path.parent / camera_path
        if not camera_path.is_file():
            raise FileNotFoundError(f"External prior bundle camera payload not found: {camera_path}")

        camera_payload = _load_npz_payload(camera_path)
        required_camera_keys = {"camera_ids", "intrinsics", "cam_to_world", "world_to_cam"}
        missing_camera_keys = sorted(required_camera_keys - set(camera_payload))
        if missing_camera_keys:
            raise KeyError(
                f"External prior bundle camera payload is missing required keys: {missing_camera_keys}"
            )

        camera_ids = [normalize_camera_id(camera_id) for camera_id in camera_payload["camera_ids"].tolist()]
        intrinsics = np.asarray(camera_payload["intrinsics"], dtype=np.float32)
        cam_to_world = np.asarray(camera_payload["cam_to_world"], dtype=np.float32)
        world_to_cam = np.asarray(camera_payload["world_to_cam"], dtype=np.float32)

        camera_count = len(camera_ids)
        if intrinsics.shape != (camera_count, 3, 3):
            raise ValueError(
                f"External prior intrinsics shape {intrinsics.shape} does not match expected {(camera_count, 3, 3)}."
            )
        if cam_to_world.shape != (camera_count, 4, 4):
            raise ValueError(
                f"External prior cam_to_world shape {cam_to_world.shape} does not match expected {(camera_count, 4, 4)}."
            )
        if world_to_cam.shape != (camera_count, 4, 4):
            raise ValueError(
                f"External prior world_to_cam shape {world_to_cam.shape} does not match expected {(camera_count, 4, 4)}."
            )
        if expected_camera_ids and camera_ids != expected_camera_ids:
            raise ValueError(
                "External prior camera ids do not match the scene manifest view order. "
                f"expected={expected_camera_ids}, got={camera_ids}"
            )

        camera_params = {
            camera_id: {
                "intrinsic": intrinsics[idx].astype(np.float32),
                "cam_to_world": cam_to_world[idx].astype(np.float32),
                "world_to_cam": world_to_cam[idx].astype(np.float32),
            }
            for idx, camera_id in enumerate(camera_ids)
        }

    resolved_meta = {
        "manifest_path": str(manifest_path),
        "smplx_path": str(smplx_path.resolve()),
        "camera_path": None if camera_path is None else str(camera_path.resolve()),
        "camera_ids_ordered": camera_ids,
        "smplx_keys": sorted(smplx_params.keys()),
        "format_version": int(bundle_manifest.get("format_version", 1)),
        "frame_idx": int(bundle_manifest.get("frame_idx", scene_manifest.get("frame_id", 0))),
    }
    return {
        "manifest": bundle_manifest,
        "smplx_params": smplx_params,
        "camera_params": camera_params,
        "resolved_meta": resolved_meta,
    }


def _make_smplx_vertex_feature_meta(enabled: bool, source: str, **extra: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "enabled": bool(enabled),
        "source": source,
        "design_stage": "v2_surface_condition_established",
        "current_v2_capabilities": list(CURRENT_V2_SURFACE_CAPABILITIES),
        "next_stage_enhancements": list(NEXT_STAGE_SURFACE_ENHANCEMENTS),
        "next_stage_goals": list(NEXT_STAGE_ENHANCEMENT_GOALS),
        "roadmap_note": (
            "These enhancements are next-stage upgrades for a stronger SMPL-X condition branch, "
            "not prerequisites for the current V2 to be considered valid."
        ),
    }
    meta.update(extra)
    return meta


def build_external_prior_stack(
    scene_manifest: dict,
    target_size: int,
    mask_only: bool,
    smplx_params: dict[str, np.ndarray],
    camera_params: dict[str, dict[str, np.ndarray]],
    external_bundle_meta: dict[str, object],
    smplx_model_dir: Path | None = None,
    smplx_gender: str = "neutral",
    mesh_fill_knn: int = 4,
    summary_token_count: int = 16,
    vertex_id_dim: int = DEFAULT_VERTEX_ID_EMBED_DIM,
    body_part_dim: int = DEFAULT_BODY_PART_EMBED_DIM,
    body_part_count: int = DEFAULT_BODY_PART_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray], np.ndarray | None, dict[str, object]]:
    view_info_by_camera = {
        normalize_camera_id(view["camera_id"]): view
        for view in scene_manifest["exported_views"]
    }

    prior_maps = []
    prior_masks = []
    prior_summary_tokens = []
    channel_names = ["silhouette"] + ([] if mask_only else ["keypoint_heatmap"])
    summary_channel_names: list[str] = []
    channel_group_meta = {"dense": {}, "summary": {}}

    world_vertices = None
    faces = None
    canonical_positions = None
    static_vertex_features = None
    canonical_scale = 1.0
    cluster_ids = None

    if smplx_model_dir is None:
        smplx_vertex_feature_meta = _make_smplx_vertex_feature_meta(
            False,
            "external_prior_bundle_missing_model_dir",
            external_prior_bundle=external_bundle_meta,
        )
    else:
        required_keys = {"betas", "fullpose"}
        if not required_keys.issubset(smplx_params):
            smplx_vertex_feature_meta = _make_smplx_vertex_feature_meta(
                False,
                "external_prior_bundle_missing_smplx_params",
                external_prior_bundle=external_bundle_meta,
                missing_keys=sorted(required_keys - set(smplx_params)),
            )
        elif not camera_params:
            smplx_vertex_feature_meta = _make_smplx_vertex_feature_meta(
                False,
                "external_prior_bundle_missing_camera_params",
                external_prior_bundle=external_bundle_meta,
            )
        else:
            model_path = resolve_smplx_model_path(smplx_model_dir, gender=smplx_gender)
            mesh = forward_smplx_mesh(
                model_path=model_path,
                betas=smplx_params["betas"],
                expression=smplx_params.get("expression"),
                fullpose=smplx_params["fullpose"],
                transl=smplx_params.get("transl"),
                scale=smplx_params.get("scale", 1.0),
            )
            vertex_feature_payload = build_smplx_vertex_features(
                model_path=model_path,
                betas=smplx_params["betas"],
                expression=smplx_params.get("expression"),
                vertex_id_dim=int(vertex_id_dim),
                body_part_dim=int(body_part_dim),
                body_part_count=int(body_part_count),
            )
            world_vertices = mesh["vertices"].astype(np.float32)
            faces = mesh["faces"].astype(np.int32)
            canonical_positions = np.asarray(vertex_feature_payload["canonical_positions"], dtype=np.float32)
            static_vertex_features = np.asarray(vertex_feature_payload["vertex_features"], dtype=np.float32)
            canonical_scale = float(np.asarray(vertex_feature_payload["canonical_scale"], dtype=np.float32))
            _, cluster_ids = build_surface_cluster_ids(
                canonical_positions,
                num_clusters=int(summary_token_count),
            )
            static_channel_names = list(vertex_feature_payload["channel_names"])
            dynamic_dense_channel_names = [
                "smplx_posed_cam_x",
                "smplx_posed_cam_y",
                "smplx_posed_cam_z",
                "smplx_cam_nx",
                "smplx_cam_ny",
                "smplx_cam_nz",
                "smplx_visible_mask",
            ]
            summary_channel_names = _prefixed_summary_channel_names(static_channel_names) + [
                "smplx_summary_posed_cam_x",
                "smplx_summary_posed_cam_y",
                "smplx_summary_posed_cam_z",
                "smplx_summary_cam_nx",
                "smplx_summary_cam_ny",
                "smplx_summary_cam_nz",
            ]
            channel_names.extend(static_channel_names + dynamic_dense_channel_names)
            channel_group_meta = _build_channel_group_meta(channel_names, summary_channel_names)
            smplx_vertex_feature_meta = _make_smplx_vertex_feature_meta(
                True,
                "external_smplx_bundle_pose_aligned_surface_prior",
                external_prior_bundle=external_bundle_meta,
                camera_source="external_prior_bundle",
                smplx_model_path=str(model_path),
                smplx_gender=smplx_gender,
                num_vertices=int(world_vertices.shape[0]),
                num_faces=int(faces.shape[0]),
                dense_channels=list(channel_names),
                summary_channels=list(summary_channel_names),
                channel_groups=channel_group_meta,
                summary_token_count=int(summary_token_count),
                fill_knn=int(mesh_fill_knn),
                vertex_id_dim=int(vertex_id_dim),
                body_part_dim=int(body_part_dim),
                body_part_count=int(body_part_count),
                views=[],
            )

    for view in scene_manifest["exported_views"]:
        mask = np.asarray(Image.open(view["mask_path"]).convert("L"), dtype=np.float32)
        mask_aligned = pad_resize_map((mask > 0).astype(np.float32), target_size, mode="nearest")
        mask_aligned = (mask_aligned > 0.5).astype(np.float32)

        channels = [mask_aligned]
        if not mask_only:
            channels.append(np.zeros_like(mask_aligned, dtype=np.float32))

        normalized_camera_id = normalize_camera_id(view["camera_id"])
        if (
            world_vertices is not None
            and faces is not None
            and canonical_positions is not None
            and static_vertex_features is not None
            and cluster_ids is not None
            and normalized_camera_id in camera_params
        ):
            pose_aligned = compute_pose_aligned_vertex_features(
                world_vertices=world_vertices,
                faces=faces,
                canonical_positions=canonical_positions,
                world_to_cam=camera_params[normalized_camera_id]["world_to_cam"],
                normalization_scale=canonical_scale,
            )
            dense_vertex_features = np.concatenate(
                [
                    static_vertex_features,
                    pose_aligned["normalized_cam_vertices"],
                    pose_aligned["cam_normals"],
                ],
                axis=1,
            ).astype(np.float32)
            intrinsic = align_intrinsics_for_scene_view(
                camera_params[normalized_camera_id]["intrinsic"],
                view_info_by_camera[normalized_camera_id],
                target_size=target_size,
            )
            _, _, _, feature_map, raster_mask, raster_meta = rasterize_world_mesh(
                world_vertices=world_vertices,
                faces=faces,
                world_to_cam=camera_params[normalized_camera_id]["world_to_cam"],
                intrinsic=intrinsic,
                image_hw=(target_size, target_size),
                silhouette_mask=mask_aligned > 0.5,
                fill_knn=max(1, int(mesh_fill_knn)),
                vertex_features=dense_vertex_features,
                return_vertex_features=True,
                return_raster_mask=True,
            )
            channels.extend(np.moveaxis(feature_map, -1, 0).astype(np.float32))
            channels.append(raster_mask.astype(np.float32))
            prior_summary_tokens.append(
                pool_vertex_features(
                    dense_vertex_features,
                    cluster_ids=cluster_ids,
                    num_clusters=int(summary_token_count),
                ).astype(np.float32)
            )
            if smplx_vertex_feature_meta.get("enabled"):
                smplx_vertex_feature_meta["views"].append(
                    {
                        "camera_id": normalized_camera_id,
                        "silhouette_pixels": int((mask_aligned > 0.5).sum()),
                        "visible_pixels": int(raster_mask.sum()),
                        **raster_meta,
                    }
                )

        prior_maps.append(np.stack(channels, axis=0).astype(np.float32))
        prior_masks.append(mask_aligned > 0.5)

    return (
        np.stack(prior_maps, axis=0),
        np.stack(prior_masks, axis=0),
        np.stack(prior_summary_tokens, axis=0) if prior_summary_tokens else None,
        smplx_params,
        None,
        {
            "channel_names": channel_names,
            "summary_channel_names": summary_channel_names,
            "channel_groups": channel_group_meta,
            "smplx_vertex_feature_meta": smplx_vertex_feature_meta,
            "external_prior_bundle": external_bundle_meta,
            "keypoint_heatmap_source": "zeros_no_external_2d_keypoints",
        },
    )


def load_and_preprocess_images_numpy(image_paths: list[str], target_size: int = 518) -> np.ndarray:
    if not image_paths:
        raise ValueError("At least one image path is required.")

    processed = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")

        width, height = image.size
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14

        new_width = max(14, int(new_width))
        new_height = max(14, int(new_height))
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        canvas.paste(image, (pad_left, pad_top))
        processed.append(np.asarray(canvas, dtype=np.uint8))

    return np.stack(processed, axis=0)


def resolve_smplx_model_dir(model_dir: str | None) -> Path | None:
    candidates = [
        model_dir,
        os.environ.get("VGGT_SMPLX_MODEL_DIR"),
        r"G:\数据集\datasets\smplx",
        r"G:\datasets\smplx",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return None


def preprocess_scene_masks(scene_manifest: dict, target_size: int) -> np.ndarray:
    masks = []
    for view in scene_manifest["exported_views"]:
        mask = np.asarray(Image.open(view["mask_path"]).convert("L"), dtype=np.float32)
        mask = pad_resize_map(mask, target_size, mode="nearest") > 0.5
        masks.append(mask)
    return np.stack(masks, axis=0)


def align_intrinsics_for_pad_mode(intrinsic: np.ndarray, image_size_wh: list[int] | tuple[int, int], target_size: int) -> np.ndarray:
    width, height = int(image_size_wh[0]), int(image_size_wh[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for intrinsic alignment: {(width, height)}")

    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    new_width = max(14, int(new_width))
    new_height = max(14, int(new_height))

    scale_x = new_width / float(width)
    scale_y = new_height / float(height)
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2

    aligned = intrinsic.astype(np.float32).copy()
    aligned[0, 0] *= scale_x
    aligned[1, 1] *= scale_y
    aligned[0, 2] = intrinsic[0, 2] * scale_x + pad_left
    aligned[1, 2] = intrinsic[1, 2] * scale_y + pad_top
    return aligned


def align_intrinsics_for_scene_view(intrinsic: np.ndarray, view: dict, target_size: int) -> np.ndarray:
    """Align camera intrinsics to the image tensor used by the scene view.

    Normal exported scenes only need the original-image pad/resize alignment.
    Preprocessed crop scenes are different: their manifest image_size is already
    518x518, but the pixels were first aligned from the original image, cropped,
    and pad-resized again. In that case we must replay the crop transform on the
    already-aligned intrinsics instead of treating the cropped image as a fresh
    uncropped camera.
    """
    source_size = view.get("source_image_size") or view.get("image_size")
    aligned = align_intrinsics_for_pad_mode(intrinsic, source_size, target_size=target_size)

    meta = view.get("preprocess_meta") or {}
    transform = meta.get("transform")
    if transform not in {"crop_pad_to_square", "raw_crop_pad_to_square"}:
        return aligned

    bbox = meta.get("crop_bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return aligned

    if transform == "raw_crop_pad_to_square":
        x0, y0, x1, y1 = [float(v) for v in bbox]
        crop_w = max(1.0, x1 - x0)
        crop_h = max(1.0, y1 - y0)
        if crop_w >= crop_h:
            new_w = float(target_size)
            new_h = float(round(crop_h * (new_w / crop_w) / 14.0) * 14)
        else:
            new_h = float(target_size)
            new_w = float(round(crop_w * (new_h / crop_h) / 14.0) * 14)
        new_w = max(14.0, new_w)
        new_h = max(14.0, new_h)
        scale_x = new_w / crop_w
        scale_y = new_h / crop_h
        pad_left = (float(target_size) - new_w) * 0.5
        pad_top = (float(target_size) - new_h) * 0.5

        out = intrinsic.astype(np.float32).copy()
        out[0, 0] *= scale_x
        out[1, 1] *= scale_y
        out[0, 2] = (intrinsic[0, 2] - x0) * scale_x + pad_left
        out[1, 2] = (intrinsic[1, 2] - y0) * scale_y + pad_top
        return out

    x0, y0, x1, y1 = [float(v) for v in bbox]
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    if crop_w >= crop_h:
        new_w = float(target_size)
        new_h = float(round(crop_h * (new_w / crop_w) / 14.0) * 14)
    else:
        new_h = float(target_size)
        new_w = float(round(crop_w * (new_h / crop_h) / 14.0) * 14)
    new_w = max(14.0, new_w)
    new_h = max(14.0, new_h)
    scale_x = new_w / crop_w
    scale_y = new_h / crop_h
    pad_left = (float(target_size) - new_w) * 0.5
    pad_top = (float(target_size) - new_h) * 0.5

    out = aligned.astype(np.float32).copy()
    out[0, 0] *= scale_x
    out[1, 1] *= scale_y
    out[0, 2] = (aligned[0, 2] - x0) * scale_x + pad_left
    out[1, 2] = (aligned[1, 2] - y0) * scale_y + pad_top
    return out


def load_real_camera_params(scene_manifest: dict, dataset_root: Path, subset_name: str) -> dict[str, dict[str, np.ndarray]]:
    context = build_context(dataset_root, subset_name)
    seq_id = scene_manifest["seq_id"]
    temp_handle = tempfile.TemporaryDirectory(prefix="vggt_4k4d_rgbcams_")
    try:
        temp_dir = Path(temp_handle.name)
        rgb_cams_path, source = materialize_rgb_cams_smc(context, seq_id, temp_dir)
        if rgb_cams_path is None:
            raise FileNotFoundError(f"Could not resolve rgb_cams.smc for sequence {seq_id}")

        camera_ids = [normalize_camera_id(view["camera_id"]) for view in scene_manifest["exported_views"]]
        h5py = require_h5py()
        params: dict[str, dict[str, np.ndarray]] = {}
        with h5py.File(rgb_cams_path, "r") as handle:
            camera_group = handle["Camera_Parameter"]
            for camera_id in camera_ids:
                if camera_id not in camera_group:
                    raise KeyError(f"Camera {camera_id} not found in {rgb_cams_path}")
                group = camera_group[camera_id]
                cam_to_world = group["RT"][()].astype(np.float32)
                params[camera_id] = {
                    "intrinsic": group["K"][()].astype(np.float32),
                    "cam_to_world": cam_to_world,
                    "world_to_cam": np.linalg.inv(cam_to_world).astype(np.float32),
                }
        params["_source"] = {"rgb_cams_smc": np.array(str(source))}  # small marker for debugging
        return params
    finally:
        temp_handle.cleanup()


def resolve_scene_camera_params(
    scene_manifest: dict,
    dataset_root: Path,
    subset_name: str,
    camera_params_override: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    if camera_params_override:
        source_tag = str(camera_params_override.get("_source_tag", "camera_override"))
        expected_camera_ids = [
            normalize_camera_id(view["camera_id"])
            for view in scene_manifest["exported_views"]
        ]
        missing = [camera_id for camera_id in expected_camera_ids if camera_id not in camera_params_override]
        if missing:
            raise KeyError(f"Missing scene cameras in override camera payload: {missing}")
        ordered = {
            camera_id: {
                "intrinsic": np.asarray(camera_params_override[camera_id]["intrinsic"], dtype=np.float32),
                "cam_to_world": np.asarray(camera_params_override[camera_id]["cam_to_world"], dtype=np.float32),
                "world_to_cam": np.asarray(camera_params_override[camera_id]["world_to_cam"], dtype=np.float32),
            }
            for camera_id in expected_camera_ids
        }
        return ordered, source_tag
    return load_real_camera_params(scene_manifest, dataset_root, subset_name), "rgb_cams_smc"


def _as_homogeneous_se3(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix.astype(np.float32)
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"Expected extrinsic with shape (3, 4) or (4, 4), got {matrix.shape}")


def build_prediction_camera_override(
    scene_manifest: dict,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    camera_ids = [normalize_camera_id(view["camera_id"]) for view in scene_manifest["exported_views"]]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics with shape [V, 3, 3], got {intrinsics.shape}")
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected extrinsics with shape [V, 3, 4] or [V, 4, 4], got {extrinsics.shape}")
    if intrinsics.shape[0] != len(camera_ids) or extrinsics.shape[0] != len(camera_ids):
        raise ValueError(
            "Prediction camera override view count does not match the scene manifest: "
            f"intrinsics={intrinsics.shape[0]} extrinsics={extrinsics.shape[0]} manifest={len(camera_ids)}"
        )

    override: dict[str, dict[str, np.ndarray]] = {"_source_tag": "prediction_bundle"}
    for view_idx, camera_id in enumerate(camera_ids):
        world_to_cam = _as_homogeneous_se3(extrinsics[view_idx])
        cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)
        override[camera_id] = {
            "intrinsic": intrinsics[view_idx].astype(np.float32),
            "cam_to_world": cam_to_world,
            "world_to_cam": world_to_cam.astype(np.float32),
        }
    return override


def _prefixed_summary_channel_names(channel_names: list[str]) -> list[str]:
    summary_names: list[str] = []
    for name in channel_names:
        if name.startswith("smplx_"):
            summary_names.append(name.replace("smplx_", "smplx_summary_", 1))
        else:
            summary_names.append(f"summary_{name}")
    return summary_names


def _build_channel_group_meta(channel_names: list[str], summary_channel_names: list[str]) -> dict[str, dict[str, list[int]]]:
    def _collect(names: list[str], predicate) -> list[int]:
        return [idx for idx, name in enumerate(names) if predicate(name)]

    dense = {
        "smplx_all": _collect(channel_names, lambda name: name.startswith("smplx_")),
        "smplx_static": _collect(
            channel_names,
            lambda name: (
                name.startswith("smplx_canonical_")
                or name.startswith("smplx_vertex_id_emb_")
                or name.startswith("smplx_body_part_emb_")
                or name.startswith("smplx_skinning_")
            ),
        ),
        "smplx_dynamic": _collect(
            channel_names,
            lambda name: (
                name.startswith("smplx_posed_cam_")
                or name.startswith("smplx_cam_n")
                or name == "smplx_visible_mask"
            ),
        ),
        "smplx_posed_cam": _collect(channel_names, lambda name: name.startswith("smplx_posed_cam_")),
        "smplx_cam_normals": _collect(channel_names, lambda name: name.startswith("smplx_cam_n")),
        "smplx_visibility": _collect(channel_names, lambda name: name == "smplx_visible_mask"),
    }
    summary = {
        "smplx_all": _collect(summary_channel_names, lambda name: name.startswith("smplx_summary_")),
        "smplx_static": _collect(
            summary_channel_names,
            lambda name: (
                name.startswith("smplx_summary_canonical_")
                or name.startswith("smplx_summary_vertex_id_emb_")
                or name.startswith("smplx_summary_body_part_emb_")
                or name.startswith("smplx_summary_skinning_")
            ),
        ),
        "smplx_dynamic": _collect(
            summary_channel_names,
            lambda name: (
                name.startswith("smplx_summary_posed_cam_")
                or name.startswith("smplx_summary_cam_n")
            ),
        ),
        "smplx_posed_cam": _collect(summary_channel_names, lambda name: name.startswith("smplx_summary_posed_cam_")),
        "smplx_cam_normals": _collect(summary_channel_names, lambda name: name.startswith("smplx_summary_cam_n")),
    }
    dense["smplx_spatial"] = sorted(set(dense["smplx_static"] + dense["smplx_dynamic"]))
    return {
        "dense": dense,
        "summary": summary,
    }


def project_world_points(
    world_points: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    cam_points = (rotation @ world_points.T).T + translation[None, :]
    depth = cam_points[:, 2]

    uvw = (intrinsic @ cam_points.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1e-6)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < target_size)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < target_size)
    )
    return uv.astype(np.float32), depth.astype(np.float32), cam_points.astype(np.float32), valid


def render_keypoints3d_geometry_prior(
    scene_manifest: dict,
    dataset_root: Path,
    subset_name: str,
    target_size: int,
    keypoints3d: np.ndarray | None,
    silhouette_masks: np.ndarray,
    min_conf: float,
    knn: int = 8,
    distance_eps: float = 1.0,
    camera_params_override: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    if keypoints3d is None:
        return None, None, None, {"source": "missing_keypoints3d"}

    world_points_all = keypoints3d[:, :3].astype(np.float32)
    confidence_all = keypoints3d[:, 3].astype(np.float32) if keypoints3d.shape[-1] >= 4 else np.ones((len(keypoints3d),), dtype=np.float32)
    valid_points = np.isfinite(world_points_all).all(axis=1) & np.isfinite(confidence_all) & (confidence_all >= float(min_conf))
    if valid_points.sum() == 0:
        return None, None, None, {"source": "empty_keypoints3d", "min_conf": float(min_conf)}

    world_points = world_points_all[valid_points]
    confidence = confidence_all[valid_points]
    camera_params, camera_source = resolve_scene_camera_params(
        scene_manifest,
        dataset_root,
        subset_name,
        camera_params_override=camera_params_override,
    )

    prior_depths = []
    prior_points = []
    prior_masks = []
    view_summaries = []

    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        camera_id = normalize_camera_id(view["camera_id"])
        intrinsic = align_intrinsics_for_scene_view(
            camera_params[camera_id]["intrinsic"],
            view,
            target_size=target_size,
        )
        uv, depth, _cam_points, valid = project_world_points(
            world_points=world_points,
            world_to_cam=camera_params[camera_id]["world_to_cam"],
            intrinsic=intrinsic,
            target_size=target_size,
        )

        mask = silhouette_masks[view_idx].astype(bool)
        geometry_mask = np.zeros((target_size, target_size), dtype=bool)
        depth_map = np.zeros((target_size, target_size), dtype=np.float32)
        point_map = np.zeros((target_size, target_size, 3), dtype=np.float32)

        if valid.sum() == 0 or mask.sum() == 0:
            prior_masks.append(geometry_mask)
            prior_depths.append(depth_map)
            prior_points.append(point_map)
            view_summaries.append(
                {
                    "camera_id": camera_id,
                    "valid_projected_keypoints": int(valid.sum()),
                    "geometry_valid_pixels": 0,
                }
            )
            continue

        uv_valid = uv[valid]
        depth_valid = depth[valid]
        world_valid = world_points[valid]
        conf_valid = confidence[valid]

        query_rc = np.argwhere(mask)
        query_xy = np.stack([query_rc[:, 1], query_rc[:, 0]], axis=1).astype(np.float32)
        tree = cKDTree(uv_valid.astype(np.float32))
        query_k = max(1, min(int(knn), len(uv_valid)))
        dists, indices = tree.query(query_xy, k=query_k, workers=-1)

        if query_k == 1:
            dists = dists[:, None]
            indices = indices[:, None]

        local_conf = conf_valid[indices]
        weights = local_conf / np.maximum(dists, float(distance_eps)) ** 2
        valid_weight = np.isfinite(weights).all(axis=1) & (weights.sum(axis=1) > 0)
        if valid_weight.any():
            weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
            depth_values = np.sum(weights * depth_valid[indices], axis=1).astype(np.float32)
            point_values = np.sum(weights[..., None] * world_valid[indices], axis=1).astype(np.float32)

            valid_rc = query_rc[valid_weight]
            depth_map[valid_rc[:, 0], valid_rc[:, 1]] = depth_values[valid_weight]
            point_map[valid_rc[:, 0], valid_rc[:, 1]] = point_values[valid_weight]
            geometry_mask[valid_rc[:, 0], valid_rc[:, 1]] = True

        prior_masks.append(geometry_mask)
        prior_depths.append(depth_map)
        prior_points.append(point_map)
        view_summaries.append(
            {
                "camera_id": camera_id,
                "valid_projected_keypoints": int(valid.sum()),
                "geometry_valid_pixels": int(geometry_mask.sum()),
            }
        )

    return (
        np.stack(prior_depths, axis=0).astype(np.float32),
        np.stack(prior_points, axis=0).astype(np.float32),
        np.stack(prior_masks, axis=0).astype(bool),
        {
            "source": (
                "keypoints3d_external_camera_idw"
                if camera_source == "external_prior_bundle"
                else "keypoints3d_real_camera_idw"
            ),
            "camera_source": camera_source,
            "min_conf": float(min_conf),
            "knn": int(knn),
            "distance_eps": float(distance_eps),
            "num_keypoints_used": int(len(world_points)),
            "views": view_summaries,
        },
    )


def render_smplx_mesh_geometry_prior(
    scene_manifest: dict,
    dataset_root: Path,
    subset_name: str,
    target_size: int,
    smplx_params: dict[str, np.ndarray],
    silhouette_masks: np.ndarray,
    smplx_model_dir: Path,
    smplx_gender: str = "neutral",
    fill_knn: int = 4,
    camera_params_override: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, object]]:
    required_keys = {"betas", "fullpose"}
    if not required_keys.issubset(smplx_params):
        missing = sorted(required_keys - set(smplx_params))
        return None, None, None, None, {"source": "missing_smplx_params", "missing_keys": missing}

    model_path = resolve_smplx_model_path(smplx_model_dir, gender=smplx_gender)
    mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
        fullpose=smplx_params["fullpose"],
        transl=smplx_params.get("transl"),
        scale=smplx_params.get("scale", 1.0),
    )
    world_vertices = mesh["vertices"].astype(np.float32)
    faces = mesh["faces"].astype(np.int32)

    camera_params, camera_source = resolve_scene_camera_params(
        scene_manifest,
        dataset_root,
        subset_name,
        camera_params_override=camera_params_override,
    )
    prior_depths = []
    prior_points = []
    prior_normals = []
    prior_masks = []
    view_summaries = []
    world_normals = compute_vertex_normals(world_vertices, faces)

    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        camera_id = normalize_camera_id(view["camera_id"])
        intrinsic = align_intrinsics_for_scene_view(
            camera_params[camera_id]["intrinsic"],
            view,
            target_size=target_size,
        )
        silhouette_mask = silhouette_masks[view_idx].astype(bool)
        rotation = np.asarray(camera_params[camera_id]["world_to_cam"][:3, :3], dtype=np.float32)
        cam_normals = world_normals @ rotation.T
        cam_normals /= np.clip(np.linalg.norm(cam_normals, axis=1, keepdims=True), 1e-8, None)
        depth_map, point_map, completed_mask, normal_map, raster_meta = rasterize_world_mesh(
            world_vertices=world_vertices,
            faces=faces,
            world_to_cam=camera_params[camera_id]["world_to_cam"],
            intrinsic=intrinsic,
            image_hw=(target_size, target_size),
            silhouette_mask=silhouette_mask,
            fill_knn=max(1, int(fill_knn)),
            vertex_features=cam_normals.astype(np.float32),
            return_vertex_features=True,
        )
        prior_depths.append(depth_map.astype(np.float32))
        prior_points.append(point_map.astype(np.float32))
        normal_map /= np.clip(np.linalg.norm(normal_map, axis=-1, keepdims=True), 1e-8, None)
        prior_normals.append(normal_map.astype(np.float32))
        prior_masks.append(completed_mask.astype(bool))
        view_summaries.append(
            {
                "camera_id": camera_id,
                "silhouette_pixels": int(silhouette_mask.sum()),
                **raster_meta,
            }
        )

    return (
        np.stack(prior_depths, axis=0).astype(np.float32),
        np.stack(prior_points, axis=0).astype(np.float32),
        np.stack(prior_normals, axis=0).astype(np.float32),
        np.stack(prior_masks, axis=0).astype(bool),
        {
            "source": (
                "smplx_mesh_external_camera_rasterize_knnfill"
                if camera_source == "external_prior_bundle"
                else "smplx_mesh_real_camera_rasterize_knnfill"
            ),
            "camera_source": camera_source,
            "smplx_model_path": str(model_path),
            "smplx_gender": smplx_gender,
            "num_vertices": int(world_vertices.shape[0]),
            "num_faces": int(faces.shape[0]),
            "fill_knn": int(fill_knn),
            "views": view_summaries,
        },
    )


def build_prior_stack(
    scene_manifest: dict,
    dataset_root: Path,
    subset_name: str,
    target_size: int,
    mask_only: bool,
    sigma: float,
    min_conf: float,
    smplx_model_dir: Path | None = None,
    smplx_gender: str = "neutral",
    mesh_fill_knn: int = 4,
    summary_token_count: int = 16,
    vertex_id_dim: int = DEFAULT_VERTEX_ID_EMBED_DIM,
    body_part_dim: int = DEFAULT_BODY_PART_EMBED_DIM,
    body_part_count: int = DEFAULT_BODY_PART_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray], np.ndarray | None, dict[str, object]]:
    context = build_context(dataset_root, subset_name)
    seq_id = scene_manifest["seq_id"]
    frame_idx = int(scene_manifest["frame_id"])
    camera_ids = [view["camera_id"] for view in scene_manifest["exported_views"]]
    view_info_by_camera = {
        normalize_camera_id(view["camera_id"]): view
        for view in scene_manifest["exported_views"]
    }

    prior_maps = []
    prior_masks = []
    prior_summary_tokens = []
    smplx_params = {}
    keypoints3d = None
    channel_names = ["silhouette"] + ([] if mask_only else ["keypoint_heatmap"])
    summary_channel_names: list[str] = []
    channel_group_meta = {"dense": {}, "summary": {}}
    smplx_vertex_feature_meta: dict[str, object] = {
        "enabled": False,
        "source": "not_requested",
        "design_stage": "v2_surface_condition_established",
        "current_v2_capabilities": list(CURRENT_V2_SURFACE_CAPABILITIES),
        "next_stage_enhancements": list(NEXT_STAGE_SURFACE_ENHANCEMENTS),
        "next_stage_goals": list(NEXT_STAGE_ENHANCEMENT_GOALS),
        "roadmap_note": (
            "These enhancements are next-stage upgrades for a stronger SMPL-X condition branch, "
            "not prerequisites for the current V2 to be considered valid."
        ),
    }

    temp_handle = tempfile.TemporaryDirectory(prefix="vggt_4k4d_case_")
    try:
        temp_dir = Path(temp_handle.name)
        annotation_path, _ = resolve_annotations_smc(
            context=context,
            subset_name=subset_name,
            seq=seq_id,
            materialize_archived=True,
            temp_dir=temp_dir,
        )

        h5py = require_h5py()
        with h5py.File(annotation_path, "r") as annotation_handle:
            smplx_params = load_smplx_params(annotation_handle, frame_idx)
            keypoints3d = load_keypoints_3d(annotation_handle, frame_idx)

            world_vertices = None
            faces = None
            camera_params = None
            canonical_positions = None
            static_vertex_features = None
            canonical_scale = 1.0
            cluster_ids = None
            if smplx_model_dir is None:
                smplx_vertex_feature_meta = {
                    "enabled": False,
                    "source": "missing_model_dir",
                    "design_stage": "v2_surface_condition_established",
                    "current_v2_capabilities": list(CURRENT_V2_SURFACE_CAPABILITIES),
                    "next_stage_enhancements": list(NEXT_STAGE_SURFACE_ENHANCEMENTS),
                    "next_stage_goals": list(NEXT_STAGE_ENHANCEMENT_GOALS),
                    "roadmap_note": (
                        "These enhancements are next-stage upgrades for a stronger SMPL-X condition branch, "
                        "not prerequisites for the current V2 to be considered valid."
                    ),
                }
            else:
                required_keys = {"betas", "fullpose"}
                if required_keys.issubset(smplx_params):
                    model_path = resolve_smplx_model_path(smplx_model_dir, gender=smplx_gender)
                    mesh = forward_smplx_mesh(
                        model_path=model_path,
                        betas=smplx_params["betas"],
                        expression=smplx_params.get("expression"),
                        fullpose=smplx_params["fullpose"],
                        transl=smplx_params.get("transl"),
                        scale=smplx_params.get("scale", 1.0),
                    )
                    vertex_feature_payload = build_smplx_vertex_features(
                        model_path=model_path,
                        betas=smplx_params["betas"],
                        expression=smplx_params.get("expression"),
                        vertex_id_dim=int(vertex_id_dim),
                        body_part_dim=int(body_part_dim),
                        body_part_count=int(body_part_count),
                    )
                    world_vertices = mesh["vertices"].astype(np.float32)
                    faces = mesh["faces"].astype(np.int32)
                    canonical_positions = np.asarray(vertex_feature_payload["canonical_positions"], dtype=np.float32)
                    static_vertex_features = np.asarray(vertex_feature_payload["vertex_features"], dtype=np.float32)
                    canonical_scale = float(np.asarray(vertex_feature_payload["canonical_scale"], dtype=np.float32))
                    _, cluster_ids = build_surface_cluster_ids(
                        canonical_positions,
                        num_clusters=int(summary_token_count),
                    )
                    static_channel_names = list(vertex_feature_payload["channel_names"])
                    dynamic_dense_channel_names = [
                        "smplx_posed_cam_x",
                        "smplx_posed_cam_y",
                        "smplx_posed_cam_z",
                        "smplx_cam_nx",
                        "smplx_cam_ny",
                        "smplx_cam_nz",
                        "smplx_visible_mask",
                    ]
                    summary_channel_names = _prefixed_summary_channel_names(static_channel_names) + [
                        "smplx_summary_posed_cam_x",
                        "smplx_summary_posed_cam_y",
                        "smplx_summary_posed_cam_z",
                        "smplx_summary_cam_nx",
                        "smplx_summary_cam_ny",
                        "smplx_summary_cam_nz",
                    ]
                    channel_names.extend(static_channel_names + dynamic_dense_channel_names)
                    channel_group_meta = _build_channel_group_meta(channel_names, summary_channel_names)
                    camera_params = load_real_camera_params(scene_manifest, dataset_root, subset_name)
                    smplx_vertex_feature_meta = {
                        "enabled": True,
                        "source": "smplx_mesh_pose_aligned_surface_prior",
                        "design_stage": "v2_surface_condition_established",
                        "current_v2_capabilities": list(CURRENT_V2_SURFACE_CAPABILITIES),
                        "next_stage_enhancements": list(NEXT_STAGE_SURFACE_ENHANCEMENTS),
                        "next_stage_goals": list(NEXT_STAGE_ENHANCEMENT_GOALS),
                        "roadmap_note": (
                            "These enhancements are next-stage upgrades for a stronger SMPL-X condition branch, "
                            "not prerequisites for the current V2 to be considered valid."
                        ),
                        "smplx_model_path": str(model_path),
                        "smplx_gender": smplx_gender,
                        "num_vertices": int(world_vertices.shape[0]),
                        "num_faces": int(faces.shape[0]),
                        "dense_channels": list(channel_names),
                        "summary_channels": list(summary_channel_names),
                        "channel_groups": channel_group_meta,
                        "summary_token_count": int(summary_token_count),
                        "fill_knn": int(mesh_fill_knn),
                        "vertex_id_dim": int(vertex_id_dim),
                        "body_part_dim": int(body_part_dim),
                        "body_part_count": int(body_part_count),
                        "views": [],
                    }
                else:
                    smplx_vertex_feature_meta = {
                        "enabled": False,
                        "source": "missing_smplx_params",
                        "design_stage": "v2_surface_condition_established",
                        "current_v2_capabilities": list(CURRENT_V2_SURFACE_CAPABILITIES),
                        "next_stage_enhancements": list(NEXT_STAGE_SURFACE_ENHANCEMENTS),
                        "next_stage_goals": list(NEXT_STAGE_ENHANCEMENT_GOALS),
                        "roadmap_note": (
                            "These enhancements are next-stage upgrades for a stronger SMPL-X condition branch, "
                            "not prerequisites for the current V2 to be considered valid."
                        ),
                        "missing_keys": sorted(required_keys - set(smplx_params)),
                    }

            for camera_id in camera_ids:
                mask = load_mask(annotation_handle, camera_id, frame_idx)
                keypoints_2d = load_keypoints_2d(annotation_handle, camera_id, frame_idx)
                image_hw = infer_image_hw(mask, keypoints_2d)

                if mask is None:
                    mask_float = np.zeros(image_hw, dtype=np.float32)
                else:
                    mask_float = (mask > 0).astype(np.float32)
                mask_aligned = pad_resize_map(mask_float, target_size, mode="nearest")
                mask_aligned = (mask_aligned > 0.5).astype(np.float32)

                channels = [mask_aligned]
                if not mask_only:
                    keypoint_heatmap = render_keypoint_heatmap(
                        keypoints_2d=keypoints_2d,
                        image_hw=image_hw,
                        sigma=sigma,
                        min_conf=min_conf,
                    )
                    keypoint_heatmap = pad_resize_map(keypoint_heatmap, target_size, mode="bilinear")
                    channels.append(keypoint_heatmap.astype(np.float32))

                normalized_camera_id = normalize_camera_id(camera_id)
                if (
                    world_vertices is not None
                    and faces is not None
                    and canonical_positions is not None
                    and static_vertex_features is not None
                    and cluster_ids is not None
                    and camera_params is not None
                ):
                    pose_aligned = compute_pose_aligned_vertex_features(
                        world_vertices=world_vertices,
                        faces=faces,
                        canonical_positions=canonical_positions,
                        world_to_cam=camera_params[normalized_camera_id]["world_to_cam"],
                        normalization_scale=canonical_scale,
                    )
                    dense_vertex_features = np.concatenate(
                        [
                            static_vertex_features,
                            pose_aligned["normalized_cam_vertices"],
                            pose_aligned["cam_normals"],
                        ],
                        axis=1,
                    ).astype(np.float32)
                    intrinsic = align_intrinsics_for_scene_view(
                        camera_params[normalized_camera_id]["intrinsic"],
                        view_info_by_camera[normalized_camera_id],
                        target_size=target_size,
                    )
                    _, _, _, feature_map, raster_mask, raster_meta = rasterize_world_mesh(
                        world_vertices=world_vertices,
                        faces=faces,
                        world_to_cam=camera_params[normalized_camera_id]["world_to_cam"],
                        intrinsic=intrinsic,
                        image_hw=(target_size, target_size),
                        silhouette_mask=mask_aligned > 0.5,
                        fill_knn=max(1, int(mesh_fill_knn)),
                        vertex_features=dense_vertex_features,
                        return_vertex_features=True,
                        return_raster_mask=True,
                    )
                    channels.extend(np.moveaxis(feature_map, -1, 0).astype(np.float32))
                    channels.append(raster_mask.astype(np.float32))
                    prior_summary_tokens.append(
                        pool_vertex_features(
                            dense_vertex_features,
                            cluster_ids=cluster_ids,
                            num_clusters=int(summary_token_count),
                        ).astype(np.float32)
                    )
                    if smplx_vertex_feature_meta.get("enabled"):
                        smplx_vertex_feature_meta["views"].append(
                            {
                                "camera_id": normalized_camera_id,
                                "silhouette_pixels": int((mask_aligned > 0.5).sum()),
                                "visible_pixels": int(raster_mask.sum()),
                                **raster_meta,
                            }
                        )
                elif smplx_vertex_feature_meta.get("enabled"):
                    prior_summary_tokens.append(
                        np.zeros((int(summary_token_count), len(summary_channel_names)), dtype=np.float32)
                    )

                prior_maps.append(np.stack(channels, axis=0).astype(np.float32))
                prior_masks.append(mask_aligned > 0.5)
    finally:
        temp_handle.cleanup()

    return (
        np.stack(prior_maps, axis=0),
        np.stack(prior_masks, axis=0),
        np.stack(prior_summary_tokens, axis=0) if prior_summary_tokens else None,
        smplx_params,
        keypoints3d,
        {
            "channel_names": channel_names,
            "summary_channel_names": summary_channel_names,
            "channel_groups": channel_group_meta,
            "smplx_vertex_feature_meta": smplx_vertex_feature_meta,
        },
    )


def load_optional_annotation_payload(
    scene_manifest: dict,
    dataset_root: Path,
    subset_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    context = build_context(dataset_root, subset_name)
    seq_id = scene_manifest["seq_id"]
    frame_idx = int(scene_manifest["frame_id"])

    temp_handle = tempfile.TemporaryDirectory(prefix="vggt_4k4d_case_ann_")
    try:
        temp_dir = Path(temp_handle.name)
        annotation_path, _ = resolve_annotations_smc(
            context=context,
            subset_name=subset_name,
            seq=seq_id,
            materialize_archived=True,
            temp_dir=temp_dir,
        )
        h5py = require_h5py()
        with h5py.File(annotation_path, "r") as annotation_handle:
            smplx_params = load_smplx_params(annotation_handle, frame_idx)
            keypoints3d = load_keypoints_3d(annotation_handle, frame_idx)
    finally:
        temp_handle.cleanup()

    return smplx_params, keypoints3d


def save_optional_annotations(output_dir: Path, frame_idx: int, smplx_params: dict, keypoints3d: np.ndarray | None) -> tuple[str | None, str | None]:
    smplx_output = None
    keypoints3d_output = None

    if smplx_params:
        smplx_output = f"smplx_frame_{frame_idx:04d}.npz"
        np.savez_compressed(output_dir / smplx_output, **smplx_params)

    if keypoints3d is not None:
        keypoints3d_output = f"keypoints3d_frame_{frame_idx:04d}.npy"
        np.save(output_dir / keypoints3d_output, keypoints3d)

    return smplx_output, keypoints3d_output


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    predictions_npz = Path(args.predictions_npz).expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = load_scene_manifest(scene_dir)
    dataset_root = Path(args.dataset_root or scene_manifest["dataset_root"]).expanduser()
    camera_ids = [view["camera_id"] for view in scene_manifest["exported_views"]]
    view_roles = [view["role"] for view in scene_manifest["exported_views"]]
    external_prior_bundle = None
    if args.external_prior_bundle:
        external_prior_bundle = load_external_prior_bundle(
            Path(args.external_prior_bundle),
            scene_manifest=scene_manifest,
        )

    predictions = np.load(predictions_npz, allow_pickle=False)
    depths = predictions["depth"].astype(np.float32)
    extrinsics = predictions["extrinsic"].astype(np.float32)
    intrinsics = predictions["intrinsic"].astype(np.float32)
    if depths.ndim != 4 or depths.shape[1] != depths.shape[2]:
        raise ValueError(f"Expected square depth maps [V, H, W, 1], got {depths.shape}")
    target_size = int(depths.shape[1])
    smplx_model_dir = resolve_smplx_model_dir(args.smplx_model_dir)
    preprocess_variant = resolve_preprocess_variant_name(scene_manifest)
    scene_local_prior_bundle = resolve_scene_local_prior_bundle_path(scene_dir, scene_manifest)
    use_scene_local_prior = external_prior_bundle is None and (
        scene_local_prior_bundle is not None or preprocess_variant is not None
    )

    image_paths = [view["image_path"] for view in scene_manifest["exported_views"]]
    images = load_and_preprocess_images_numpy(image_paths, target_size=target_size)
    point_masks = preprocess_scene_masks(scene_manifest, target_size)

    if depths.shape[0] != len(camera_ids):
        raise ValueError(
            f"Predictions view count {depths.shape[0]} does not match scene manifest {len(camera_ids)}."
        )

    if args.point_source == "world_points":
        world_points = predictions["world_points"].astype(np.float32)
    else:
        world_points = unproject_depth_map_to_point_map_numpy(depths, extrinsics, intrinsics)

    cam_points = np.stack(
        [depth_to_cam_coords_points_numpy(depths[idx, ..., 0], intrinsics[idx]) for idx in range(depths.shape[0])],
        axis=0,
    ).astype(np.float32)

    depth_valid = np.isfinite(depths[..., 0]) & (depths[..., 0] > 1e-8)
    world_valid = np.isfinite(world_points).all(axis=-1)
    point_masks = point_masks & depth_valid & world_valid

    smplx_params: dict[str, np.ndarray] = {}
    keypoints3d = None
    if external_prior_bundle is not None:
        (
            prior_maps,
            prior_mask,
            prior_summary_tokens,
            smplx_params,
            keypoints3d,
            prior_input_meta,
        ) = build_external_prior_stack(
            scene_manifest=scene_manifest,
            target_size=target_size,
            mask_only=args.mask_only,
            smplx_params=external_prior_bundle["smplx_params"],
            camera_params=external_prior_bundle["camera_params"],
            external_bundle_meta=external_prior_bundle["resolved_meta"],
            smplx_model_dir=smplx_model_dir,
            smplx_gender=args.smplx_gender,
            mesh_fill_knn=args.mesh_fill_knn,
            summary_token_count=args.summary_token_count,
            vertex_id_dim=args.vertex_id_dim,
            body_part_dim=args.body_part_dim,
            body_part_count=args.body_part_count,
        )
    elif use_scene_local_prior:
        prior_maps, prior_mask, prior_summary_tokens, prior_input_meta, scene_local_prior_bundle = load_scene_local_prior_bundle(
            scene_dir,
            scene_manifest,
            target_size=target_size,
            expected_view_count=len(camera_ids),
        )
        smplx_params, keypoints3d = load_optional_annotation_payload(
            scene_manifest=scene_manifest,
            dataset_root=dataset_root,
            subset_name=args.subset_name,
        )
    else:
        prior_maps, prior_mask, prior_summary_tokens, smplx_params, keypoints3d, prior_input_meta = build_prior_stack(
            scene_manifest=scene_manifest,
            dataset_root=dataset_root,
            subset_name=args.subset_name,
            target_size=target_size,
            mask_only=args.mask_only,
            sigma=args.sigma,
            min_conf=args.min_conf,
            smplx_model_dir=smplx_model_dir,
            smplx_gender=args.smplx_gender,
            mesh_fill_knn=args.mesh_fill_knn,
            summary_token_count=args.summary_token_count,
            vertex_id_dim=args.vertex_id_dim,
            body_part_dim=args.body_part_dim,
            body_part_count=args.body_part_count,
        )
    prior_depths = None
    prior_points = None
    prior_normals = None
    prior_geometry_mask = None
    prior_geometry_meta: dict[str, object] = {"source": "missing_geometry_prior"}
    camera_params_override = None if external_prior_bundle is None else external_prior_bundle["camera_params"]
    if preprocess_variant is not None:
        camera_params_override = build_prediction_camera_override(
            scene_manifest=scene_manifest,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    if args.geometry_prior_source == "smplx_mesh":
        if smplx_model_dir is None:
            raise FileNotFoundError(
                "SMPL-X geometry prior was requested, but no model directory was found. "
                "Pass --smplx-model-dir or set VGGT_SMPLX_MODEL_DIR."
            )
        if not smplx_params:
            raise ValueError("SMPL-X geometry prior was requested, but the annotation file has no SMPLx group.")

    use_smplx_mesh = (
        args.geometry_prior_source in {"auto", "smplx_mesh"}
        and smplx_model_dir is not None
        and bool(smplx_params)
    )
    if use_smplx_mesh:
        prior_depths, prior_points, prior_normals, prior_geometry_mask, prior_geometry_meta = render_smplx_mesh_geometry_prior(
            scene_manifest=scene_manifest,
            dataset_root=dataset_root,
            subset_name=args.subset_name,
            target_size=target_size,
            smplx_params=smplx_params,
            silhouette_masks=prior_mask,
            smplx_model_dir=smplx_model_dir,
            smplx_gender=args.smplx_gender,
            fill_knn=args.mesh_fill_knn,
            camera_params_override=camera_params_override,
        )
    elif args.geometry_prior_source in {"auto", "keypoints3d"}:
        prior_depths, prior_points, prior_geometry_mask, prior_geometry_meta = render_keypoints3d_geometry_prior(
            scene_manifest=scene_manifest,
            dataset_root=dataset_root,
            subset_name=args.subset_name,
            target_size=target_size,
            keypoints3d=keypoints3d,
            silhouette_masks=prior_mask,
            min_conf=args.min_conf,
            camera_params_override=camera_params_override,
        )

    if prior_geometry_mask is not None:
        prior_mask = prior_geometry_mask

    input_payload = {
        "images": images,
        "point_masks": point_masks.astype(bool),
        "prior_maps": prior_maps.astype(np.float16),
        "prior_mask": prior_mask.astype(bool),
        "camera_ids": np.asarray(camera_ids),
        "view_roles": np.asarray(view_roles),
    }
    if prior_summary_tokens is not None:
        input_payload["prior_summary_tokens"] = prior_summary_tokens.astype(np.float16)
    np.savez_compressed(output_dir / "inputs.npz", **input_payload)
    target_payload = {
        "depths": depths[..., 0].astype(np.float32),
        "extrinsics": extrinsics.astype(np.float32),
        "intrinsics": intrinsics.astype(np.float32),
        "cam_points": cam_points.astype(np.float32),
        "world_points": world_points.astype(np.float32),
        "depth_conf": predictions["depth_conf"].astype(np.float32),
        "world_points_conf": predictions["world_points_conf"].astype(np.float32),
    }
    if prior_depths is not None:
        target_payload["prior_depths"] = prior_depths.astype(np.float32)
    if prior_points is not None:
        target_payload["prior_points"] = prior_points.astype(np.float32)
    if prior_normals is not None:
        target_payload["prior_normals"] = prior_normals.astype(np.float32)
    np.savez_compressed(output_dir / "targets.npz", **target_payload)

    frame_idx = int(scene_manifest["frame_id"])
    smplx_output, keypoints3d_output = save_optional_annotations(output_dir, frame_idx, smplx_params, keypoints3d)

    case_manifest = {
        "case_id": output_dir.name,
        "scene_dir": str(scene_dir),
        "predictions_npz": str(predictions_npz),
        "dataset_root": str(dataset_root),
        "subset_name": args.subset_name,
        "seq_id": scene_manifest["seq_id"],
        "frame_id": scene_manifest["frame_id"],
        "target_camera": scene_manifest["target_camera"],
        "source_cameras": scene_manifest["source_cameras"],
        "camera_ids": camera_ids,
        "view_roles": view_roles,
        "num_views": len(camera_ids),
        "target_view_index": 0,
        "image_hw": list(images.shape[1:3]),
        "point_source": args.point_source,
        "prior_channels": prior_input_meta["channel_names"],
        "prior_summary_channels": prior_input_meta.get("summary_channel_names", []),
        "prior_summary_token_count": 0 if prior_summary_tokens is None else int(prior_summary_tokens.shape[1]),
        "prior_input_meta": prior_input_meta,
        "prior_geometry_source": prior_geometry_meta.get("source"),
        "prior_geometry_meta": prior_geometry_meta,
        "inputs_npz": "inputs.npz",
        "targets_npz": "targets.npz",
        "smplx_output": smplx_output,
        "keypoints3d_output": keypoints3d_output,
        "external_prior_bundle": None if external_prior_bundle is None else external_prior_bundle["resolved_meta"],
    }
    (output_dir / "case_manifest.json").write_text(json.dumps(case_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Prepared 4K4D pseudo-training case at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
