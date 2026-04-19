from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


MODEL_FILENAMES = {
    "neutral": "SMPLX_NEUTRAL.npz",
    "female": "SMPLX_FEMALE.npz",
    "male": "SMPLX_MALE.npz",
}

SMPLX_VERTEX_FEATURE_CHANNELS = (
    "smplx_template_x",
    "smplx_template_y",
    "smplx_template_z",
    "smplx_template_nx",
    "smplx_template_ny",
    "smplx_template_nz",
)


def resolve_smplx_model_path(model_dir: str | Path, gender: str = "neutral") -> Path:
    normalized_gender = str(gender).strip().lower()
    if normalized_gender not in MODEL_FILENAMES:
        raise ValueError(f"Unsupported SMPL-X gender: {gender}")

    model_dir = Path(model_dir).expanduser().resolve()
    filename = MODEL_FILENAMES[normalized_gender]
    candidates = [
        model_dir / filename,
        model_dir / "smplx_npz" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename} under {model_dir}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


@lru_cache(maxsize=4)
def _load_smplx_model_cached(model_path_str: str) -> dict[str, np.ndarray]:
    model_path = Path(model_path_str)
    with np.load(model_path, allow_pickle=True) as data:
        parents = np.asarray(data["kintree_table"][0], dtype=np.int64)
        invalid_parent = np.iinfo(np.uint32).max
        parents[parents == invalid_parent] = -1

        shapedirs = np.asarray(data["shapedirs"], dtype=np.float32)
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        posedirs = posedirs.reshape(-1, posedirs.shape[-1])

        return {
            "v_template": np.asarray(data["v_template"], dtype=np.float32),
            "faces": np.asarray(data["f"], dtype=np.int32),
            "weights": np.asarray(data["weights"], dtype=np.float32),
            "J_regressor": np.asarray(data["J_regressor"], dtype=np.float32),
            "shapedirs": shapedirs,
            "posedirs": posedirs,
            "parents": parents.astype(np.int32),
        }


def load_smplx_model(model_path: str | Path) -> dict[str, np.ndarray]:
    return _load_smplx_model_cached(str(Path(model_path).expanduser().resolve()))


def get_smplx_vertex_feature_channel_names() -> tuple[str, ...]:
    return SMPLX_VERTEX_FEATURE_CHANNELS


def _blend_shape_slice(shapedirs: np.ndarray, coeffs: np.ndarray, start: int) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size == 0 or start >= shapedirs.shape[-1]:
        return np.zeros(shapedirs.shape[:2], dtype=np.float32)

    stop = min(shapedirs.shape[-1], start + coeffs.size)
    dirs = shapedirs[:, :, start:stop]
    local_coeffs = coeffs[: dirs.shape[-1]]
    if dirs.shape[-1] == 0:
        return np.zeros(shapedirs.shape[:2], dtype=np.float32)
    return np.tensordot(dirs, local_coeffs, axes=([2], [0])).astype(np.float32)


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ).astype(np.float32)
    for corner_idx in range(3):
        np.add.at(normals, faces[:, corner_idx], face_normals)
    return _safe_normalize(normals).astype(np.float32)


def build_smplx_vertex_features(
    model_path: str | Path,
    betas: np.ndarray,
    expression: np.ndarray | None = None,
    expression_start: int = 300,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    model = load_smplx_model(model_path)
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    expression = (
        np.zeros((0,), dtype=np.float32)
        if expression is None
        else np.asarray(expression, dtype=np.float32).reshape(-1)
    )

    rest_vertices = model["v_template"].copy()
    rest_vertices += _blend_shape_slice(model["shapedirs"], betas, start=0)
    rest_vertices += _blend_shape_slice(model["shapedirs"], expression, start=int(expression_start))
    rest_vertices = rest_vertices.astype(np.float32)

    centered_vertices = rest_vertices - rest_vertices.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered_vertices, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    normalized_vertices = centered_vertices / scale
    vertex_normals = compute_vertex_normals(rest_vertices, model["faces"])
    vertex_features = np.concatenate([normalized_vertices, vertex_normals], axis=1).astype(np.float32)

    return {
        "vertex_features": vertex_features,
        "channel_names": SMPLX_VERTEX_FEATURE_CHANNELS,
        "rest_vertices": rest_vertices,
        "canonical_positions": normalized_vertices.astype(np.float32),
        "canonical_normals": vertex_normals.astype(np.float32),
        "canonical_scale": np.asarray(scale, dtype=np.float32),
    }


def compute_pose_aligned_vertex_features(
    world_vertices: np.ndarray,
    faces: np.ndarray,
    canonical_positions: np.ndarray,
    world_to_cam: np.ndarray,
    normalization_scale: float | np.ndarray,
) -> dict[str, np.ndarray]:
    world_vertices = np.asarray(world_vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    canonical_positions = np.asarray(canonical_positions, dtype=np.float32)
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)

    world_normals = compute_vertex_normals(world_vertices, faces)
    cam_vertices = world_vertices @ rotation.T + translation[None, :]
    cam_normals = _safe_normalize(world_normals @ rotation.T).astype(np.float32)

    scale = float(np.asarray(normalization_scale, dtype=np.float32))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    normalized_cam_vertices = (cam_vertices / scale).astype(np.float32)

    return {
        "canonical_positions": canonical_positions.astype(np.float32),
        "normalized_cam_vertices": normalized_cam_vertices,
        "cam_normals": cam_normals.astype(np.float32),
    }


def build_surface_cluster_ids(canonical_positions: np.ndarray, num_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    canonical_positions = np.asarray(canonical_positions, dtype=np.float32)
    num_vertices = canonical_positions.shape[0]
    if num_vertices == 0:
        raise ValueError("canonical_positions must contain at least one vertex.")

    num_clusters = max(1, min(int(num_clusters), num_vertices))
    anchor_indices = np.zeros((num_clusters,), dtype=np.int64)
    distances = np.full((num_vertices,), np.inf, dtype=np.float32)

    centroid = canonical_positions.mean(axis=0, keepdims=True)
    farthest_idx = int(np.argmax(np.linalg.norm(canonical_positions - centroid, axis=1)))
    for cluster_idx in range(num_clusters):
        anchor_indices[cluster_idx] = farthest_idx
        anchor = canonical_positions[farthest_idx]
        sq_dists = np.sum((canonical_positions - anchor[None, :]) ** 2, axis=1).astype(np.float32)
        distances = np.minimum(distances, sq_dists)
        farthest_idx = int(np.argmax(distances))

    anchor_positions = canonical_positions[anchor_indices]
    sq_distances = np.sum(
        (canonical_positions[:, None, :] - anchor_positions[None, :, :]) ** 2,
        axis=2,
    ).astype(np.float32)
    cluster_ids = np.argmin(sq_distances, axis=1).astype(np.int64)
    return anchor_indices, cluster_ids


def pool_vertex_features(vertex_features: np.ndarray, cluster_ids: np.ndarray, num_clusters: int) -> np.ndarray:
    vertex_features = np.asarray(vertex_features, dtype=np.float32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    num_clusters = max(1, int(num_clusters))
    pooled = np.zeros((num_clusters, vertex_features.shape[1]), dtype=np.float32)
    for cluster_idx in range(num_clusters):
        mask = cluster_ids == cluster_idx
        if not np.any(mask):
            continue
        pooled[cluster_idx] = vertex_features[mask].mean(axis=0).astype(np.float32)
    return pooled.astype(np.float32)


def _compute_skinning_transforms(
    rot_mats: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_joints = int(joints.shape[0])
    relative_joints = joints.copy()
    for joint_idx in range(1, num_joints):
        parent_idx = int(parents[joint_idx])
        relative_joints[joint_idx] -= joints[parent_idx]

    global_transforms = np.zeros((num_joints, 4, 4), dtype=np.float32)
    for joint_idx in range(num_joints):
        local_transform = _make_transform(rot_mats[joint_idx], relative_joints[joint_idx])
        parent_idx = int(parents[joint_idx])
        if parent_idx < 0:
            global_transforms[joint_idx] = local_transform
        else:
            global_transforms[joint_idx] = global_transforms[parent_idx] @ local_transform

    rest_transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_joints, axis=0)
    rest_transforms[:, :3, 3] = -joints
    skinning_transforms = global_transforms @ rest_transforms
    posed_joints = global_transforms[:, :3, 3].astype(np.float32)
    return skinning_transforms.astype(np.float32), posed_joints


def forward_smplx_mesh(
    model_path: str | Path,
    betas: np.ndarray,
    fullpose: np.ndarray,
    transl: np.ndarray | None = None,
    expression: np.ndarray | None = None,
    scale: float | np.ndarray = 1.0,
    expression_start: int = 300,
) -> dict[str, np.ndarray]:
    model = load_smplx_model(model_path)

    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    expression = np.zeros((0,), dtype=np.float32) if expression is None else np.asarray(expression, dtype=np.float32).reshape(-1)
    transl = np.zeros((3,), dtype=np.float32) if transl is None else np.asarray(transl, dtype=np.float32).reshape(3)
    fullpose = np.asarray(fullpose, dtype=np.float32).reshape(-1, 3)

    num_joints = int(model["parents"].shape[0])
    if fullpose.shape[0] != num_joints:
        raise ValueError(f"Expected fullpose with {num_joints} joints, got {fullpose.shape}")

    vertices = model["v_template"].copy()
    vertices += _blend_shape_slice(model["shapedirs"], betas, start=0)
    vertices += _blend_shape_slice(model["shapedirs"], expression, start=int(expression_start))

    joints = model["J_regressor"] @ vertices
    rot_mats = Rotation.from_rotvec(fullpose.astype(np.float64)).as_matrix().astype(np.float32)

    pose_feature = (rot_mats[1:] - np.eye(3, dtype=np.float32)[None, :, :]).reshape(-1)
    posedirs = model["posedirs"]
    if pose_feature.shape[0] < posedirs.shape[1]:
        pose_feature = np.pad(pose_feature, (0, posedirs.shape[1] - pose_feature.shape[0]), constant_values=0.0)
    elif pose_feature.shape[0] > posedirs.shape[1]:
        pose_feature = pose_feature[: posedirs.shape[1]]
    vertices = vertices + (posedirs @ pose_feature).reshape(-1, 3).astype(np.float32)

    transforms, posed_joints = _compute_skinning_transforms(rot_mats, joints, model["parents"])
    vertex_transforms = np.einsum("vj,jab->vab", model["weights"], transforms, optimize=True)
    vertices_h = np.concatenate(
        [vertices.astype(np.float32), np.ones((vertices.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    vertices = np.einsum("vab,vb->va", vertex_transforms, vertices_h, optimize=True)[:, :3]

    scale_value = float(np.asarray(scale, dtype=np.float32))
    vertices = vertices * scale_value + transl[None, :]
    posed_joints = posed_joints * scale_value + transl[None, :]

    return {
        "vertices": vertices.astype(np.float32),
        "joints": posed_joints.astype(np.float32),
        "faces": model["faces"].astype(np.int32),
    }


def _edge_function(a: np.ndarray, b: np.ndarray, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
    return (points_x - a[0]) * (b[1] - a[1]) - (points_y - a[1]) * (b[0] - a[0])


def _compute_knn_fill_plan(
    raster_mask: np.ndarray,
    target_mask: np.ndarray,
    knn: int = 4,
    distance_eps: float = 1.0,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    dict[str, int],
]:
    target_mask = np.asarray(target_mask, dtype=bool)
    raster_mask = np.asarray(raster_mask, dtype=bool)

    if not target_mask.any():
        return None, None, None, None, raster_mask.copy(), {
            "filled_pixels": 0,
            "completed_pixels": int(raster_mask.sum()),
        }
    if not raster_mask.any():
        return None, None, None, None, raster_mask.copy(), {
            "filled_pixels": 0,
            "completed_pixels": 0,
        }

    query_mask = target_mask & ~raster_mask
    if not query_mask.any():
        return None, None, None, None, target_mask.copy(), {
            "filled_pixels": 0,
            "completed_pixels": int(target_mask.sum()),
        }

    source_rc = np.argwhere(raster_mask)
    query_rc = np.argwhere(query_mask)
    source_xy = np.stack([source_rc[:, 1], source_rc[:, 0]], axis=1).astype(np.float32)
    query_xy = np.stack([query_rc[:, 1], query_rc[:, 0]], axis=1).astype(np.float32)

    query_k = max(1, min(int(knn), len(source_xy)))
    tree = cKDTree(source_xy)
    dists, indices = tree.query(query_xy, k=query_k, workers=-1)
    if query_k == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    weights = 1.0 / np.maximum(dists, float(distance_eps)) ** 2
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)

    completed_mask = raster_mask.copy()
    completed_mask[target_mask] = True
    return source_rc, query_rc, indices, weights, completed_mask, {
        "filled_pixels": int(query_mask.sum()),
        "completed_pixels": int(completed_mask.sum()),
    }


def _fill_values_with_knn_plan(
    value_map: np.ndarray,
    source_rc: np.ndarray | None,
    query_rc: np.ndarray | None,
    indices: np.ndarray | None,
    weights: np.ndarray | None,
) -> np.ndarray:
    if source_rc is None or query_rc is None or indices is None or weights is None or len(query_rc) == 0:
        return np.asarray(value_map).copy()

    value_map = np.asarray(value_map)
    source_values = value_map[source_rc[:, 0], source_rc[:, 1]]
    if value_map.ndim == 2:
        filled_values = np.sum(weights * source_values[indices], axis=1)
    else:
        filled_values = np.sum(weights[..., None] * source_values[indices], axis=1)

    filled_map = value_map.copy()
    filled_map[query_rc[:, 0], query_rc[:, 1]] = filled_values.astype(value_map.dtype, copy=False)
    return filled_map


def _complete_dense_prior_from_mask(
    depth_map: np.ndarray,
    point_map: np.ndarray,
    raster_mask: np.ndarray,
    target_mask: np.ndarray,
    knn: int = 4,
    distance_eps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    source_rc, query_rc, indices, weights, completed_mask, fill_meta = _compute_knn_fill_plan(
        raster_mask=raster_mask,
        target_mask=target_mask,
        knn=knn,
        distance_eps=distance_eps,
    )
    depth_map = _fill_values_with_knn_plan(depth_map, source_rc, query_rc, indices, weights).astype(np.float32)
    point_map = _fill_values_with_knn_plan(point_map, source_rc, query_rc, indices, weights).astype(np.float32)
    return depth_map, point_map, completed_mask.astype(bool), fill_meta


def rasterize_world_mesh(
    world_vertices: np.ndarray,
    faces: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    silhouette_mask: np.ndarray | None = None,
    fill_knn: int = 4,
    vertex_features: np.ndarray | None = None,
    return_vertex_features: bool = False,
    return_raster_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid raster image size: {(height, width)}")

    world_vertices = np.asarray(world_vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)

    cam_vertices = world_vertices @ rotation.T + translation[None, :]
    depth = cam_vertices[:, 2]
    uvw = (intrinsic @ cam_vertices.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    valid_vertices = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)

    if silhouette_mask is not None:
        silhouette_mask = np.asarray(silhouette_mask, dtype=bool)
        if silhouette_mask.shape != (height, width):
            raise ValueError(f"Silhouette mask shape {silhouette_mask.shape} does not match {(height, width)}")

    if vertex_features is not None:
        vertex_features = np.asarray(vertex_features, dtype=np.float32)
        if vertex_features.ndim != 2 or vertex_features.shape[0] != world_vertices.shape[0]:
            raise ValueError(
                f"vertex_features must have shape [V, C] aligned with world_vertices; got {vertex_features.shape}"
            )
    if return_vertex_features and vertex_features is None:
        raise ValueError("return_vertex_features=True requires vertex_features to be provided.")

    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    depth_map = np.zeros((height, width), dtype=np.float32)
    point_map = np.zeros((height, width, 3), dtype=np.float32)
    raster_mask = np.zeros((height, width), dtype=bool)
    feature_map = None
    if vertex_features is not None:
        feature_map = np.zeros((height, width, vertex_features.shape[1]), dtype=np.float32)

    faces_rasterized = 0
    for face in faces:
        face = np.asarray(face, dtype=np.int32)
        if not valid_vertices[face].all():
            continue

        tri_uv = uv[face]
        tri_depth = depth[face]
        tri_world = world_vertices[face]
        tri_features = vertex_features[face] if feature_map is not None else None

        min_x = max(int(np.floor(float(tri_uv[:, 0].min()) - 0.5)), 0)
        max_x = min(int(np.ceil(float(tri_uv[:, 0].max()) - 0.5)), width - 1)
        min_y = max(int(np.floor(float(tri_uv[:, 1].min()) - 0.5)), 0)
        max_y = min(int(np.ceil(float(tri_uv[:, 1].max()) - 0.5)), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area = _edge_function(tri_uv[0], tri_uv[1], tri_uv[2][0], tri_uv[2][1])
        if not np.isfinite(area) or abs(float(area)) < 1e-8:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        bary0 = _edge_function(tri_uv[1], tri_uv[2], grid_x, grid_y) / area
        bary1 = _edge_function(tri_uv[2], tri_uv[0], grid_x, grid_y) / area
        bary2 = 1.0 - bary0 - bary1

        inside = (bary0 >= -1e-5) & (bary1 >= -1e-5) & (bary2 >= -1e-5)
        if silhouette_mask is not None:
            inside &= silhouette_mask[min_y : max_y + 1, min_x : max_x + 1]
        if not inside.any():
            continue

        interpolated_depth = bary0 * tri_depth[0] + bary1 * tri_depth[1] + bary2 * tri_depth[2]
        local_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (interpolated_depth < local_z)
        if not update.any():
            continue

        interpolated_world = (
            bary0[..., None] * tri_world[0][None, None, :]
            + bary1[..., None] * tri_world[1][None, None, :]
            + bary2[..., None] * tri_world[2][None, None, :]
        )
        interpolated_features = None
        if tri_features is not None:
            interpolated_features = (
                bary0[..., None] * tri_features[0][None, None, :]
                + bary1[..., None] * tri_features[1][None, None, :]
                + bary2[..., None] * tri_features[2][None, None, :]
            )

        local_depth = depth_map[min_y : max_y + 1, min_x : max_x + 1]
        local_points = point_map[min_y : max_y + 1, min_x : max_x + 1]
        local_mask = raster_mask[min_y : max_y + 1, min_x : max_x + 1]
        local_features = feature_map[min_y : max_y + 1, min_x : max_x + 1] if feature_map is not None else None

        local_z[update] = interpolated_depth[update]
        local_depth[update] = interpolated_depth[update].astype(np.float32)
        local_points[update] = interpolated_world[update].astype(np.float32)
        local_mask[update] = True
        if local_features is not None and interpolated_features is not None:
            local_features[update] = interpolated_features[update].astype(np.float32)
        faces_rasterized += 1

    completed_mask = raster_mask
    fill_meta = {"filled_pixels": 0, "completed_pixels": int(raster_mask.sum())}
    if silhouette_mask is not None and fill_knn > 0:
        source_rc, query_rc, indices, weights, completed_mask, fill_meta = _compute_knn_fill_plan(
            raster_mask=raster_mask,
            target_mask=silhouette_mask,
            knn=fill_knn,
        )
        depth_map = _fill_values_with_knn_plan(depth_map, source_rc, query_rc, indices, weights).astype(np.float32)
        point_map = _fill_values_with_knn_plan(point_map, source_rc, query_rc, indices, weights).astype(np.float32)
        if feature_map is not None:
            feature_map = _fill_values_with_knn_plan(feature_map, source_rc, query_rc, indices, weights).astype(np.float32)

    meta = {
        "valid_vertices": int(valid_vertices.sum()),
        "faces_total": int(len(faces)),
        "faces_rasterized": int(faces_rasterized),
        "rasterized_pixels": int(raster_mask.sum()),
        "filled_pixels": int(fill_meta["filled_pixels"]),
        "completed_pixels": int(fill_meta["completed_pixels"]),
    }
    if return_vertex_features and return_raster_mask:
        return (
            depth_map.astype(np.float32),
            point_map.astype(np.float32),
            completed_mask.astype(bool),
            feature_map.astype(np.float32),
            raster_mask.astype(bool),
            meta,
        )
    if return_vertex_features:
        return (
            depth_map.astype(np.float32),
            point_map.astype(np.float32),
            completed_mask.astype(bool),
            feature_map.astype(np.float32),
            meta,
        )
    if return_raster_mask:
        return (
            depth_map.astype(np.float32),
            point_map.astype(np.float32),
            completed_mask.astype(bool),
            raster_mask.astype(bool),
            meta,
        )
    return depth_map.astype(np.float32), point_map.astype(np.float32), completed_mask.astype(bool), meta
