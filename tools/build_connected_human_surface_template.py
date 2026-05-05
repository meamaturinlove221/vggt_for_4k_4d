from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_smplx_model_dir,
)
from tools.smplx_numpy import (  # noqa: E402
    build_smplx_vertex_features,
    compute_vertex_normals,
    forward_smplx_mesh,
    resolve_smplx_model_path,
)


PART_NAMES = {
    0: "torso_limbs",
    1: "left_hand",
    2: "right_hand",
    3: "head_face",
    4: "head_top_hairline",
    5: "lower_clothing_proxy",
}

PART_COLORS = {
    0: (170, 170, 170),
    1: (30, 180, 255),
    2: (255, 140, 30),
    3: (245, 210, 170),
    4: (80, 30, 140),
    5: (30, 180, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the connected part-aware raw-image human surface template used by "
            "the v2 upper-bound route. This is a geometry carrier only: it does not "
            "train VGGT, does not use VGGT depth/point/normal as a teacher, and does "
            "not create a strict-passing candidate."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--hair-ring-count", type=int, default=96)
    parser.add_argument("--hair-outer-offset", type=float, default=0.035)
    parser.add_argument("--hair-up-offset", type=float, default=0.030)
    parser.add_argument("--hair-cap-height", type=float, default=0.070)
    parser.add_argument(
        "--subdivide-parts",
        default="",
        help=(
            "Optional comma-separated part ids or names to locally split triangle faces "
            "after the connected hair cap is attached. This increases connected surface "
            "capacity for face/hair/hands/clothing without introducing floating patches."
        ),
    )
    parser.add_argument("--subdivision-levels", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    return value


def safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.clip(np.linalg.norm(vectors, axis=-1, keepdims=True), eps, None)


def classify_vertex_parts(canonical_positions: np.ndarray) -> dict[str, np.ndarray]:
    canonical = np.asarray(canonical_positions, dtype=np.float32)
    x = canonical[:, 0]
    y = canonical[:, 1]
    z = canonical[:, 2]
    center_x = float(np.median(x))
    abs_x = np.abs(x - center_x)
    y20, y82, y88, y94, y96 = np.percentile(y, [20, 82, 88, 94, 96])
    abs_x88 = np.percentile(abs_x, 88)
    abs_x94 = np.percentile(abs_x, 94)
    z_head_median = float(np.median(z[y > y82])) if np.any(y > y82) else float(np.median(z))

    parts = np.zeros((canonical.shape[0],), dtype=np.int64)
    lower_clothing = y < y20
    head = y > y82
    hairline = y > y96
    hand_wide = (abs_x > abs_x88) & (y > y20) & (y < y94)
    hand_far = (abs_x > abs_x94) & (y > y20) & (y < y96)
    left_hand = (hand_wide | hand_far) & (x < center_x)
    right_hand = (hand_wide | hand_far) & (x >= center_x)
    face_front = head & (z >= z_head_median)

    parts[lower_clothing] = 5
    parts[head] = 3
    parts[hairline] = 4
    parts[left_hand] = 1
    parts[right_hand] = 2

    masks = {
        "part_ids": parts,
        "torso_limbs": parts == 0,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "head_face": head,
        "head_top_hairline": hairline,
        "face_front_proxy": face_front,
        "lower_clothing_proxy": lower_clothing,
        "thresholds": {
            "y20": float(y20),
            "y82": float(y82),
            "y88": float(y88),
            "y94": float(y94),
            "y96": float(y96),
            "center_x": center_x,
            "abs_x88": float(abs_x88),
            "abs_x94": float(abs_x94),
            "z_head_median": z_head_median,
        },
    }
    return masks


def face_mask_from_vertices(faces: np.ndarray, vertex_mask: np.ndarray, mode: str = "any") -> np.ndarray:
    face_hits = np.asarray(vertex_mask, dtype=bool)[np.asarray(faces, dtype=np.int64)]
    if mode == "all":
        return face_hits.all(axis=1)
    return face_hits.any(axis=1)


def submesh(vertices: np.ndarray, faces: np.ndarray, face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_faces = np.asarray(faces, dtype=np.int64)[np.asarray(face_mask, dtype=bool)]
    if selected_faces.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int64)
    used = np.unique(selected_faces.reshape(-1))
    remap = {int(old): idx for idx, old in enumerate(used.tolist())}
    remapped = np.asarray([[remap[int(v)] for v in face] for face in selected_faces], dtype=np.int32)
    return vertices[used].astype(np.float32), remapped, used.astype(np.int64)


def save_colored_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if colors is None:
        colors = np.full((vertices.shape[0], 3), 200, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            handle.write(
                f"{float(vertex[0])} {float(vertex[1])} {float(vertex[2])} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_points(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{float(point[0])} {float(point[1])} {float(point[2])} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def mesh_colors_from_parts(parts: np.ndarray) -> np.ndarray:
    out = np.zeros((parts.shape[0], 3), dtype=np.uint8)
    for part_id, color in PART_COLORS.items():
        out[np.asarray(parts) == int(part_id)] = np.asarray(color, dtype=np.uint8)
    return out


def parse_subdivide_parts(spec: str) -> set[int]:
    aliases = {name: int(part_id) for part_id, name in PART_NAMES.items()}
    aliases.update(
        {
            "body": 0,
            "torso": 0,
            "limbs": 0,
            "left": 1,
            "left_hand": 1,
            "right": 2,
            "right_hand": 2,
            "head": 3,
            "face": 3,
            "head_face": 3,
            "hair": 4,
            "hairline": 4,
            "head_top": 4,
            "clothing": 5,
            "lower": 5,
            "skirt": 5,
        }
    )
    out: set[int] = set()
    for raw in str(spec).split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
            part_id = int(item)
        else:
            if item not in aliases:
                raise ValueError(f"Unknown subdivision part '{raw}'. Known names: {sorted(aliases)}")
            part_id = int(aliases[item])
        if part_id not in PART_NAMES:
            raise ValueError(f"Subdivision part id {part_id} is not in PART_NAMES")
        out.add(part_id)
    return out


def most_common_part(part_values: np.ndarray, *, preferred: set[int] | None = None) -> int:
    values = [int(value) for value in np.asarray(part_values).reshape(-1).tolist()]
    if preferred:
        preferred_values = [value for value in values if value in preferred]
        if preferred_values:
            values = preferred_values
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=max(PART_NAMES) + 1)
    return int(np.argmax(counts))


def subdivide_selected_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    part_ids: np.ndarray,
    face_front_mask: np.ndarray,
    selected_parts: set[int],
    *,
    levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32)
    part_ids_np = np.asarray(part_ids, dtype=np.int64)
    face_front_np = np.asarray(face_front_mask, dtype=bool)
    stats: list[dict[str, int]] = []
    if int(levels) <= 0 or not selected_parts:
        return vertices_np, faces_np, part_ids_np, face_front_np, {
            "enabled": False,
            "levels": int(levels),
            "selected_parts": sorted(int(part) for part in selected_parts),
            "stats": stats,
        }

    for level in range(int(levels)):
        selected_face_mask: list[bool] = []
        split_edges: set[tuple[int, int]] = set()
        for face in faces_np:
            a, b, c = [int(value) for value in face.tolist()]
            face_parts = part_ids_np[[a, b, c]]
            should_split = any(int(value) in selected_parts for value in face_parts.tolist())
            selected_face_mask.append(bool(should_split))
            if should_split:
                for u, v in ((a, b), (b, c), (c, a)):
                    split_edges.add((int(u), int(v)) if int(u) < int(v) else (int(v), int(u)))
        edge_to_mid: dict[tuple[int, int], int] = {}
        vertex_list = [vertices_np[index].copy() for index in range(vertices_np.shape[0])]
        part_list = [int(value) for value in part_ids_np.tolist()]
        face_front_list = [bool(value) for value in face_front_np.tolist()]
        new_faces: list[list[int]] = []
        selected_faces = 0

        def midpoint(a: int, b: int, face_has_front: bool, face_part_values: np.ndarray, prefer_selected: bool) -> int:
            key = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            if key in edge_to_mid:
                return edge_to_mid[key]
            idx = len(vertex_list)
            vertex_list.append(((vertices_np[int(a)] + vertices_np[int(b)]) * 0.5).astype(np.float32))
            preferred = selected_parts if prefer_selected else None
            edge_parts = np.asarray([part_ids_np[int(a)], part_ids_np[int(b)], most_common_part(face_part_values, preferred=preferred)])
            part_list.append(most_common_part(edge_parts, preferred=preferred))
            face_front_list.append(bool(face_has_front or face_front_np[int(a)] or face_front_np[int(b)]))
            edge_to_mid[key] = idx
            return idx

        def edge_key(a: int, b: int) -> tuple[int, int]:
            return (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))

        for face_idx, face in enumerate(faces_np):
            a, b, c = [int(value) for value in face.tolist()]
            face_parts = part_ids_np[[a, b, c]]
            should_split = bool(selected_face_mask[face_idx])
            face_has_front = bool(face_front_np[[a, b, c]].any())
            split_ab = edge_key(a, b) in split_edges
            split_bc = edge_key(b, c) in split_edges
            split_ca = edge_key(c, a) in split_edges
            if not (split_ab or split_bc or split_ca):
                new_faces.append([a, b, c])
                continue
            selected_faces += int(should_split)
            ab = midpoint(a, b, face_has_front, face_parts, should_split) if split_ab else None
            bc = midpoint(b, c, face_has_front, face_parts, should_split) if split_bc else None
            ca = midpoint(c, a, face_has_front, face_parts, should_split) if split_ca else None
            if split_ab and split_bc and split_ca:
                assert ab is not None and bc is not None and ca is not None
                new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
            elif split_ab and not split_bc and not split_ca:
                assert ab is not None
                new_faces.extend([[a, ab, c], [ab, b, c]])
            elif split_bc and not split_ab and not split_ca:
                assert bc is not None
                new_faces.extend([[a, b, bc], [a, bc, c]])
            elif split_ca and not split_ab and not split_bc:
                assert ca is not None
                new_faces.extend([[a, b, ca], [b, c, ca]])
            elif split_ab and split_bc and not split_ca:
                assert ab is not None and bc is not None
                new_faces.extend([[a, ab, bc], [a, bc, c], [ab, b, bc]])
            elif split_bc and split_ca and not split_ab:
                assert bc is not None and ca is not None
                new_faces.extend([[a, b, bc], [a, bc, ca], [ca, bc, c]])
            elif split_ca and split_ab and not split_bc:
                assert ca is not None and ab is not None
                new_faces.extend([[a, ab, ca], [ab, b, c], [ab, c, ca]])

        vertices_np = np.asarray(vertex_list, dtype=np.float32)
        faces_np = np.asarray(new_faces, dtype=np.int32)
        part_ids_np = np.asarray(part_list, dtype=np.int64)
        face_front_np = np.asarray(face_front_list, dtype=bool)
        stats.append(
            {
                "level": int(level + 1),
                "selected_faces": int(selected_faces),
                "vertices": int(vertices_np.shape[0]),
                "faces": int(faces_np.shape[0]),
                "new_midpoint_vertices": int(len(edge_to_mid)),
            }
        )

    return vertices_np, faces_np, part_ids_np, face_front_np, {
        "enabled": True,
        "levels": int(levels),
        "selected_parts": sorted(int(part) for part in selected_parts),
        "stats": stats,
    }


def select_hair_ring(
    vertices: np.ndarray,
    normals: np.ndarray,
    canonical_positions: np.ndarray,
    vertex_mask: np.ndarray,
    ring_count: int,
) -> dict[str, np.ndarray]:
    canonical = np.asarray(canonical_positions, dtype=np.float32)
    vertices = np.asarray(vertices, dtype=np.float32)
    candidates = np.nonzero(np.asarray(vertex_mask, dtype=bool))[0]
    if candidates.size < 8:
        raise RuntimeError(f"Need at least 8 hairline candidates, got {candidates.size}")
    center = canonical[candidates].mean(axis=0)
    angles = np.arctan2(canonical[candidates, 2] - center[2], canonical[candidates, 0] - center[0])
    radii = np.linalg.norm(canonical[candidates][:, [0, 2]] - center[[0, 2]][None, :], axis=1)
    count = int(max(8, min(int(ring_count), candidates.size)))
    target_angles = np.linspace(-np.pi, np.pi, count, endpoint=False, dtype=np.float32)
    target_radius = float(np.percentile(radii, 82))
    radius_scale = float(np.percentile(radii, 95) - np.percentile(radii, 15) + 1e-6)
    chosen: list[int] = []
    used: set[int] = set()
    for target in target_angles:
        angular = np.abs(np.angle(np.exp(1j * (angles - target))))
        radial = np.abs(radii - target_radius) / radius_scale
        score = angular + 0.18 * radial
        order = np.argsort(score)
        best = int(order[0])
        for item in order:
            candidate_id = int(candidates[int(item)])
            if candidate_id not in used:
                best = int(item)
                used.add(candidate_id)
                break
        chosen.append(int(candidates[best]))
    ring_ids = np.asarray(chosen, dtype=np.int64)
    ring_points = vertices[ring_ids]
    ring_center = ring_points.mean(axis=0)
    return {
        "ring_vertex_ids": ring_ids.astype(np.int64),
        "ring_points": vertices[ring_ids].astype(np.float32),
        "ring_normals": normals[ring_ids].astype(np.float32),
        "ring_center": ring_center.astype(np.float32),
    }


def build_connected_hair_cap(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    canonical_positions: np.ndarray,
    hairline_mask: np.ndarray,
    *,
    ring_count: int,
    outer_offset: float,
    up_offset: float,
    cap_height: float,
) -> dict[str, np.ndarray]:
    ring = select_hair_ring(vertices, normals, canonical_positions, hairline_mask, ring_count)
    ring_ids = ring["ring_vertex_ids"]
    ring_points = ring["ring_points"]
    n = ring_ids.shape[0]
    center = ring_points.mean(axis=0)
    up = np.zeros((1, 3), dtype=np.float32)
    up[0, 1] = -1.0 if float(center[1]) < float(vertices[:, 1].mean()) else 1.0

    smoothed = ring_points.copy()
    for _ in range(3):
        smoothed = 0.25 * np.roll(smoothed, 1, axis=0) + 0.50 * smoothed + 0.25 * np.roll(smoothed, -1, axis=0)
    radial = smoothed - center[None, :]
    radial[:, 1] = 0.0
    radial = safe_normalize(radial)
    base_radius = float(np.median(np.linalg.norm((smoothed - center[None, :])[:, [0, 2]], axis=1)))
    smooth_inner = center[None, :] + base_radius * radial
    smooth_inner[:, 1] = smoothed[:, 1]
    inner = (0.65 * smoothed + 0.35 * smooth_inner).astype(np.float32)
    outer = inner + float(outer_offset) * radial + float(up_offset) * up
    top_radius = max(1e-4, base_radius * 0.55)
    top = center[None, :] + top_radius * radial + float(cap_height) * up
    apex = center + np.asarray([0.0, float(cap_height) * 1.35, 0.0], dtype=np.float32)
    apex[1] = center[1] + float(up[0, 1]) * float(cap_height) * 1.35

    new_vertices = np.concatenate([inner, outer, top, apex[None, :]], axis=0).astype(np.float32)
    inner_ids = np.arange(vertices.shape[0], vertices.shape[0] + n, dtype=np.int64)
    outer_ids = np.arange(vertices.shape[0] + n, vertices.shape[0] + 2 * n, dtype=np.int64)
    top_ids = np.arange(vertices.shape[0] + 2 * n, vertices.shape[0] + 3 * n, dtype=np.int64)
    apex_id = int(vertices.shape[0] + 3 * n)
    hybrid_faces: list[list[int]] = []
    local_faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        # Weld scalp anchors to a smoothed inner ring before expanding outward.
        hybrid_faces.append([int(ring_ids[i]), int(ring_ids[j]), int(inner_ids[j])])
        hybrid_faces.append([int(ring_ids[i]), int(inner_ids[j]), int(inner_ids[i])])
        # Inner to outer band.
        hybrid_faces.append([int(inner_ids[i]), int(inner_ids[j]), int(outer_ids[j])])
        hybrid_faces.append([int(inner_ids[i]), int(outer_ids[j]), int(outer_ids[i])])
        # Outer to top cap band.
        hybrid_faces.append([int(outer_ids[i]), int(outer_ids[j]), int(top_ids[j])])
        hybrid_faces.append([int(outer_ids[i]), int(top_ids[j]), int(top_ids[i])])
        # Top cap to apex.
        hybrid_faces.append([int(top_ids[i]), int(top_ids[j]), apex_id])

        local_faces.append([i, j, n + j])
        local_faces.append([i, n + j, n + i])
        local_faces.append([n + i, n + j, 2 * n + j])
        local_faces.append([n + i, 2 * n + j, 2 * n + i])
        local_faces.append([2 * n + i, 2 * n + j, 3 * n + j])
        local_faces.append([2 * n + i, 3 * n + j, 3 * n + i])
        local_faces.append([3 * n + i, 3 * n + j, 4 * n])
    local_vertices = np.concatenate([ring_points, inner, outer, top, apex[None, :]], axis=0).astype(np.float32)
    return {
        "ring_vertex_ids": ring_ids.astype(np.int64),
        "new_vertices": new_vertices.astype(np.float32),
        "hybrid_faces": np.asarray(hybrid_faces, dtype=np.int32),
        "local_vertices": local_vertices,
        "local_faces": np.asarray(local_faces, dtype=np.int32),
        "ring_points": ring_points,
        "inner_points": inner.astype(np.float32),
        "outer_points": outer.astype(np.float32),
        "top_points": top.astype(np.float32),
        "apex": apex.astype(np.float32),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Connected Human Surface Template v2",
        "",
        "Status: `template_only_not_teacher_not_candidate`",
        "",
        "This starts the raw-image surface upper-bound v2 route. It only builds",
        "the connected geometry carrier. It does not optimize the surface, does not",
        "train VGGT, and does not permit cloud.",
        "",
        "## Counts",
        "",
        "```text",
        f"base vertices = {summary['base_mesh']['vertices']}",
        f"base faces = {summary['base_mesh']['faces']}",
        f"hybrid vertices = {summary['hybrid_mesh']['vertices']}",
        f"hybrid faces = {summary['hybrid_mesh']['faces']}",
        f"hair seam vertices = {summary['hair_cap']['ring_vertices']}",
        f"hair cap new vertices = {summary['hair_cap']['new_vertices']}",
        f"hair cap faces = {summary['hair_cap']['faces']}",
        f"local subdivision enabled = {summary['subdivision']['enabled']}",
        f"local subdivision levels = {summary['subdivision']['levels']}",
        f"local subdivision parts = {summary['subdivision']['selected_part_names']}",
        "```",
        "",
        "## Part Regions",
        "",
    ]
    for name, item in summary["regions"].items():
        lines.append(f"- `{name}`: vertices `{item['vertices']}`, faces `{item['faces']}`")
    if summary["subdivision"]["enabled"]:
        lines.extend(["", "## Local Subdivision"])
        for item in summary["subdivision"]["stats"]:
            lines.append(
                "- level `{level}`: split faces `{selected_faces}`, midpoint vertices `{new_midpoint_vertices}`, "
                "vertices `{vertices}`, faces `{faces}`".format(**item)
            )
    lines.extend(
        [
            "",
            "## Key Constraint",
            "",
            "The hair/head cap is connected to existing SMPL-X scalp vertices through",
            "explicit seam faces. This avoids the previous floating-hairline-point",
            "failure mode, but it is still only a scaffold. It needs a depth-ordered",
            "renderer and raw-image losses before it can be evaluated as a surface",
            "teacher.",
            "",
            "## Next Required Action",
            "",
            "Use this template in a v2 optimizer with connected residual variables,",
            "soft/depth-ordered rasterization, multi-view photometric consistency,",
            "boundary/edge loss, face weak reprojection, hand connectivity, and the",
            "existing strict teacher gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene_dir.expanduser().resolve()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or manifest["dataset_root"]).expanduser()
    smplx_model_dir = resolve_smplx_model_dir(None if args.smplx_model_dir is None else str(args.smplx_model_dir))
    if smplx_model_dir is None:
        raise FileNotFoundError("Could not resolve SMPL-X model dir; pass --smplx-model-dir.")
    model_path = resolve_smplx_model_path(smplx_model_dir, args.smplx_gender)
    smplx_params, _ = load_optional_annotation_payload(manifest, dataset_root, args.subset_name)
    if not smplx_params:
        raise ValueError("Scene annotations do not provide SMPL-X parameters.")

    mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
        fullpose=smplx_params["fullpose"],
        transl=smplx_params.get("transl"),
        scale=smplx_params.get("scale", 1.0),
    )
    static_features = build_smplx_vertex_features(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
    )
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    canonical = np.asarray(static_features["canonical_positions"], dtype=np.float32)
    masks = classify_vertex_parts(canonical)
    part_ids = np.asarray(masks["part_ids"], dtype=np.int64)
    colors = mesh_colors_from_parts(part_ids)

    hair_cap = build_connected_hair_cap(
        vertices=vertices,
        faces=faces,
        normals=normals,
        canonical_positions=canonical,
        hairline_mask=np.asarray(masks["head_top_hairline"], dtype=bool),
        ring_count=int(args.hair_ring_count),
        outer_offset=float(args.hair_outer_offset),
        up_offset=float(args.hair_up_offset),
        cap_height=float(args.hair_cap_height),
    )
    hair_new_vertices = np.asarray(hair_cap["new_vertices"], dtype=np.float32)
    hybrid_vertices = np.concatenate([vertices, hair_new_vertices], axis=0).astype(np.float32)
    hybrid_faces = np.concatenate([faces, hair_cap["hybrid_faces"]], axis=0).astype(np.int32)
    hybrid_part_ids = np.concatenate(
        [part_ids, np.full((hair_new_vertices.shape[0],), 4, dtype=np.int64)],
        axis=0,
    ).astype(np.int64)
    hybrid_face_front_mask = np.concatenate(
        [
            np.asarray(masks["face_front_proxy"], dtype=bool),
            np.zeros((hair_new_vertices.shape[0],), dtype=bool),
        ],
        axis=0,
    )
    selected_subdivide_parts = parse_subdivide_parts(args.subdivide_parts)
    subdivision_summary: dict[str, Any]
    hybrid_vertices, hybrid_faces, hybrid_part_ids, hybrid_face_front_mask, subdivision_summary = subdivide_selected_faces(
        hybrid_vertices,
        hybrid_faces,
        hybrid_part_ids,
        hybrid_face_front_mask,
        selected_subdivide_parts,
        levels=int(args.subdivision_levels),
    )
    subdivision_summary["selected_part_names"] = [PART_NAMES[int(part)] for part in subdivision_summary["selected_parts"]]
    hybrid_colors = mesh_colors_from_parts(hybrid_part_ids)

    save_colored_mesh(output_dir / "smplx_part_template_full.ply", vertices, faces, colors)
    save_colored_mesh(output_dir / "connected_human_surface_template_hybrid.ply", hybrid_vertices, hybrid_faces, hybrid_colors)
    save_colored_mesh(
        output_dir / "connected_head_hair_cap_template.ply",
        hair_cap["local_vertices"],
        hair_cap["local_faces"],
        np.tile(np.asarray(PART_COLORS[4], dtype=np.uint8)[None, :], (hair_cap["local_vertices"].shape[0], 1)),
    )
    save_points(
        output_dir / "scalp_hairline_seam_ring_points.ply",
        hair_cap["ring_points"],
        np.tile(np.asarray([255, 40, 220], dtype=np.uint8)[None, :], (hair_cap["ring_points"].shape[0], 1)),
    )

    region_masks = {
        "face_front_proxy": hybrid_face_front_mask.astype(bool),
        "head_face": (hybrid_part_ids == 3) | (hybrid_part_ids == 4),
        "head_top_hairline": hybrid_part_ids == 4,
        "left_hand": hybrid_part_ids == 1,
        "right_hand": hybrid_part_ids == 2,
        "lower_clothing_proxy": hybrid_part_ids == 5,
    }
    regions: dict[str, dict[str, Any]] = {}
    for name, vertex_mask in region_masks.items():
        region_face_mask = face_mask_from_vertices(hybrid_faces, vertex_mask, mode="any")
        sub_vertices, sub_faces, used_vertices = submesh(hybrid_vertices, hybrid_faces, region_face_mask)
        save_colored_mesh(
            output_dir / f"{name}_region.ply",
            sub_vertices,
            sub_faces,
            np.tile(np.asarray(PART_COLORS.get(3 if "face" in name or "head" in name else 0), dtype=np.uint8)[None, :], (sub_vertices.shape[0], 1)),
        )
        regions[name] = {
            "vertices": int(vertex_mask.sum()),
            "faces": int(region_face_mask.sum()),
            "used_vertices": int(used_vertices.shape[0]),
            "ply": output_dir / f"{name}_region.ply",
        }

    payload_path = output_dir / "connected_human_surface_template_payload.npz"
    np.savez_compressed(
        payload_path,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals.astype(np.float32),
        canonical_positions=canonical.astype(np.float32),
        part_ids=hybrid_part_ids.astype(np.int64),
        hybrid_vertices=hybrid_vertices.astype(np.float32),
        hybrid_faces=hybrid_faces.astype(np.int32),
        hair_ring_vertex_ids=hair_cap["ring_vertex_ids"].astype(np.int64),
        hair_new_vertices=hair_new_vertices.astype(np.float32),
        hair_hybrid_faces=hair_cap["hybrid_faces"].astype(np.int32),
        face_front_vertex_mask=region_masks["face_front_proxy"].astype(bool),
        head_vertex_mask=region_masks["head_face"].astype(bool),
        hairline_vertex_mask=region_masks["head_top_hairline"].astype(bool),
        left_hand_vertex_mask=region_masks["left_hand"].astype(bool),
        right_hand_vertex_mask=region_masks["right_hand"].astype(bool),
        lower_clothing_vertex_mask=region_masks["lower_clothing_proxy"].astype(bool),
        subdivision_selected_parts=np.asarray(subdivision_summary["selected_parts"], dtype=np.int64),
        subdivision_levels=np.asarray([int(subdivision_summary["levels"])], dtype=np.int64),
    )

    summary = {
        "task": "connected_human_surface_template_v2",
        "truthful_status": "template_only_not_teacher_not_candidate",
        "allows_cloud": False,
        "uses_vggt_depth_point_normal": False,
        "scene_dir": scene_dir,
        "output_dir": output_dir,
        "base_mesh": {"vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])},
        "hybrid_mesh": {"vertices": int(hybrid_vertices.shape[0]), "faces": int(hybrid_faces.shape[0])},
        "hair_cap": {
            "ring_vertices": int(hair_cap["ring_vertex_ids"].shape[0]),
            "new_vertices": int(hair_new_vertices.shape[0]),
            "faces": int(hair_cap["hybrid_faces"].shape[0]),
            "outer_offset": float(args.hair_outer_offset),
            "up_offset": float(args.hair_up_offset),
            "cap_height": float(args.hair_cap_height),
        },
        "subdivision": subdivision_summary,
        "regions": regions,
        "thresholds": masks["thresholds"],
        "outputs": {
            "payload_npz": payload_path,
            "full_part_mesh": output_dir / "smplx_part_template_full.ply",
            "hybrid_mesh": output_dir / "connected_human_surface_template_hybrid.ply",
            "hair_cap_mesh": output_dir / "connected_head_hair_cap_template.ply",
            "seam_ring_points": output_dir / "scalp_hairline_seam_ring_points.ply",
            "report_md": output_dir / "report.md",
            "summary_json": output_dir / "connected_human_surface_template_summary.json",
        },
        "non_wall_decision": (
            "This replaces floating hairline points and pure SMPL-X head offsets with a connected "
            "hair/head cap scaffold. It is still only a carrier; success requires a renderer, "
            "raw-image losses, Open3D visual pass, full-body/hands pass, and strict teacher gate."
        ),
    }
    (output_dir / "connected_human_surface_template_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summary)
    print(json.dumps(json_ready({"truthful_status": summary["truthful_status"], "outputs": summary["outputs"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
