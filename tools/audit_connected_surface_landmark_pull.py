from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from normal_line_multiview_eval import load_scene_view  # noqa: E402
from optimize_raw_smplx_softsurfel_torch import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    detect_face_landmarks_2d,
    detect_hand_landmarks_2d,
    create_face_landmarker,
    create_hand_landmarker,
    homogeneous,
    project_points,
    resolve_scene_path,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)


PART_NAMES = {
    0: "torso_limbs",
    1: "left_hand",
    2: "right_hand",
    3: "head_face",
    4: "head_top_hairline_proxy",
    5: "lower_clothing_proxy",
}

FACE_LANDMARK_GROUPS = {
    "face_oval": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    "central_nose": [1, 2, 4, 5, 6, 19, 45, 94, 97, 98, 129, 168, 195, 197, 275, 326, 327],
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
}

HAND_LANDMARK_GROUPS = {
    "palm": [0, 1, 5, 9, 13, 17],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether 2D MediaPipe face/hand landmark constraints are pulling the "
            "intended connected surface vertices. This is diagnostic only: no teacher, "
            "candidate, training, or cloud unlock is produced."
        )
    )
    parser.add_argument("--mesh-ply", required=True, type=Path)
    parser.add_argument("--template-payload", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--view-stride", type=int, default=10)
    parser.add_argument("--face-landmarker-task", type=Path)
    parser.add_argument("--hand-landmarker-task", type=Path)
    parser.add_argument("--face-landmark-min-confidence", type=float, default=0.02)
    parser.add_argument("--hand-landmark-min-confidence", type=float, default=0.02)
    parser.add_argument("--face-landmark-pad", type=int, default=-1)
    parser.add_argument("--overlay-limit", type=int, default=6)
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


def read_ascii_ply_vertices(path: Path) -> np.ndarray:
    vertex_count = None
    header_lines = 0
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            header_lines += 1
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Could not read vertex count from {path}")
        rows = []
        for _ in range(vertex_count):
            parts = handle.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Unexpected PLY vertex row in {path}")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    _ = header_lines
    return np.asarray(rows, dtype=np.float32)


def project_np(vertices: np.ndarray, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    points = torch.from_numpy(vertices.astype(np.float32))
    uv, z, _ = project_points(
        points,
        torch.from_numpy(world_to_cam.astype(np.float32)),
        torch.from_numpy(intrinsic.astype(np.float32)),
    )
    return uv.detach().cpu().numpy().astype(np.float32), z.detach().cpu().numpy().astype(np.float32)


def nearest_stats(landmarks_xy: np.ndarray, uv: np.ndarray, z: np.ndarray, height: int, width: int) -> dict[str, Any]:
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-5)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    if landmarks_xy.size == 0 or not valid.any():
        return {"usable": False, "valid_vertices": int(valid.sum())}
    uv_valid = uv[valid]
    valid_ids = np.nonzero(valid)[0].astype(np.int64)
    d2 = ((landmarks_xy[:, None, :2] - uv_valid[None, :, :]) ** 2).sum(axis=2)
    nearest_local = d2.argmin(axis=1)
    lm_dist = np.sqrt(d2[np.arange(d2.shape[0]), nearest_local])
    mesh_d2 = d2.min(axis=0)
    mesh_dist = np.sqrt(mesh_d2)
    return {
        "usable": True,
        "landmarks": int(landmarks_xy.shape[0]),
        "valid_vertices": int(valid.sum()),
        "lm_to_mesh_mean_px": float(lm_dist.mean()),
        "lm_to_mesh_p90_px": float(np.percentile(lm_dist, 90)),
        "mesh_to_lm_mean_px": float(mesh_dist.mean()),
        "mesh_to_lm_p90_px": float(np.percentile(mesh_dist, 90)),
        "nearest_vertex_ids": valid_ids[nearest_local],
    }


def add_nearest_part_stats(stats: dict[str, Any], candidate_ids: np.ndarray, part_ids: np.ndarray) -> np.ndarray:
    nearest_local = np.asarray(stats.pop("nearest_vertex_ids"), dtype=np.int64)
    nearest = candidate_ids[nearest_local]
    unique_count = int(np.unique(nearest).size)
    stats["nearest_part_counts"] = {
        PART_NAMES[int(pid)]: int((part_ids[nearest] == int(pid)).sum()) for pid in sorted(set(part_ids[nearest].tolist()))
    }
    stats["unique_nearest_vertices"] = unique_count
    stats["nearest_unique_ratio"] = float(unique_count / max(1, int(nearest.shape[0])))
    stats["nearest_global_vertex_ids"] = nearest.astype(np.int64).tolist()
    return nearest


def landmark_group_stats(
    landmarks: np.ndarray,
    uv: np.ndarray,
    z: np.ndarray,
    candidate_ids: np.ndarray,
    part_ids: np.ndarray,
    groups: dict[str, list[int]],
    height: int,
    width: int,
) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    count = int(landmarks.shape[0])
    for group_name, raw_indices in groups.items():
        indices = np.asarray([idx for idx in raw_indices if 0 <= int(idx) < count], dtype=np.int64)
        if indices.size == 0:
            grouped[group_name] = {"usable": False, "reason": "indices_out_of_range"}
            continue
        stats = nearest_stats(landmarks[indices, :2], uv, z, height, width)
        if stats.get("usable"):
            add_nearest_part_stats(stats, candidate_ids, part_ids)
        grouped[group_name] = stats
    return grouped


def append_group_means(accumulator: dict[str, list[float]], prefix: str, grouped: dict[str, Any]) -> None:
    for group_name, stats in grouped.items():
        if stats.get("usable") and stats.get("lm_to_mesh_mean_px") is not None:
            accumulator.setdefault(f"{prefix}{group_name}", []).append(float(stats["lm_to_mesh_mean_px"]))


def summarize_group_means(accumulator: dict[str, list[float]]) -> dict[str, Any]:
    return {group_name: summarize(values) for group_name, values in sorted(accumulator.items())}


def accumulate_semantic_vertices(
    accumulator: dict[str, Counter[int]],
    prefix: str,
    grouped: dict[str, Any],
) -> None:
    for group_name, stats in grouped.items():
        ids = stats.get("nearest_global_vertex_ids")
        if not ids:
            continue
        counter = accumulator.setdefault(f"{prefix}{group_name}", Counter())
        counter.update(int(idx) for idx in ids)


def write_semantic_correspondence_payload(
    output_path: Path,
    semantic_vertex_counts: dict[str, Counter[int]],
) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for group_name in sorted(semantic_vertex_counts):
        counter = semantic_vertex_counts[group_name]
        items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ids = np.asarray([idx for idx, _count in items], dtype=np.int64)
        counts = np.asarray([count for _idx, count in items], dtype=np.int64)
        key = group_name.replace(".", "__")
        arrays[f"{key}__ids"] = ids
        arrays[f"{key}__counts"] = counts
        total = int(counts.sum()) if counts.size else 0
        summary[group_name] = {
            "unique_vertices": int(ids.size),
            "total_assignments": total,
            "top_count": int(counts[0]) if counts.size else 0,
            "top_count_ratio": float(counts[0] / max(1, total)) if counts.size else 0.0,
        }
    arrays["group_names"] = np.asarray(sorted(semantic_vertex_counts), dtype=str)
    np.savez_compressed(output_path, **arrays)
    return summary


def save_overlay(
    image_path: Path,
    mask_path: Path,
    target_size: int,
    landmarks: np.ndarray,
    uv: np.ndarray,
    nearest_ids: np.ndarray,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB").resize((target_size, target_size), Image.Resampling.BICUBIC)
    mask = Image.open(mask_path).convert("L").resize((target_size, target_size), Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_mask = Image.new("RGBA", image.size, (0, 180, 0, 45))
    overlay.paste(overlay_mask, mask=mask)
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    for lm, vertex_id in zip(landmarks[:, :2], nearest_ids):
        x, y = float(lm[0]), float(lm[1])
        vx, vy = float(uv[int(vertex_id), 0]), float(uv[int(vertex_id), 1])
        draw.line((x, y, vx, vy), fill=(255, 0, 0, 120), width=1)
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 230, 0, 255))
        draw.ellipse((vx - 1.5, vy - 1.5, vx + 1.5, vy + 1.5), fill=(0, 170, 255, 255))
    image.convert("RGB").save(output_path)


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists and is not empty; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene_dir.expanduser().resolve()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or manifest["dataset_root"]).expanduser()
    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)
    views = list(manifest["exported_views"])
    selected = list(range(0, len(views), max(1, int(args.view_stride))))[: max(1, int(args.max_views))]
    height = width = int(args.target_size)

    vertices = read_ascii_ply_vertices(args.mesh_ply.expanduser().resolve())
    with np.load(args.template_payload.expanduser().resolve(), allow_pickle=False) as payload:
        part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
        if part_ids.shape[0] != vertices.shape[0]:
            raise ValueError(f"part_ids shape {part_ids.shape} does not match mesh vertices {vertices.shape}")
        face_mask = np.asarray(payload["face_front_vertex_mask"], dtype=bool)
    face_ids = np.nonzero(face_mask | (part_ids == 3))[0].astype(np.int64)
    left_ids = np.nonzero(part_ids == 1)[0].astype(np.int64)
    right_ids = np.nonzero(part_ids == 2)[0].astype(np.int64)

    face_mp = face_detector = None
    hand_mp = hand_detector = None
    face_meta: dict[str, Any] = {"requested": bool(args.face_landmarker_task)}
    hand_meta: dict[str, Any] = {"requested": bool(args.hand_landmarker_task)}
    if args.face_landmarker_task is not None:
        face_mp, face_detector, face_meta = create_face_landmarker(
            args.face_landmarker_task,
            min_confidence=float(args.face_landmark_min_confidence),
        )
    if args.hand_landmarker_task is not None:
        hand_mp, hand_detector, hand_meta = create_hand_landmarker(
            args.hand_landmarker_task,
            min_confidence=float(args.hand_landmark_min_confidence),
        )
    face_pad = int(args.face_landmark_pad) if int(args.face_landmark_pad) >= 0 else max(4, int(round(height * 0.08)))
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    face_lm_mean: list[float] = []
    hand_lm_mean: list[float] = []
    face_group_means: dict[str, list[float]] = {}
    hand_group_means: dict[str, list[float]] = {}
    semantic_vertex_counts: dict[str, Counter[int]] = {}
    hand_wrong_side = 0
    hand_matches = 0
    duplicate_side_views = 0
    overlays: list[Path] = []
    for view_idx in selected:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        _ = load_scene_view(scene_dir, view_idx, (height, width))
        image_path = resolve_scene_path(scene_dir, view["image_path"])
        mask_path = resolve_scene_path(scene_dir, view["mask_path"])
        intrinsic = align_intrinsics_for_loaded_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=height,
        )
        world_to_cam = homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32))
        uv_all, z_all = project_np(vertices, world_to_cam, intrinsic)
        view_row: dict[str, Any] = {"view_index": int(view_idx), "camera_id": camera_id}

        if face_detector is not None and face_mp is not None:
            face_lm, face_detect_meta, face_img = detect_face_landmarks_2d(
                mp_module=face_mp,
                detector=face_detector,
                image_path=image_path,
                mask_path=mask_path,
                target_size=height,
                pad=face_pad,
            )
            view_row["face_detection"] = face_detect_meta
            if face_lm is not None and face_lm.shape[0] > 0:
                stats = nearest_stats(face_lm[:, :2], uv_all[face_ids], z_all[face_ids], height, width)
                if stats.get("usable"):
                    nearest = add_nearest_part_stats(stats, face_ids, part_ids)
                    face_lm_mean.append(float(stats["lm_to_mesh_mean_px"]))
                    if len(overlays) < int(args.overlay_limit):
                        overlay_path = overlay_dir / f"view_{view_idx:02d}_cam{camera_id}_face_pull.png"
                        save_overlay(image_path, mask_path, height, face_lm, uv_all, nearest, overlay_path)
                        overlays.append(overlay_path)
                face_groups = landmark_group_stats(
                    face_lm,
                    uv_all[face_ids],
                    z_all[face_ids],
                    face_ids,
                    part_ids,
                    FACE_LANDMARK_GROUPS,
                    height,
                    width,
                )
                append_group_means(face_group_means, "", face_groups)
                accumulate_semantic_vertices(semantic_vertex_counts, "face.", face_groups)
                stats["semantic_groups"] = face_groups
                view_row["face_pull"] = stats

        if hand_detector is not None and hand_mp is not None:
            hand_sets, hand_detect_meta, _hand_img = detect_hand_landmarks_2d(
                mp_module=hand_mp,
                detector=hand_detector,
                image_path=image_path,
                mask_path=mask_path,
                target_size=height,
            )
            view_row["hand_detection"] = hand_detect_meta
            hand_rows = []
            matched_sides_this_view: list[str] = []
            for hand_idx, hand_lm in enumerate(hand_sets):
                left_stats = nearest_stats(hand_lm[:, :2], uv_all[left_ids], z_all[left_ids], height, width)
                right_stats = nearest_stats(hand_lm[:, :2], uv_all[right_ids], z_all[right_ids], height, width)
                candidates = []
                if left_stats.get("usable"):
                    candidates.append(("left_hand", left_stats, left_ids))
                if right_stats.get("usable"):
                    candidates.append(("right_hand", right_stats, right_ids))
                if not candidates:
                    hand_rows.append({"hand_index": int(hand_idx), "usable": False})
                    continue
                side, stats, ids = min(candidates, key=lambda item: float(item[1].get("lm_to_mesh_mean_px", 1e9)))
                nearest = add_nearest_part_stats(stats, ids, part_ids)
                stats["matched_side"] = side
                matched_sides_this_view.append(side)
                hand_lm_mean.append(float(stats["lm_to_mesh_mean_px"]))
                hand_matches += 1
                if side not in stats["nearest_part_counts"]:
                    hand_wrong_side += 1
                if len(overlays) < int(args.overlay_limit):
                    overlay_path = overlay_dir / f"view_{view_idx:02d}_cam{camera_id}_hand{hand_idx}_pull.png"
                    save_overlay(image_path, mask_path, height, hand_lm, uv_all, nearest, overlay_path)
                    overlays.append(overlay_path)
                hand_groups = landmark_group_stats(
                    hand_lm,
                    uv_all[ids],
                    z_all[ids],
                    ids,
                    part_ids,
                    HAND_LANDMARK_GROUPS,
                    height,
                    width,
                )
                append_group_means(hand_group_means, f"{side}.", hand_groups)
                accumulate_semantic_vertices(semantic_vertex_counts, f"{side}.", hand_groups)
                stats["semantic_groups"] = hand_groups
                hand_rows.append({"hand_index": int(hand_idx), **stats})
            if len(matched_sides_this_view) >= 2 and len(set(matched_sides_this_view)) < len(matched_sides_this_view):
                duplicate_side_views += 1
            view_row["hand_pull"] = hand_rows
        rows.append(view_row)

    if face_detector is not None and hasattr(face_detector, "close"):
        face_detector.close()
    if hand_detector is not None and hasattr(hand_detector, "close"):
        hand_detector.close()

    semantic_payload_path = out / "semantic_correspondence_payload.npz"
    semantic_correspondence_summary = write_semantic_correspondence_payload(
        semantic_payload_path,
        semantic_vertex_counts,
    )

    summary = {
        "task": "connected_surface_landmark_pull_audit",
        "truthful_status": "diagnostic_only_not_teacher_not_candidate",
        "allows_cloud": False,
        "mesh_ply": args.mesh_ply,
        "template_payload": args.template_payload,
        "scene_dir": scene_dir,
        "camera_source": camera_source,
        "selected_views": selected,
        "vertex_counts": {
            "all": int(vertices.shape[0]),
            "face_candidates": int(face_ids.shape[0]),
            "left_hand": int(left_ids.shape[0]),
            "right_hand": int(right_ids.shape[0]),
        },
        "face_detector": face_meta,
        "hand_detector": hand_meta,
        "face_lm_to_mesh_mean_px": summarize(face_lm_mean),
        "hand_lm_to_mesh_mean_px": summarize(hand_lm_mean),
        "face_semantic_group_lm_to_mesh_mean_px": summarize_group_means(face_group_means),
        "hand_semantic_group_lm_to_mesh_mean_px": summarize_group_means(hand_group_means),
        "semantic_correspondence_payload": semantic_payload_path,
        "semantic_correspondence_summary": semantic_correspondence_summary,
        "hand_matches": int(hand_matches),
        "hand_wrong_side_proxy": int(hand_wrong_side),
        "hand_duplicate_side_views": int(duplicate_side_views),
        "rows": rows,
        "overlays": overlays,
        "current_blocker": (
            "This audit checks whether 2D weak landmark losses have usable connected-surface pull. "
            "It does not create a teacher or candidate and cannot unblock cloud."
        ),
    }
    (out / "landmark_pull_audit_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Connected Surface Landmark Pull Audit",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        f"- selected views: `{selected}`",
        f"- face candidate vertices: `{face_ids.shape[0]}`",
        f"- left/right hand vertices: `{left_ids.shape[0]}` / `{right_ids.shape[0]}`",
        f"- face lm->mesh mean px: `{summary['face_lm_to_mesh_mean_px']}`",
        f"- hand lm->mesh mean px: `{summary['hand_lm_to_mesh_mean_px']}`",
        f"- face semantic group lm->mesh mean px: `{summary['face_semantic_group_lm_to_mesh_mean_px']}`",
        f"- hand semantic group lm->mesh mean px: `{summary['hand_semantic_group_lm_to_mesh_mean_px']}`",
        f"- semantic correspondence payload: `{str(semantic_payload_path).replace(chr(92), '/')}`",
        f"- semantic correspondence summary: `{semantic_correspondence_summary}`",
        f"- hand matches: `{hand_matches}`",
        f"- hand duplicate-side views: `{duplicate_side_views}`",
        f"- overlays: `{[str(p).replace(chr(92), '/') for p in overlays]}`",
        "",
        "This is diagnostic only. It audits whether broad MediaPipe landmark losses",
        "are likely to pull the intended connected vertices and which semantic",
        "subregions are failing; it is not a teacher gate.",
        "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            json_ready(
                {
                    k: summary[k]
                    for k in (
                        "truthful_status",
                        "face_lm_to_mesh_mean_px",
                        "hand_lm_to_mesh_mean_px",
                        "face_semantic_group_lm_to_mesh_mean_px",
                        "hand_semantic_group_lm_to_mesh_mean_px",
                        "semantic_correspondence_payload",
                        "overlays",
                    )
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
