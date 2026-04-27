from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a COLMAP fused point cloud by multi-view scene masks. This is a teacher-quality "
            "gate, not a sparse-view reconstruction result."
        )
    )
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--export-summary", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-mask-votes", type=int, default=8)
    parser.add_argument("--max-views", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=180000)
    parser.add_argument("--robust-percentile", type=float, default=0.2)
    parser.add_argument("--drop-white", action="store_true")
    parser.add_argument("--max-output-points", type=int, default=900000)
    parser.add_argument("--seed", type=int, default=20260427)
    return parser.parse_args()


def _read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = path.read_bytes()
    header_end_token = data.find(b"end_header")
    if header_end_token < 0:
        raise ValueError(f"PLY missing end_header: {path}")
    header_end = data.find(b"\n", header_end_token) + 1
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    if not any(line.strip() == "format binary_little_endian 1.0" for line in header):
        raise ValueError("Only binary_little_endian PLY is supported.")
    vertex_count = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
    has_normals = any(line.strip() == "property float nx" for line in header)
    if has_normals:
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ]
        )
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    records = np.frombuffer(data, dtype=dtype, count=vertex_count, offset=header_end)
    points = np.stack([records["x"], records["y"], records["z"]], axis=-1).astype(np.float32)
    colors = np.stack([records["r"], records["g"], records["b"]], axis=-1).astype(np.uint8)
    return points, colors, {"input_vertex_count": int(vertex_count), "has_normals": bool(has_normals)}


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    records = np.empty(points.shape[0], dtype=dtype)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]
    with path.open("wb") as handle:
        handle.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {points.shape[0]}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
        )
        handle.write(records.tobytes())


def _load_masks(scene_dir: Path, target_size: tuple[int, int], max_views: int) -> list[np.ndarray]:
    manifest = json.loads((scene_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    masks: list[np.ndarray] = []
    for view in manifest["exported_views"][:max_views]:
        mask = Image.open(view["mask_path"]).convert("L")
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        masks.append(np.asarray(mask) > 127)
    return masks


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points, colors, ply_meta = _read_binary_ply(args.input_ply)
    summary = json.loads(args.export_summary.read_text(encoding="utf-8"))
    views = summary["exported_views"][: int(args.max_views)]
    width, height = [int(value) for value in views[0]["image_size"]]
    masks = _load_masks(args.scene_dir, (width, height), len(views))

    finite = np.isfinite(points).all(axis=1)
    if args.drop_white:
        finite &= ~((colors > 245).all(axis=1))
    if 0.0 < float(args.robust_percentile) < 50.0 and finite.any():
        lo, hi = np.percentile(points[finite], [float(args.robust_percentile), 100.0 - float(args.robust_percentile)], axis=0)
        finite &= np.all((points >= lo) & (points <= hi), axis=1)

    candidate_indices = np.flatnonzero(finite)
    votes = np.zeros(points.shape[0], dtype=np.uint16)
    chunk_size = max(1, int(args.chunk_size))
    for start in range(0, len(candidate_indices), chunk_size):
        indices = candidate_indices[start : start + chunk_size]
        chunk = points[indices].astype(np.float32)
        homo = np.concatenate([chunk, np.ones((chunk.shape[0], 1), dtype=np.float32)], axis=1)
        chunk_votes = np.zeros(chunk.shape[0], dtype=np.uint16)
        for view, mask in zip(views, masks):
            intrinsic = np.asarray(view["intrinsic"], dtype=np.float32)
            world_to_cam = np.asarray(view["world_to_cam"], dtype=np.float32)
            cam = (homo @ world_to_cam.T)[:, :3]
            z = cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.rint(intrinsic[0, 0] * cam[:, 0] / z + intrinsic[0, 2]).astype(np.int32)
                v = np.rint(intrinsic[1, 1] * cam[:, 1] / z + intrinsic[1, 2]).astype(np.int32)
            inside = np.isfinite(z) & (z > 0.2) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if np.any(inside):
                valid_positions = np.flatnonzero(inside)
                chunk_votes[valid_positions] += mask[v[inside], u[inside]]
        votes[indices] = chunk_votes

    keep = finite & (votes >= int(args.min_mask_votes))
    keep_indices = np.flatnonzero(keep)
    if len(keep_indices) > int(args.max_output_points) > 0:
        rng = np.random.default_rng(int(args.seed))
        keep_indices = rng.choice(keep_indices, size=int(args.max_output_points), replace=False)
        keep_indices.sort()

    filtered_path = output_dir / f"colmap_fused_maskvote_v{int(args.min_mask_votes):02d}.ply"
    _write_binary_ply(filtered_path, points[keep_indices], colors[keep_indices])
    vote_valid = votes[finite]
    vote_hist = {str(int(vote)): int((vote_valid == vote).sum()) for vote in np.unique(vote_valid)}
    payload = {
        **ply_meta,
        "input_ply": str(args.input_ply.resolve()),
        "export_summary": str(args.export_summary.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_ply": str(filtered_path),
        "max_views": int(args.max_views),
        "min_mask_votes": int(args.min_mask_votes),
        "robust_percentile": float(args.robust_percentile),
        "drop_white": bool(args.drop_white),
        "finite_candidate_points": int(finite.sum()),
        "kept_points_before_sampling": int(keep.sum()),
        "written_points": int(len(keep_indices)),
        "vote_percentiles_on_candidates": [float(v) for v in np.percentile(vote_valid, [0, 25, 50, 75, 90, 95, 99, 100])] if vote_valid.size else [],
        "vote_histogram": vote_hist,
        "truthful_status": "multi_view_mask_filtered_teacher_candidate_not_sparse_view_pass",
    }
    (output_dir / "maskvote_filter_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
