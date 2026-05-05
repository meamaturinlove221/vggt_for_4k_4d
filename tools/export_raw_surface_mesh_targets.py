from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from optimize_raw_smplx_softsurfel_torch import (  # noqa: E402
    compute_vertex_normals,
    export_raster_targets,
    json_ready,
)
from prepare_4k4d_prior_training_case import load_scene_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw-camera raster targets from an existing raw-image optimized "
            "human surface mesh. This is an export/gate diagnostic only; the result "
            "must still pass strict teacher gate plus visual review before any training."
        )
    )
    parser.add_argument("--mesh-ply", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--target-views", default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ascii_triangle_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex_count: int | None = None
    face_count: int | None = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        header_lines += 1
        if first != "ply":
            raise ValueError(f"Expected ASCII PLY header in {path}, got {first!r}")
        fmt = ""
        for line in handle:
            header_lines += 1
            text = line.strip()
            if text.startswith("format "):
                fmt = text
            elif text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("element face "):
                face_count = int(text.split()[-1])
            elif text == "end_header":
                break
        if "ascii" not in fmt:
            raise ValueError(f"Only ASCII PLY is supported, got format line {fmt!r}")
    if vertex_count is None or face_count is None:
        raise ValueError(f"PLY is missing vertex/face counts: {path}")

    vertices = np.zeros((vertex_count, 3), dtype=np.float32)
    faces = np.zeros((face_count, 3), dtype=np.int32)
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(header_lines):
            next(handle)
        for idx in range(vertex_count):
            parts = next(handle).strip().split()
            if len(parts) < 3:
                raise ValueError(f"Bad vertex row {idx} in {path}")
            vertices[idx] = [float(parts[0]), float(parts[1]), float(parts[2])]
        for idx in range(face_count):
            parts = next(handle).strip().split()
            if len(parts) < 4 or int(parts[0]) != 3:
                raise ValueError(f"Only triangle faces are supported; bad row {idx} in {path}")
            faces[idx] = [int(parts[1]), int(parts[2]), int(parts[3])]
    return vertices, faces


def main() -> int:
    args = parse_args()
    mesh_path = args.mesh_ply.expanduser().resolve()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces = read_ascii_triangle_ply(mesh_path)
    manifest = load_scene_manifest(scene_dir)
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else Path(manifest["dataset_root"]).expanduser().resolve()
    vertex_normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    export_summary = export_raster_targets(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        extra_points=None,
        extra_normals=None,
        extra_splat_radius=0,
        scene_dir=scene_dir,
        dataset_root=dataset_root,
        subset_name=str(args.subset_name),
        target_size=int(args.target_size),
        view_spec=str(args.target_views),
        output_dir=output_dir,
    )
    summary: dict[str, Any] = {
        "task": "export_raw_surface_mesh_targets",
        "truthful_status": "raw_surface_mesh_export_complete_not_teacher_or_candidate",
        "mesh_ply": mesh_path,
        "scene_dir": scene_dir,
        "output_dir": output_dir,
        "dataset_root": dataset_root,
        "subset_name": str(args.subset_name),
        "target_size": int(args.target_size),
        "target_views": str(args.target_views),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "uses_vggt_depth_point_normal_as_teacher": False,
        "creates_candidate_predictions": False,
        "creates_teacher_targets": True,
        "strict_teacher_gate_required": True,
        "allows_cloud": False,
        "export": export_summary,
        "note": (
            "This only exports raw-camera raster targets from an existing optimized mesh. "
            "It does not create a strict-passing teacher or unblock cloud."
        ),
    }
    (output_dir / "mesh_export_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
