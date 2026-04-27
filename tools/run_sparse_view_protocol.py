from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-export sparse-view 4K4D scenes for view-count protocols such as 6/8/12/20/60.")
    parser.add_argument("--dataset-root", required=True, help="Extracted data_used_in_4K4D root or its parent folder.")
    parser.add_argument("--seq", required=True, help="Sequence id such as 0012_11")
    parser.add_argument("--frame", default="0", help="Frame id. Default: 0")
    parser.add_argument("--target-camera", default="00", help="Target camera id. Default: 00")
    parser.add_argument("--view-counts", nargs="+", type=int, default=[6, 8, 12, 20, 60], help="Total view counts including target view.")
    parser.add_argument("--output-root", required=True, help="Output root for exported sparse-view scene folders")
    parser.add_argument("--target-size", type=int, default=518, help="Target square size for exported scenes")
    parser.add_argument("--subset-name", default="dna_rendering_part1_main", help="Subset name passed through to downstream tools if needed")
    parser.add_argument("--smplx-model-dir", help="Directory containing SMPL-X model files")
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--mesh-fill-knn", type=int, default=4, help="KNN used to densify SMPL-X feature priors inside the silhouette")
    parser.add_argument("--summary-token-count", type=int, default=16, help="Number of pooled SMPL-X summary tokens per view")
    parser.add_argument("--vertex-id-dim", type=int, default=8, help="Deterministic vertex ID embedding dimension")
    parser.add_argument("--body-part-dim", type=int, default=8, help="Deterministic body-part embedding dimension")
    parser.add_argument("--body-part-count", type=int, default=24, help="Number of coarse body-part groups")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported scenes")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def _scene_name(seq: str, frame: str, total_views: int) -> str:
    return f"{seq}_frame{int(frame):04d}_{int(total_views)}views_sparseproto"


def _build_export_command(args: argparse.Namespace, total_views: int, output_dir: Path) -> list[str]:
    auto_sources = max(0, int(total_views) - 1)
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "export_4k4d_scene.py"),
        "--dataset-root",
        args.dataset_root,
        "--seq",
        args.seq,
        "--frame",
        str(args.frame),
        "--target-camera",
        str(args.target_camera),
        "--auto-sources",
        str(auto_sources),
        "--output-dir",
        str(output_dir),
        "--target-size",
        str(args.target_size),
        "--smplx-gender",
        args.smplx_gender,
        "--mesh-fill-knn",
        str(args.mesh_fill_knn),
        "--summary-token-count",
        str(args.summary_token_count),
        "--vertex-id-dim",
        str(args.vertex_id_dim),
        "--body-part-dim",
        str(args.body_part_dim),
        "--body-part-count",
        str(args.body_part_count),
    ]
    if args.smplx_model_dir:
        command.extend(["--smplx-model-dir", args.smplx_model_dir])
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_entries: list[dict[str, object]] = []
    for total_views in [int(v) for v in args.view_counts]:
        scene_dir = output_root / _scene_name(args.seq, args.frame, total_views)
        command = _build_export_command(args, total_views=total_views, output_dir=scene_dir)
        entry = {
            "view_count": total_views,
            "scene_dir": str(scene_dir),
            "command": command,
            "status": "planned" if args.dry_run else "pending",
        }
        protocol_entries.append(entry)
        if args.dry_run:
            print("DRY RUN:", " ".join(command))
            continue
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        entry["status"] = "exported"

    summary = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "seq": args.seq,
        "frame": int(args.frame),
        "target_camera": str(args.target_camera),
        "view_counts": [int(v) for v in args.view_counts],
        "output_root": str(output_root),
        "entries": protocol_entries,
    }
    summary_path = output_root / f"{args.seq}_frame{int(args.frame):04d}_sparse_protocol_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
