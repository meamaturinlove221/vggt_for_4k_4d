from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_4k4d_prior_training_case import (
    DEFAULT_BODY_PART_COUNT,
    DEFAULT_BODY_PART_EMBED_DIM,
    DEFAULT_VERTEX_ID_EMBED_DIM,
    build_external_prior_stack,
    load_external_prior_bundle,
    load_scene_manifest,
    resolve_smplx_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a scene-local prior_maps.npz from an external SMPL-X prior bundle, "
            "so the scene can be consumed directly by modal_4k4d_vggt_infer.py."
        )
    )
    parser.add_argument("--scene-dir", required=True, help="Input scene directory containing scene_manifest.json.")
    parser.add_argument(
        "--external-prior-bundle",
        required=True,
        help="External prior bundle manifest or directory produced by run_realdata_smplx_driver.py.",
    )
    parser.add_argument(
        "--output-scene-dir",
        help="Optional output scene directory. If omitted, prior_maps.npz is written in place.",
    )
    parser.add_argument("--target-size", type=int, default=518, help="Spatial size of the generated prior maps.")
    parser.add_argument("--mask-only", action="store_true", help="Emit silhouette-only prior channels.")
    parser.add_argument("--smplx-model-dir", help="Directory containing SMPL-X model files.")
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--mesh-fill-knn", type=int, default=4)
    parser.add_argument("--summary-token-count", type=int, default=16)
    parser.add_argument("--vertex-id-dim", type=int, default=DEFAULT_VERTEX_ID_EMBED_DIM)
    parser.add_argument("--body-part-dim", type=int, default=DEFAULT_BODY_PART_EMBED_DIM)
    parser.add_argument("--body-part-count", type=int, default=DEFAULT_BODY_PART_COUNT)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files/directories.")
    return parser.parse_args()


def _copy_or_prepare_scene(scene_dir: Path, output_scene_dir: Path, *, overwrite: bool) -> None:
    if scene_dir == output_scene_dir:
        output_scene_dir.mkdir(parents=True, exist_ok=True)
        return
    if output_scene_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_scene_dir} already exists. Re-run with --overwrite.")
        shutil.rmtree(output_scene_dir)
    shutil.copytree(scene_dir, output_scene_dir)


def _rewrite_manifest_asset_paths(scene_manifest: dict, scene_dir: Path) -> dict:
    manifest = json.loads(json.dumps(scene_manifest))
    image_dir = scene_dir / "images"
    mask_dir = scene_dir / "masks"
    for view in manifest.get("exported_views", []):
        image_name = Path(str(view["image_path"])).name
        mask_name = Path(str(view["mask_path"])).name
        local_image = image_dir / image_name
        local_mask = mask_dir / mask_name
        if local_image.is_file():
            view["image_path"] = str(local_image.resolve())
        if local_mask.is_file():
            view["mask_path"] = str(local_mask.resolve())
    return manifest


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_scene_dir = (
        Path(args.output_scene_dir).expanduser().resolve()
        if args.output_scene_dir
        else scene_dir
    )
    bundle_ref = Path(args.external_prior_bundle).expanduser().resolve()

    _copy_or_prepare_scene(scene_dir, output_scene_dir, overwrite=bool(args.overwrite))

    scene_manifest = load_scene_manifest(output_scene_dir)
    scene_manifest = _rewrite_manifest_asset_paths(scene_manifest, output_scene_dir)
    external_prior_bundle = load_external_prior_bundle(bundle_ref, scene_manifest)
    smplx_model_dir = resolve_smplx_model_dir(args.smplx_model_dir)

    prior_maps, prior_mask, prior_summary_tokens, smplx_params, _, prior_input_meta = build_external_prior_stack(
        scene_manifest=scene_manifest,
        target_size=int(args.target_size),
        mask_only=bool(args.mask_only),
        smplx_params=external_prior_bundle["smplx_params"],
        camera_params=external_prior_bundle["camera_params"],
        external_bundle_meta=external_prior_bundle["resolved_meta"],
        smplx_model_dir=smplx_model_dir,
        smplx_gender=args.smplx_gender,
        mesh_fill_knn=int(args.mesh_fill_knn),
        summary_token_count=int(args.summary_token_count),
        vertex_id_dim=int(args.vertex_id_dim),
        body_part_dim=int(args.body_part_dim),
        body_part_count=int(args.body_part_count),
    )

    prior_npz_path = output_scene_dir / "prior_maps.npz"
    if prior_npz_path.exists() and not args.overwrite:
        raise FileExistsError(f"{prior_npz_path} already exists. Re-run with --overwrite.")

    payload = {
        "prior_maps": prior_maps.astype(np.float16),
        "prior_mask": prior_mask.astype(bool),
        "prior_channels": np.asarray(prior_input_meta["channel_names"]),
        "prior_summary_channels": np.asarray(prior_input_meta.get("summary_channel_names", [])),
    }
    if prior_summary_tokens is not None:
        payload["prior_summary_tokens"] = prior_summary_tokens.astype(np.float16)
    np.savez_compressed(prior_npz_path, **payload)

    scene_manifest["prior_maps_file"] = prior_npz_path.name
    scene_manifest["prior_channels"] = prior_input_meta["channel_names"]
    scene_manifest["prior_summary_channels"] = prior_input_meta.get("summary_channel_names", [])
    scene_manifest["prior_summary_token_count"] = (
        0 if prior_summary_tokens is None else int(prior_summary_tokens.shape[1])
    )
    scene_manifest["prior_input_meta"] = prior_input_meta
    scene_manifest["external_prior_bundle"] = external_prior_bundle["resolved_meta"]
    scene_manifest["prior_source"] = "external_prior_bundle"
    scene_manifest["source_scene_dir"] = str(scene_dir)
    (output_scene_dir / "scene_manifest.json").write_text(
        json.dumps(scene_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "scene_dir": str(scene_dir),
        "output_scene_dir": str(output_scene_dir),
        "external_prior_bundle": external_prior_bundle["resolved_meta"],
        "prior_maps_file": str(prior_npz_path),
        "target_size": int(args.target_size),
        "mask_only": bool(args.mask_only),
        "prior_shape": list(prior_maps.shape),
        "prior_summary_shape": None if prior_summary_tokens is None else list(prior_summary_tokens.shape),
        "prior_channel_count": int(prior_maps.shape[1]),
        "smplx_keys": sorted(smplx_params.keys()),
    }
    (output_scene_dir / "external_prior_scene_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote scene-local prior bundle to {prior_npz_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
