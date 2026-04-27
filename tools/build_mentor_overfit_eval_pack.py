from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AssetSpec:
    category: str
    description: str
    source_path: Path
    relative_output_path: Path


def default_output_dir() -> Path:
    return REPO_ROOT / "reports" / f"mentor_overfit_eval_pack_{date.today().strftime('%Y%m%d')}"


def default_report_path() -> Path:
    return REPO_ROOT / "reports" / f"{date.today().strftime('%Y%m%d')}_mentor_overfit_eval_pack.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mentor-facing overfit eval pack from existing outputs only."
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for the packaged report and copied assets.",
    )
    parser.add_argument(
        "--report-path",
        default=str(default_report_path()),
        help="Short note written under reports/ with changed paths and pack location.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def pct(delta: float) -> str:
    return f"{delta * 100:.1f}%"


def span(summary: dict[str, Any], lo_key: str, hi_key: str) -> float:
    roi_summary = summary["roi_summary"]
    return float(roi_summary[hi_key]) - float(roi_summary[lo_key])


def copy_asset(output_dir: Path, asset: AssetSpec, inventory: list[dict[str, str]]) -> dict[str, str]:
    target_path = output_dir / asset.relative_output_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset.source_path, target_path)
    record = {
        "category": asset.category,
        "description": asset.description,
        "source_path": str(asset.source_path.resolve()),
        "copied_path": str(target_path.resolve()),
        "relative_output_path": asset.relative_output_path.as_posix(),
    }
    inventory.append(record)
    return record


def build_branch_record(
    *,
    branch: str,
    compare_dir: Path,
    trained_face_summary_path: Path,
    baseline_face_summary_path: Path,
) -> dict[str, Any]:
    comparison_summary = load_json(compare_dir / "comparison_summary.json")
    aligned_rows = comparison_summary["aligned_diff_summary_rows"]
    if len(aligned_rows) != 1:
        raise ValueError(f"Expected exactly one aligned diff row for {branch}, found {len(aligned_rows)}")
    diff_row = aligned_rows[0]

    variant_rows = {row["label"]: row for row in comparison_summary["variant_summary_rows"]}
    baseline_label = comparison_summary["baseline"]
    trained_label = next(label for label in variant_rows if label != baseline_label)

    baseline_row = variant_rows[baseline_label]
    trained_row = variant_rows[trained_label]
    baseline_face = load_json(baseline_face_summary_path)
    trained_face = load_json(trained_face_summary_path)

    baseline_z_span = span(baseline_face, "z_lo", "z_hi")
    trained_z_span = span(trained_face, "z_lo", "z_hi")
    baseline_x_span = span(baseline_face, "x_lo", "x_hi")
    trained_x_span = span(trained_face, "x_lo", "x_hi")

    return {
        "branch": branch,
        "compare_dir": str(compare_dir.resolve()),
        "comparison_summary_path": str((compare_dir / "comparison_summary.json").resolve()),
        "aligned_diff_csv_path": str((compare_dir / "aligned_diff_summary_vs_baseline.csv").resolve()),
        "baseline_label": baseline_label,
        "trained_label": trained_label,
        "depth_mae": float(diff_row["depth_mae"]),
        "world_points_l2_mean": float(diff_row["world_points_l2_mean"]),
        "normal_angle_mean_deg": float(diff_row["normal_angle_mean_deg"]),
        "translation_l2_mean": float(diff_row["translation_l2_mean"]),
        "depth_conf_mae": float(diff_row["depth_conf_mae"]),
        "world_points_conf_mae": float(diff_row["world_points_conf_mae"]),
        "baseline_depth_conf_fg_mean": float(baseline_row["depth_conf_fg_mean"]),
        "trained_depth_conf_fg_mean": float(trained_row["depth_conf_fg_mean"]),
        "baseline_world_points_conf_fg_mean": float(baseline_row["world_points_conf_fg_mean"]),
        "trained_world_points_conf_fg_mean": float(trained_row["world_points_conf_fg_mean"]),
        "baseline_face_points": int(baseline_face["roi_summary"]["points_after_roi"]),
        "trained_face_points": int(trained_face["roi_summary"]["points_after_roi"]),
        "baseline_face_z_span": baseline_z_span,
        "trained_face_z_span": trained_z_span,
        "baseline_face_x_span": baseline_x_span,
        "trained_face_x_span": trained_x_span,
        "baseline_face_y_lo": float(baseline_face["roi_summary"]["y_lo"]),
        "trained_face_y_lo": float(trained_face["roi_summary"]["y_lo"]),
        "baseline_face_summary_path": str(baseline_face_summary_path.resolve()),
        "trained_face_summary_path": str(trained_face_summary_path.resolve()),
        "baseline_face_summary": baseline_face,
        "trained_face_summary": trained_face,
    }


def build_pack_data() -> tuple[dict[str, Any], list[AssetSpec]]:
    crop_compare_dir = (
        REPO_ROOT / "output" / "modal_results" / "20260422_crop_trained_vs_untrained_compare"
    )
    softmatte_compare_dir = (
        REPO_ROOT / "output" / "modal_results" / "20260422_softmatte_trained_vs_untrained_compare"
    )
    overfit_root = REPO_ROOT / "output" / "overfit_trained_eval_20260422" / "open3d_compare"
    preprocess_root = REPO_ROOT / "output" / "preprocess_ablation_20260421" / "open3d_compare"

    crop_record = build_branch_record(
        branch="crop",
        compare_dir=crop_compare_dir,
        trained_face_summary_path=overfit_root / "crop_trained" / "face" / "open3d_summary.json",
        baseline_face_summary_path=preprocess_root / "crop" / "face" / "open3d_summary.json",
    )
    softmatte_record = build_branch_record(
        branch="softmatte",
        compare_dir=softmatte_compare_dir,
        trained_face_summary_path=overfit_root / "softmatte_trained" / "face" / "open3d_summary.json",
        baseline_face_summary_path=preprocess_root / "softmatte" / "face" / "open3d_summary.json",
    )

    branch_records = [crop_record, softmatte_record]
    for record in branch_records:
        record["face_point_delta"] = record["trained_face_points"] - record["baseline_face_points"]
        record["face_z_span_delta"] = record["trained_face_z_span"] - record["baseline_face_z_span"]
        record["face_z_span_change_ratio"] = record["face_z_span_delta"] / record["baseline_face_z_span"]
        record["depth_conf_fg_ratio"] = (
            record["trained_depth_conf_fg_mean"] / record["baseline_depth_conf_fg_mean"]
        )
        record["world_points_conf_fg_ratio"] = (
            record["trained_world_points_conf_fg_mean"] / record["baseline_world_points_conf_fg_mean"]
        )

    assets = [
        AssetSpec(
            category="baseline_panels",
            description="Preprocess ablation face ROI comparison across full/crop/hardmask/softmatte",
            source_path=preprocess_root / "compare_face_roi_variants.png",
            relative_output_path=Path("assets/preprocess/compare_face_roi_variants.png"),
        ),
        AssetSpec(
            category="crop_compare",
            description="Crop branch aligned depth preview versus crop_untrained baseline",
            source_path=crop_compare_dir / "preview_sheets" / "depth_aligned_to_crop_untrained.png",
            relative_output_path=Path("assets/crop_compare/depth_aligned_to_crop_untrained.png"),
        ),
        AssetSpec(
            category="crop_compare",
            description="Crop branch aligned point confidence preview versus crop_untrained baseline",
            source_path=crop_compare_dir / "preview_sheets" / "point_conf_aligned_to_crop_untrained.png",
            relative_output_path=Path("assets/crop_compare/point_conf_aligned_to_crop_untrained.png"),
        ),
        AssetSpec(
            category="crop_open3d",
            description="Crop trained face ROI close render",
            source_path=overfit_root / "crop_trained" / "face" / "face_close.png",
            relative_output_path=Path("assets/crop_open3d/face_close.png"),
        ),
        AssetSpec(
            category="crop_open3d",
            description="Crop trained face ROI side render",
            source_path=overfit_root / "crop_trained" / "face" / "side.png",
            relative_output_path=Path("assets/crop_open3d/side.png"),
        ),
        AssetSpec(
            category="softmatte_compare",
            description="Softmatte branch aligned depth preview versus softmatte_untrained baseline",
            source_path=softmatte_compare_dir / "preview_sheets" / "depth_aligned_to_softmatte_untrained.png",
            relative_output_path=Path("assets/softmatte_compare/depth_aligned_to_softmatte_untrained.png"),
        ),
        AssetSpec(
            category="softmatte_compare",
            description="Softmatte branch aligned point confidence preview versus softmatte_untrained baseline",
            source_path=softmatte_compare_dir / "preview_sheets" / "point_conf_aligned_to_softmatte_untrained.png",
            relative_output_path=Path("assets/softmatte_compare/point_conf_aligned_to_softmatte_untrained.png"),
        ),
        AssetSpec(
            category="softmatte_open3d",
            description="Softmatte trained face ROI close render",
            source_path=overfit_root / "softmatte_trained" / "face" / "face_close.png",
            relative_output_path=Path("assets/softmatte_open3d/face_close.png"),
        ),
        AssetSpec(
            category="softmatte_open3d",
            description="Softmatte trained face ROI side render",
            source_path=overfit_root / "softmatte_trained" / "face" / "side.png",
            relative_output_path=Path("assets/softmatte_open3d/side.png"),
        ),
        AssetSpec(
            category="evidence",
            description="Crop trained vs untrained comparison summary JSON",
            source_path=crop_compare_dir / "comparison_summary.json",
            relative_output_path=Path("evidence/crop_trained_vs_untrained/comparison_summary.json"),
        ),
        AssetSpec(
            category="evidence",
            description="Crop trained vs untrained aligned diff summary CSV",
            source_path=crop_compare_dir / "aligned_diff_summary_vs_baseline.csv",
            relative_output_path=Path("evidence/crop_trained_vs_untrained/aligned_diff_summary_vs_baseline.csv"),
        ),
        AssetSpec(
            category="evidence",
            description="Softmatte trained vs untrained comparison summary JSON",
            source_path=softmatte_compare_dir / "comparison_summary.json",
            relative_output_path=Path("evidence/softmatte_trained_vs_untrained/comparison_summary.json"),
        ),
        AssetSpec(
            category="evidence",
            description="Softmatte trained vs untrained aligned diff summary CSV",
            source_path=softmatte_compare_dir / "aligned_diff_summary_vs_baseline.csv",
            relative_output_path=Path("evidence/softmatte_trained_vs_untrained/aligned_diff_summary_vs_baseline.csv"),
        ),
        AssetSpec(
            category="evidence",
            description="Crop trained face ROI Open3D summary JSON",
            source_path=overfit_root / "crop_trained" / "face" / "open3d_summary.json",
            relative_output_path=Path("evidence/open3d/crop_trained_face_open3d_summary.json"),
        ),
        AssetSpec(
            category="evidence",
            description="Crop untrained face ROI Open3D summary JSON",
            source_path=preprocess_root / "crop" / "face" / "open3d_summary.json",
            relative_output_path=Path("evidence/open3d/crop_untrained_face_open3d_summary.json"),
        ),
        AssetSpec(
            category="evidence",
            description="Softmatte trained face ROI Open3D summary JSON",
            source_path=overfit_root / "softmatte_trained" / "face" / "open3d_summary.json",
            relative_output_path=Path("evidence/open3d/softmatte_trained_face_open3d_summary.json"),
        ),
        AssetSpec(
            category="evidence",
            description="Softmatte untrained face ROI Open3D summary JSON",
            source_path=preprocess_root / "softmatte" / "face" / "open3d_summary.json",
            relative_output_path=Path("evidence/open3d/softmatte_untrained_face_open3d_summary.json"),
        ),
    ]

    return {"generated_at": datetime.now().isoformat(timespec="seconds"), "branches": branch_records}, assets


def build_readme_text(output_dir: Path) -> str:
    return "\n".join(
        [
            "# Mentor Overfit Eval Pack",
            "",
            "- Scope: copied from existing outputs already present in the workspace.",
            "- No new training, inference, or Open3D rendering was run for this pack.",
            f"- Main report: `{(output_dir / 'mentor_overfit_eval_report.md').relative_to(output_dir).as_posix()}`",
            f"- Manifest: `{(output_dir / 'pack_manifest.json').relative_to(output_dir).as_posix()}`",
            "",
            "## Quick Read",
            "",
            "- Both single-case overfit checkpoints remain below mentor-final quality.",
            "- In both branches, the face ROI still collapses toward planar or silhouette-like geometry in the copied Open3D renders.",
            "- The largest numeric change is confidence inflation, not recovered facial volume or denser retained face ROI points.",
            "",
            "Open `mentor_overfit_eval_report.md` for the concise summary table, copied visuals, and source paths.",
        ]
    )


def build_report_text(data: dict[str, Any]) -> str:
    crop = next(record for record in data["branches"] if record["branch"] == "crop")
    softmatte = next(record for record in data["branches"] if record["branch"] == "softmatte")

    summary_lines = [
        "| Branch | Depth MAE vs untrained | World-point L2 mean | Mean normal angle | Mean translation L2 | FG depth conf mean | FG world-point conf mean | Face ROI points | Face ROI z-span |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for record in data["branches"]:
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record['branch']}`",
                    fmt(record["depth_mae"]),
                    fmt(record["world_points_l2_mean"]),
                    fmt(record["normal_angle_mean_deg"]),
                    fmt(record["translation_l2_mean"]),
                    f"{fmt(record['baseline_depth_conf_fg_mean'])} -> {fmt(record['trained_depth_conf_fg_mean'])}",
                    f"{fmt(record['baseline_world_points_conf_fg_mean'])} -> {fmt(record['trained_world_points_conf_fg_mean'])}",
                    f"{fmt(record['baseline_face_points'], 0)} -> {fmt(record['trained_face_points'], 0)}",
                    f"{fmt(record['baseline_face_z_span'])} -> {fmt(record['trained_face_z_span'])}",
                ]
            )
            + " |"
        )

    asset_lines = [
        "- `assets/preprocess/compare_face_roi_variants.png`",
        "- `assets/crop_compare/depth_aligned_to_crop_untrained.png`",
        "- `assets/crop_compare/point_conf_aligned_to_crop_untrained.png`",
        "- `assets/crop_open3d/face_close.png`",
        "- `assets/crop_open3d/side.png`",
        "- `assets/softmatte_compare/depth_aligned_to_softmatte_untrained.png`",
        "- `assets/softmatte_compare/point_conf_aligned_to_softmatte_untrained.png`",
        "- `assets/softmatte_open3d/face_close.png`",
        "- `assets/softmatte_open3d/side.png`",
    ]

    lines = [
        "# Mentor Overfit Eval Report",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from existing outputs only.",
        "",
        "## Bottom Line",
        "",
        "- The single-case overfit checkpoints do not meet mentor-final quality.",
        "- Both trained branches still collapse the face ROI toward thin planar or silhouette-like geometry in the copied `face_close` and `side` Open3D renders.",
        "- The main numeric change is confidence inflation, not stronger 3D face structure: foreground confidence means jump from about `1.0` to `30+`, while retained face ROI points decrease in both branches.",
        "",
        "## Key Metrics",
        "",
        *summary_lines,
        "",
        f"- `crop`: face ROI points go from `{fmt(crop['baseline_face_points'], 0)}` to `{fmt(crop['trained_face_points'], 0)}`, while the exported ROI z-span drops from `{fmt(crop['baseline_face_z_span'])}` to `{fmt(crop['trained_face_z_span'])}` (`{pct(-crop['face_z_span_change_ratio'])}` narrower).",
        f"- `softmatte`: face ROI points go from `{fmt(softmatte['baseline_face_points'], 0)}` to `{fmt(softmatte['trained_face_points'], 0)}`, while the exported ROI z-span drops from `{fmt(softmatte['baseline_face_z_span'])}` to `{fmt(softmatte['trained_face_z_span'])}` (`{pct(-softmatte['face_z_span_change_ratio'])}` narrower).",
        "- Inference from the summaries plus the side renders: the trained exports are not recovering facial volume; they are staying thin while moving confidence upward.",
        "",
        "## Visual Evidence",
        "",
        "Baseline face ROI context from the preprocess ablation:",
        "",
        "![Preprocess face ROI comparison](assets/preprocess/compare_face_roi_variants.png)",
        "",
        "Crop overfit branch:",
        "",
        "![Crop aligned depth](assets/crop_compare/depth_aligned_to_crop_untrained.png)",
        "",
        "![Crop aligned point confidence](assets/crop_compare/point_conf_aligned_to_crop_untrained.png)",
        "",
        "![Crop trained face close](assets/crop_open3d/face_close.png)",
        "",
        "![Crop trained face side](assets/crop_open3d/side.png)",
        "",
        "Softmatte overfit branch:",
        "",
        "![Softmatte aligned depth](assets/softmatte_compare/depth_aligned_to_softmatte_untrained.png)",
        "",
        "![Softmatte aligned point confidence](assets/softmatte_compare/point_conf_aligned_to_softmatte_untrained.png)",
        "",
        "![Softmatte trained face close](assets/softmatte_open3d/face_close.png)",
        "",
        "![Softmatte trained face side](assets/softmatte_open3d/side.png)",
        "",
        "## Evidence Included In This Pack",
        "",
        *asset_lines,
        "- `evidence/crop_trained_vs_untrained/comparison_summary.json`",
        "- `evidence/crop_trained_vs_untrained/aligned_diff_summary_vs_baseline.csv`",
        "- `evidence/softmatte_trained_vs_untrained/comparison_summary.json`",
        "- `evidence/softmatte_trained_vs_untrained/aligned_diff_summary_vs_baseline.csv`",
        "- `evidence/open3d/crop_trained_face_open3d_summary.json`",
        "- `evidence/open3d/crop_untrained_face_open3d_summary.json`",
        "- `evidence/open3d/softmatte_trained_face_open3d_summary.json`",
        "- `evidence/open3d/softmatte_untrained_face_open3d_summary.json`",
        "",
        "## Source Paths",
        "",
        f"- `crop` comparison summary: `{repo_rel(Path(crop['comparison_summary_path']))}`",
        f"- `softmatte` comparison summary: `{repo_rel(Path(softmatte['comparison_summary_path']))}`",
        f"- `crop` trained face ROI summary: `{repo_rel(Path(crop['trained_face_summary_path']))}`",
        f"- `crop` untrained face ROI summary: `{repo_rel(Path(crop['baseline_face_summary_path']))}`",
        f"- `softmatte` trained face ROI summary: `{repo_rel(Path(softmatte['trained_face_summary_path']))}`",
        f"- `softmatte` untrained face ROI summary: `{repo_rel(Path(softmatte['baseline_face_summary_path']))}`",
        "",
        "## Limitations",
        "",
        "- This is a single-case, trained-vs-untrained comparison. It is not a generalization result and it does not use external ground truth.",
        "- The dense deltas are relative to each branch's own untrained baseline, not to a calibrated geometry target.",
        "- The planar-collapse statement is based on the copied Open3D renders and the narrower exported ROI z-span, not on a new reconstruction metric.",
    ]
    return "\n".join(lines)


def build_report_note_text(output_dir: Path, report_path: Path) -> str:
    lines = [
        f"# {date.today().strftime('%Y-%m-%d')} Mentor Overfit Eval Pack",
        "",
        "- Scope: built from existing outputs only; no new model runs or renders.",
        f"- Changed files: `tools/build_mentor_overfit_eval_pack.py`, `{repo_rel(report_path)}`",
        f"- Main pack directory: `{repo_rel(output_dir)}`",
        "",
        "## Truthful Mentor Read",
        "",
        "- The single-case overfit checkpoints do not meet mentor-final quality.",
        "- Both trained branches still visually collapse the face ROI toward planar or silhouette-like geometry.",
        "- Confidence values rise sharply, but retained face ROI points and exported z-span do not improve.",
        "",
        "## Generated Outputs",
        "",
        f"- `{repo_rel(output_dir / 'README.md')}`",
        f"- `{repo_rel(output_dir / 'mentor_overfit_eval_report.md')}`",
        f"- `{repo_rel(output_dir / 'pack_manifest.json')}`",
        f"- `{repo_rel(output_dir / 'assets/preprocess/compare_face_roi_variants.png')}`",
        f"- `{repo_rel(output_dir / 'assets/crop_open3d/face_close.png')}`",
        f"- `{repo_rel(output_dir / 'assets/crop_open3d/side.png')}`",
        f"- `{repo_rel(output_dir / 'assets/softmatte_open3d/face_close.png')}`",
        f"- `{repo_rel(output_dir / 'assets/softmatte_open3d/side.png')}`",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, assets = build_pack_data()
    copied_assets: list[dict[str, str]] = []
    for asset in assets:
        copy_asset(output_dir, asset, copied_assets)

    manifest = {
        "generated_at": data["generated_at"],
        "generator": str(Path(__file__).resolve()),
        "scope": "existing outputs only",
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "summary": data,
        "copied_assets": copied_assets,
    }

    write_text(output_dir / "README.md", build_readme_text(output_dir))
    write_text(output_dir / "mentor_overfit_eval_report.md", build_report_text(data))
    save_json(output_dir / "pack_manifest.json", manifest)
    write_text(report_path, build_report_note_text(output_dir, report_path))

    summary = {
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "copied_files": [asset["relative_output_path"] for asset in copied_assets],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
