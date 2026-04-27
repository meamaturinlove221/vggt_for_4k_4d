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


@dataclass(frozen=True)
class VariantSpec:
    label: str
    open3d_key: str


@dataclass(frozen=True)
class EvalSpec:
    label: str
    run_dir: Path
    teacher_note: str


def default_output_dir() -> Path:
    return REPO_ROOT / "output" / f"mentor_benchmark_pack_{date.today().strftime('%Y%m%d')}"


def default_report_path() -> Path:
    return REPO_ROOT / "reports" / f"{date.today().strftime('%Y%m%d')}_mentor_benchmark_pack.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mentor-facing benchmark pack from existing sparse-view outputs only."
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for the consolidated mentor pack.",
    )
    parser.add_argument(
        "--report-path",
        default=str(default_report_path()),
        help="Concise note written under reports/ with changed files and generated outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def build_preprocess_section() -> tuple[dict[str, Any], list[AssetSpec]]:
    comparison_summary_path = (
        REPO_ROOT
        / "output"
        / "modal_results"
        / "20260421_6views_preprocess_ablation_compare_b40_softmatte"
        / "comparison_summary.json"
    )
    comparison_summary = load_json(comparison_summary_path)
    variant_rows = {row["label"]: row for row in comparison_summary["variant_summary_rows"]}
    diff_rows = {row["label"]: row for row in comparison_summary["aligned_diff_summary_rows"]}

    variants = [
        VariantSpec(label="full", open3d_key="full"),
        VariantSpec(label="human_crop", open3d_key="crop"),
        VariantSpec(label="human_crop_hardmask", open3d_key="hardmask"),
        VariantSpec(label="human_crop_softmatte", open3d_key="softmatte"),
    ]
    open3d_root = REPO_ROOT / "output" / "preprocess_ablation_20260421" / "open3d_compare"

    variant_table: list[dict[str, Any]] = []
    for spec in variants:
        full_summary = load_json(open3d_root / spec.open3d_key / "full" / "open3d_summary.json")
        head_summary = load_json(open3d_root / spec.open3d_key / "head" / "open3d_summary.json")
        face_summary = load_json(open3d_root / spec.open3d_key / "face" / "open3d_summary.json")
        row = {
            "label": spec.label,
            "mask_coverage_mean": variant_rows[spec.label]["mask_coverage_mean"],
            "crop_area_ratio_mean": variant_rows[spec.label]["crop_area_ratio_mean"],
            "full_points": full_summary["roi_summary"]["points_after_roi"],
            "head_points": head_summary["roi_summary"]["points_after_roi"],
            "face_points": face_summary["roi_summary"]["points_after_roi"],
            "diff_vs_full": diff_rows.get(spec.label),
        }
        variant_table.append(row)

    comparison_assets = [
        AssetSpec(
            category="preprocess_panels",
            description="Head ROI Open3D comparison across full/crop/hardmask/softmatte preprocess variants",
            source_path=open3d_root / "compare_head_roi_variants.png",
            relative_output_path=Path("assets/preprocess/compare_head_roi_variants.png"),
        ),
        AssetSpec(
            category="preprocess_panels",
            description="Face ROI Open3D comparison across full/crop/hardmask/softmatte preprocess variants",
            source_path=open3d_root / "compare_face_roi_variants.png",
            relative_output_path=Path("assets/preprocess/compare_face_roi_variants.png"),
        ),
        AssetSpec(
            category="preprocess_panels",
            description="Full-body plus face-close Open3D comparison across preprocess variants",
            source_path=open3d_root / "compare_fullbody_faceclose_variants.png",
            relative_output_path=Path("assets/preprocess/compare_fullbody_faceclose_variants.png"),
        ),
    ]

    non_baseline = [row for row in variant_table if row["diff_vs_full"]]
    stability_metrics = ["depth_mae", "world_points_l2_mean", "normal_angle_mean_deg", "translation_l2_mean"]
    metric_winners = {
        metric: min(non_baseline, key=lambda row: float(row["diff_vs_full"][metric]))["label"] for metric in stability_metrics
    }
    roi_density_winners = {
        "head": max(non_baseline, key=lambda row: int(row["head_points"]))["label"],
        "face": max(non_baseline, key=lambda row: int(row["face_points"]))["label"],
    }

    return (
        {
            "comparison_summary_path": str(comparison_summary_path.resolve()),
            "variant_table": variant_table,
            "metric_winners": metric_winners,
            "roi_density_winners": roi_density_winners,
        },
        comparison_assets,
    )


def build_focus_open3d_section() -> tuple[dict[str, Any], list[AssetSpec]]:
    focus_root = REPO_ROOT / "output" / "comparisons" / "20260421_6views_sparseproto_from_6view_focus_smoke_r1"
    head_summary_path = focus_root / "world_points_head_roi" / "open3d_summary.json"
    face_summary_path = focus_root / "world_points_face_roi" / "open3d_summary.json"
    full_summary_path = focus_root / "world_points_human" / "open3d_summary.json"
    head_summary = load_json(head_summary_path)
    face_summary = load_json(face_summary_path)
    full_summary = load_json(full_summary_path)

    assets = [
        AssetSpec(
            category="current_open3d",
            description="Current 6-view sparseproto head ROI close render",
            source_path=focus_root / "world_points_head_roi" / "head_close.png",
            relative_output_path=Path("assets/open3d/current_sparseproto_head_close.png"),
        ),
        AssetSpec(
            category="current_open3d",
            description="Current 6-view sparseproto face ROI close render",
            source_path=focus_root / "world_points_face_roi" / "face_close.png",
            relative_output_path=Path("assets/open3d/current_sparseproto_face_close.png"),
        ),
    ]
    return (
        {
            "head_summary_path": str(head_summary_path.resolve()),
            "face_summary_path": str(face_summary_path.resolve()),
            "full_summary_path": str(full_summary_path.resolve()),
            "full_points_after_conf": full_summary["summary"]["points_after_conf"],
            "head_points_after_roi": head_summary["roi_summary"]["points_after_roi"],
            "face_points_after_roi": face_summary["roi_summary"]["points_after_roi"],
        },
        assets,
    )


def build_detail_section() -> tuple[dict[str, Any], list[AssetSpec]]:
    detail_root = REPO_ROOT / "output" / "detail_normal_refiner_20260421"
    head_run_dir = detail_root / "remote_head_60to6v_e50"
    face_run_dir = detail_root / "remote_face_60to6v_e50"

    head_best = load_json(head_run_dir / "best_metrics.json")
    face_best = load_json(face_run_dir / "best_metrics.json")
    head_run_summary = load_json(head_run_dir / "run_summary.json")
    face_run_summary = load_json(face_run_dir / "run_summary.json")

    eval_specs = [
        EvalSpec(label="6v", run_dir=detail_root / "eval_head_6v_teacher60", teacher_note="teacher60"),
        EvalSpec(label="7v", run_dir=detail_root / "eval_head_7v_teacherlocal", teacher_note="teacherlocal"),
        EvalSpec(label="12v", run_dir=detail_root / "eval_head_12v_teacher60", teacher_note="teacher60"),
        EvalSpec(label="13v", run_dir=detail_root / "eval_head_13v_teacherlocal", teacher_note="teacherlocal"),
        EvalSpec(label="20v", run_dir=detail_root / "eval_head_20v_teacher60", teacher_note="teacher60"),
    ]
    eval_rows = []
    for spec in eval_specs:
        payload = load_json(spec.run_dir / "run_summary.json")
        eval_rows.append(
            {
                "label": spec.label,
                "run_dir": str(spec.run_dir.resolve()),
                "teacher_note": spec.teacher_note,
                "num_samples": payload["num_samples"],
                "loss_detail_normal_total": payload["mean_metrics"]["loss_detail_normal_total"],
                "loss_detail_normal_cosine": payload["mean_metrics"]["loss_detail_normal_cosine"],
                "loss_detail_normal_edge": payload["mean_metrics"]["loss_detail_normal_edge"],
                "loss_detail_normal_mask_restricted": payload["mean_metrics"]["loss_detail_normal_mask_restricted"],
            }
        )

    assets = [
        AssetSpec(
            category="detail_normal",
            description="Head ROI refiner best-val summary strip for target camera",
            source_path=head_run_dir / "best_val" / "visuals" / "00_00_tgt_cam00_summary_strip.png",
            relative_output_path=Path("assets/detail_normal/head_best_val_tgt_cam00_summary_strip.png"),
        ),
        AssetSpec(
            category="detail_normal",
            description="Head ROI refiner best-val summary strip for off-axis source camera 30",
            source_path=head_run_dir / "best_val" / "visuals" / "03_30_src_cam30_summary_strip.png",
            relative_output_path=Path("assets/detail_normal/head_best_val_src_cam30_summary_strip.png"),
        ),
        AssetSpec(
            category="detail_normal",
            description="Face ROI refiner best-val summary strip for target camera",
            source_path=face_run_dir / "best_val" / "visuals" / "00_00_tgt_cam00_summary_strip.png",
            relative_output_path=Path("assets/detail_normal/face_best_val_tgt_cam00_summary_strip.png"),
        ),
        AssetSpec(
            category="detail_normal",
            description="Face ROI refiner best-val summary strip for off-axis source camera 30",
            source_path=face_run_dir / "best_val" / "visuals" / "03_30_src_cam30_summary_strip.png",
            relative_output_path=Path("assets/detail_normal/face_best_val_src_cam30_summary_strip.png"),
        ),
    ]

    return (
        {
            "detail_root": str(detail_root.resolve()),
            "head_best_metrics_path": str((head_run_dir / "best_metrics.json").resolve()),
            "face_best_metrics_path": str((face_run_dir / "best_metrics.json").resolve()),
            "head_best": head_best,
            "face_best": face_best,
            "head_run_summary_path": str((head_run_dir / "run_summary.json").resolve()),
            "face_run_summary_path": str((face_run_dir / "run_summary.json").resolve()),
            "head_run_summary": head_run_summary,
            "face_run_summary": face_run_summary,
            "eval_rows": eval_rows,
        },
        assets,
    )


def build_readme_text(output_dir: Path, pack_manifest_path: Path) -> str:
    return "\n".join(
        [
            "# Mentor Benchmark Pack",
            "",
            "- Scope: assembled from existing outputs already present in the workspace.",
            "- No new training, inference, or Open3D rendering was run for this pack.",
            f"- Main report: `{(output_dir / 'mentor_benchmark_report.md').relative_to(output_dir).as_posix()}`",
            f"- Pack manifest: `{pack_manifest_path.relative_to(output_dir).as_posix()}`",
            "",
            "## Quick Read",
            "",
            "- `human_crop` remains the most stable sparse-view preprocess branch relative to `full` on this 6-view case.",
            "- `human_crop_softmatte` currently produces the densest head/face ROI point clouds, but with larger aligned deltas than plain `human_crop`.",
            "- `coarse prior normal + ROI-first detail_normal_refiner` is the current validated normal path, but it is not yet a mentor-final point-cloud result.",
            "",
            "Open `mentor_benchmark_report.md` for the compact tables, copied visuals, and source-of-truth paths.",
        ]
    )


def build_report_text(
    preprocess: dict[str, Any],
    focus_open3d: dict[str, Any],
    detail: dict[str, Any],
    copied_assets: list[dict[str, str]],
    source_reports: list[Path],
) -> str:
    preprocess_assets = [asset for asset in copied_assets if asset["category"] == "preprocess_panels"]
    open3d_assets = [asset for asset in copied_assets if asset["category"] == "current_open3d"]
    detail_assets = [asset for asset in copied_assets if asset["category"] == "detail_normal"]

    variant_lines = [
        "| Variant | Full pts | Head ROI pts | Face ROI pts | Mask cov. | Crop area | Depth MAE vs full | World-point L2 vs full | Normal angle vs full | Translation L2 vs full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in preprocess["variant_table"]:
        diff = row["diff_vs_full"]
        variant_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    fmt(row["full_points"], 0),
                    fmt(row["head_points"], 0),
                    fmt(row["face_points"], 0),
                    fmt(row["mask_coverage_mean"]),
                    fmt(row["crop_area_ratio_mean"]),
                    "-" if diff is None else fmt(diff["depth_mae"]),
                    "-" if diff is None else fmt(diff["world_points_l2_mean"]),
                    "-" if diff is None else fmt(diff["normal_angle_mean_deg"]),
                    "-" if diff is None else fmt(diff["translation_l2_mean"]),
                ]
            )
            + " |"
        )

    detail_head = detail["head_best"]
    detail_face = detail["face_best"]
    detail_lines = [
        "| ROI | Best epoch | Best val total | Cosine | Edge | Mask-restricted | Hairline cosine | Ear-band cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| `head` | "
        + " | ".join(
            [
                fmt(detail_head["best_epoch"], 0),
                fmt(detail_head["best_val_loss"]),
                fmt(detail_head["best_val_metrics"]["loss_detail_normal_cosine"]),
                fmt(detail_head["best_val_metrics"]["loss_detail_normal_edge"]),
                fmt(detail_head["best_val_metrics"]["loss_detail_normal_mask_restricted"]),
                fmt(detail_head["best_val_metrics"]["metric_hairline_cosine"]),
                fmt(detail_head["best_val_metrics"]["metric_ear_band_cosine"]),
            ]
        )
        + " |",
        "| `face` | "
        + " | ".join(
            [
                fmt(detail_face["best_epoch"], 0),
                fmt(detail_face["best_val_loss"]),
                fmt(detail_face["best_val_metrics"]["loss_detail_normal_cosine"]),
                fmt(detail_face["best_val_metrics"]["loss_detail_normal_edge"]),
                fmt(detail_face["best_val_metrics"]["loss_detail_normal_mask_restricted"]),
                fmt(detail_face["best_val_metrics"]["metric_hairline_cosine"]),
                fmt(detail_face["best_val_metrics"]["metric_ear_band_cosine"]),
            ]
        )
        + " |",
    ]

    eval_lines = [
        "| Eval set | Teacher source | Samples | Total | Cosine | Edge | Mask-restricted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in detail["eval_rows"]:
        eval_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    f"`{row['teacher_note']}`",
                    fmt(row["num_samples"], 0),
                    fmt(row["loss_detail_normal_total"]),
                    fmt(row["loss_detail_normal_cosine"]),
                    fmt(row["loss_detail_normal_edge"]),
                    fmt(row["loss_detail_normal_mask_restricted"]),
                ]
            )
            + " |"
        )

    lines = [
        "# Mentor Benchmark Report",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from existing workspace outputs only.",
        "",
        "## Scope",
        "",
        "- This pack does not fabricate improvements or rerun experiments.",
        "- It consolidates the current 6-view preprocess comparison, ROI Open3D evidence, and detail-normal status already present in the repo.",
        "- Current mentor-final gap remains open: the existing reports still say the work is not at the mentor-final bar yet.",
        "",
        "## 1. Sparse-View Preprocess Snapshot",
        "",
        f"Source summary: `{repo_rel(Path(preprocess['comparison_summary_path']))}`",
        "",
        *variant_lines,
        "",
        "Current truthful read from the available outputs:",
        "",
        f"- `human_crop` is the same winner on all four aligned-stability metrics tracked here: depth MAE, world-point L2, normal angle, and translation L2.",
        f"- `human_crop_softmatte` is the densest ROI branch in both `head` and `face` point counts for this case.",
        f"- Copied comparison panels: `{preprocess_assets[0]['relative_output_path']}`, `{preprocess_assets[1]['relative_output_path']}`, `{preprocess_assets[2]['relative_output_path']}`.",
        "",
        "## 2. Current ROI Open3D Renders",
        "",
        f"Source head summary: `{repo_rel(Path(focus_open3d['head_summary_path']))}`",
        f"Source face summary: `{repo_rel(Path(focus_open3d['face_summary_path']))}`",
        "",
        f"- Current 6-view sparseproto focus smoke keeps `{fmt(focus_open3d['full_points_after_conf'], 0)}` human points after confidence filtering.",
        f"- Within that same export, the retained ROI counts are `{fmt(focus_open3d['head_points_after_roi'], 0)}` for `head` and `{fmt(focus_open3d['face_points_after_roi'], 0)}` for `face`.",
        f"- Copied close renders: `{open3d_assets[0]['relative_output_path']}`, `{open3d_assets[1]['relative_output_path']}`.",
        "",
        "## 3. Detail-Normal Status",
        "",
        f"Head best metrics: `{repo_rel(Path(detail['head_best_metrics_path']))}`",
        f"Face best metrics: `{repo_rel(Path(detail['face_best_metrics_path']))}`",
        "",
        *detail_lines,
        "",
        "Head checkpoint reuse across sparse-view exports:",
        "",
        *eval_lines,
        "",
        "Notes for interpretation:",
        "",
        "- `6v / 12v / 20v` use `teacher60` exports; `7v / 13v` use the older `teacherlocal` exports and should be read as stability checks rather than strict apples-to-apples ranking.",
        "- The current validated path is still `coarse prior normal + ROI-first detail_normal_refiner`, not the older global sparse-view normal head branch.",
        f"- Copied representative strips: `{detail_assets[0]['relative_output_path']}`, `{detail_assets[1]['relative_output_path']}`, `{detail_assets[2]['relative_output_path']}`, `{detail_assets[3]['relative_output_path']}`.",
        "",
        "## 4. Mentor-Facing Summary",
        "",
        "- `human_crop` is the safest default sparse-view preprocess from the current 6-view benchmark evidence.",
        "- `human_crop_softmatte` is worth keeping as a denser ROI branch, but the current outputs do not support promoting it above `human_crop` yet.",
        "- ROI Open3D head/face renders are now a real acceptance view, and this pack includes both the preprocess-variant comparisons and the current sparseproto close renders.",
        "- Detail-normal refinement is real and measurable on ROI exports, but the workspace does not yet prove a mentor-final jump in final face/head point-cloud quality.",
        "",
        "## Source Reports",
        "",
    ]
    for path in source_reports:
        lines.append(f"- `{repo_rel(path)}`")
    return "\n".join(lines)


def build_report_note_text(
    output_dir: Path,
    report_path: Path,
    pack_manifest_path: Path,
    copied_assets: list[dict[str, str]],
) -> str:
    key_outputs = [
        output_dir / "README.md",
        output_dir / "mentor_benchmark_report.md",
        pack_manifest_path,
    ]
    key_outputs.extend(output_dir / Path(asset["relative_output_path"]) for asset in copied_assets[:4])
    unique_outputs = []
    seen = set()
    for path in key_outputs:
        resolved = path.resolve()
        if resolved not in seen:
            unique_outputs.append(path)
            seen.add(resolved)

    lines = [
        "# 2026-04-22 Mentor Benchmark Pack",
        "",
        "- Scope: consolidated from existing outputs only; no new model runs or renders.",
        f"- Changed files: `tools/build_mentor_benchmark_pack.py`, `{repo_rel(report_path)}`",
        f"- Main pack directory: `{repo_rel(output_dir)}`",
        "",
        "## Generated Outputs",
        "",
    ]
    for path in unique_outputs:
        lines.append(f"- `{repo_rel(path)}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The pack ties together the preprocess-variant benchmark table, ROI Open3D evidence, and detail-normal status in one report.",
            "- The report keeps the current truthful read: `human_crop` is the stable default, `human_crop_softmatte` is the densest ROI branch, and ROI detail refinement is promising but not yet mentor-final.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess, preprocess_assets = build_preprocess_section()
    focus_open3d, open3d_assets = build_focus_open3d_section()
    detail, detail_assets = build_detail_section()

    copied_assets: list[dict[str, str]] = []
    for asset in [*preprocess_assets, *open3d_assets, *detail_assets]:
        copy_asset(output_dir, asset, copied_assets)

    source_reports = [
        REPO_ROOT / "reports" / "20260421_6views_preprocess_ablation_comparison.md",
        REPO_ROOT / "reports" / "20260421_sparse_view_detail_normal_status.md",
        REPO_ROOT / "reports" / "20260422_mentor_taskboard_status.md",
    ]

    pack_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": str(Path(__file__).resolve()),
        "scope": "existing outputs only",
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "source_reports": [str(path.resolve()) for path in source_reports],
        "preprocess": preprocess,
        "focus_open3d": focus_open3d,
        "detail_normal": detail,
        "copied_assets": copied_assets,
    }
    pack_manifest_path = output_dir / "pack_manifest.json"

    write_text(output_dir / "README.md", build_readme_text(output_dir, pack_manifest_path))
    write_text(
        output_dir / "mentor_benchmark_report.md",
        build_report_text(preprocess, focus_open3d, detail, copied_assets, source_reports),
    )
    save_json(pack_manifest_path, pack_manifest)
    write_text(
        report_path,
        build_report_note_text(output_dir, report_path, pack_manifest_path, copied_assets),
    )

    summary = {
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "pack_manifest": str(pack_manifest_path),
        "copied_assets": [asset["relative_output_path"] for asset in copied_assets],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
