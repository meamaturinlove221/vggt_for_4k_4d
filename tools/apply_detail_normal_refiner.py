from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.detail_normal_refiner_loss import compute_detail_normal_refiner_loss  # noqa: E402
from vggt.models.detail_normal_refiner import DetailNormalRefiner  # noqa: E402
from vggt.utils.normal_refiner import normal_to_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained detail_normal_refiner checkpoint to an ROI dataset.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train_detail_normal_refiner.py")
    parser.add_argument("--dataset-npz", required=True, help="ROI dataset exported by export_detail_normal_refiner_dataset.py")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualize-count", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RefinerNpzDataset(Dataset):
    def __init__(self, dataset_npz: str | Path, *, max_samples: int = 0) -> None:
        payload = np.load(Path(dataset_npz).expanduser().resolve(), allow_pickle=False)
        self.dataset_npz = str(Path(dataset_npz).expanduser().resolve())
        limit = int(max_samples) if max_samples and int(max_samples) > 0 else int(payload["rgb"].shape[0])
        self.rgb = payload["rgb"][:limit]
        self.human_mask = payload["human_mask"][:limit]
        self.coarse_prior_normal = payload["coarse_prior_normal"][:limit]
        self.teacher_normal = payload["teacher_normal"][:limit]
        self.teacher_mask = payload["teacher_mask"][:limit]
        self.hairline_mask = payload["hairline_mask"][:limit]
        self.ear_band_mask = payload["ear_band_mask"][:limit]
        self.view_name = payload["view_name"][:limit] if "view_name" in payload.files else np.asarray([f"sample_{idx:03d}" for idx in range(limit)])

    def __len__(self) -> int:
        return int(self.rgb.shape[0])

    @staticmethod
    def _to_chw_float(arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor.float()

    @staticmethod
    def _to_mask(arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr.astype(np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return (tensor > 0.5).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "rgb": self._to_chw_float(self.rgb[index]) / 255.0,
            "human_mask": self._to_mask(self.human_mask[index]),
            "coarse_prior_normal": self._to_chw_float(self.coarse_prior_normal[index]),
            "teacher_normal": self._to_chw_float(self.teacher_normal[index]),
            "teacher_mask": self._to_mask(self.teacher_mask[index]),
            "hairline_mask": self._to_mask(self.hairline_mask[index]),
            "ear_band_mask": self._to_mask(self.ear_band_mask[index]),
            "view_name": str(self.view_name[index]),
        }


def _move_batch_to_device(batch: dict[str, torch.Tensor | str], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        if key == "view_name":
            if isinstance(value, (list, tuple)):
                moved[key] = [str(item) for item in value]
            else:
                moved[key] = [str(value)]
            continue
        assert isinstance(value, torch.Tensor)
        moved[key] = value.to(device)
    return moved


def _save_visuals(
    *,
    output_dir: Path,
    dataset: RefinerNpzDataset,
    predictions_list: list[dict[str, np.ndarray | str]],
) -> None:
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(predictions_list):
        rgb_np = record["rgb"]
        coarse_np = record["coarse_prior_normal"]
        refined_np = record["refined_normal"]
        teacher_np = record["teacher_normal"]
        human_mask_np = record["human_mask"]
        teacher_mask_np = record["teacher_mask"]
        diff = np.abs(refined_np - coarse_np).mean(axis=-1)
        diff = np.clip(diff / max(1e-6, float(diff.max())), 0.0, 1.0)
        diff_img = np.stack(
            [
                (diff * 255.0).astype(np.uint8),
                np.zeros_like(diff, dtype=np.uint8),
                ((1.0 - diff) * 255.0).astype(np.uint8),
            ],
            axis=-1,
        )
        coarse_img = normal_to_rgb(coarse_np, human_mask_np)
        refined_img = normal_to_rgb(refined_np, human_mask_np)
        teacher_img = normal_to_rgb(teacher_np, teacher_mask_np)
        summary = np.concatenate([rgb_np, coarse_img, refined_img, diff_img, teacher_img], axis=1)

        stem = f"{idx:02d}_{record['view_name']}"
        Image.fromarray(rgb_np).save(visual_dir / f"{stem}_rgb.png")
        Image.fromarray(coarse_img).save(visual_dir / f"{stem}_coarse_prior_normal.png")
        Image.fromarray(refined_img).save(visual_dir / f"{stem}_refined_normal.png")
        Image.fromarray(diff_img).save(visual_dir / f"{stem}_coarse_vs_refined_diff.png")
        Image.fromarray(teacher_img).save(visual_dir / f"{stem}_teacher_normal.png")
        Image.fromarray(summary).save(visual_dir / f"{stem}_summary_strip.png")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    checkpoint = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = DetailNormalRefiner(
        base_dim=int(ckpt_args.get("base_dim", 32)),
        residual_scale=float(ckpt_args.get("residual_scale", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = RefinerNpzDataset(args.dataset_npz, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    losses: list[dict[str, float]] = []
    visual_records: list[dict[str, np.ndarray | str]] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            predictions = model(
                rgb=moved["rgb"],
                coarse_normal=moved["coarse_prior_normal"],
                human_mask=moved["human_mask"],
            )
            loss_dict = compute_detail_normal_refiner_loss(predictions, moved)
            losses.append({key: float(value.item()) for key, value in loss_dict.items()})

            rgb_np = (moved["rgb"].detach().cpu().permute(0, 2, 3, 1).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            coarse_np = predictions["coarse_normal"].detach().cpu().permute(0, 2, 3, 1).numpy()
            refined_np = predictions["refined_normal"].detach().cpu().permute(0, 2, 3, 1).numpy()
            teacher_np = moved["teacher_normal"].detach().cpu().permute(0, 2, 3, 1).numpy()
            human_mask_np = moved["human_mask"].detach().cpu().numpy()[:, 0] > 0.5
            teacher_mask_np = moved["teacher_mask"].detach().cpu().numpy()[:, 0] > 0.5
            view_names = moved["view_name"]

            for idx in range(rgb_np.shape[0]):
                visual_records.append(
                    {
                        "view_name": str(view_names[idx]),
                        "rgb": rgb_np[idx],
                        "coarse_prior_normal": coarse_np[idx],
                        "refined_normal": refined_np[idx],
                        "teacher_normal": teacher_np[idx],
                        "human_mask": human_mask_np[idx],
                        "teacher_mask": teacher_mask_np[idx],
                    }
                )

    _save_visuals(
        output_dir=output_dir,
        dataset=dataset,
        predictions_list=visual_records[: max(1, min(args.visualize_count, len(visual_records)))],
    )

    keys = sorted(losses[0].keys()) if losses else []
    mean_metrics = {key: float(np.mean([item[key] for item in losses])) for key in keys}
    summary = {
        "message": "detail_normal_refiner checkpoint applied",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "dataset_npz": dataset.dataset_npz,
        "device": str(device),
        "num_samples": len(dataset),
        "mean_metrics": mean_metrics,
        "notes": [
            "Visual exports follow the fixed protocol: RGB, coarse prior normal, refined normal, diff, teacher.",
            "This tool evaluates a trained detail_normal_refiner on any ROI dataset export.",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
