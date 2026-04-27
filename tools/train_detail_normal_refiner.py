from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
    parser = argparse.ArgumentParser(description="Train the ROI-first detail normal refiner.")
    parser.add_argument("--train-dataset-npz", required=True, help="Training dataset exported by export_detail_normal_refiner_dataset.py")
    parser.add_argument("--val-dataset-npz", default="", help="Optional validation dataset; defaults to train dataset when omitted")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap for fast experiments")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional cap for fast experiments")
    parser.add_argument("--visualize-count", type=int, default=8)
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


@dataclass
class EpochMetrics:
    loss_detail_normal_total: float = 0.0
    loss_detail_normal_cosine: float = 0.0
    loss_detail_normal_edge: float = 0.0
    loss_detail_normal_mask_restricted: float = 0.0
    metric_boundary_weight_mean: float = 0.0
    metric_hairline_cosine: float = 0.0
    metric_ear_band_cosine: float = 0.0
    num_batches: int = 0

    def update(self, metrics: dict[str, torch.Tensor]) -> None:
        for key in metrics:
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + float(metrics[key].item()))
        self.num_batches += 1

    def mean(self) -> dict[str, float]:
        denom = max(1, self.num_batches)
        return {
            "loss_detail_normal_total": self.loss_detail_normal_total / denom,
            "loss_detail_normal_cosine": self.loss_detail_normal_cosine / denom,
            "loss_detail_normal_edge": self.loss_detail_normal_edge / denom,
            "loss_detail_normal_mask_restricted": self.loss_detail_normal_mask_restricted / denom,
            "metric_boundary_weight_mean": self.metric_boundary_weight_mean / denom,
            "metric_hairline_cosine": self.metric_hairline_cosine / denom,
            "metric_ear_band_cosine": self.metric_ear_band_cosine / denom,
            "num_batches": float(self.num_batches),
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


def _make_loader(dataset: RefinerNpzDataset, *, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _save_visuals(
    *,
    output_dir: Path,
    model: DetailNormalRefiner,
    dataset: RefinerNpzDataset,
    device: torch.device,
    count: int,
) -> list[dict[str, str]]:
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    model.eval()
    limit = min(max(1, int(count)), len(dataset))
    with torch.no_grad():
        for idx in range(limit):
            sample = dataset[idx]
            batch = {
                key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
                if key != "view_name"
            }
            predictions = model(
                rgb=batch["rgb"],
                coarse_normal=batch["coarse_prior_normal"],
                human_mask=batch["human_mask"],
            )

            rgb_np = (batch["rgb"][0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            coarse_np = predictions["coarse_normal"][0].detach().cpu().permute(1, 2, 0).numpy()
            refined_np = predictions["refined_normal"][0].detach().cpu().permute(1, 2, 0).numpy()
            teacher_np = batch["teacher_normal"][0].detach().cpu().permute(1, 2, 0).numpy()
            human_mask_np = batch["human_mask"][0, 0].detach().cpu().numpy() > 0.5
            teacher_mask_np = batch["teacher_mask"][0, 0].detach().cpu().numpy() > 0.5
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

            stem = f"{idx:02d}_{str(sample['view_name'])}"
            Image.fromarray(rgb_np).save(visual_dir / f"{stem}_rgb.png")
            Image.fromarray(coarse_img).save(visual_dir / f"{stem}_coarse_prior_normal.png")
            Image.fromarray(refined_img).save(visual_dir / f"{stem}_refined_normal.png")
            Image.fromarray(diff_img).save(visual_dir / f"{stem}_coarse_vs_refined_diff.png")
            Image.fromarray(teacher_img).save(visual_dir / f"{stem}_teacher_normal.png")
            Image.fromarray(summary).save(visual_dir / f"{stem}_summary_strip.png")
            records.append(
                {
                    "view_name": str(sample["view_name"]),
                    "summary_strip": str((visual_dir / f"{stem}_summary_strip.png").resolve()),
                }
            )
    return records


def _train_one_epoch(
    model: DetailNormalRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    tracker = EpochMetrics()
    for batch in loader:
        moved = _move_batch_to_device(batch, device)
        predictions = model(
            rgb=moved["rgb"],
            coarse_normal=moved["coarse_prior_normal"],
            human_mask=moved["human_mask"],
        )
        loss_dict = compute_detail_normal_refiner_loss(predictions, moved)
        loss = loss_dict["loss_detail_normal_total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tracker.update(loss_dict)
    return tracker.mean()


def _eval_one_epoch(
    model: DetailNormalRefiner,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tracker = EpochMetrics()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            predictions = model(
                rgb=moved["rgb"],
                coarse_normal=moved["coarse_prior_normal"],
                human_mask=moved["human_mask"],
            )
            loss_dict = compute_detail_normal_refiner_loss(predictions, moved)
            tracker.update(loss_dict)
    return tracker.mean()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RefinerNpzDataset(args.train_dataset_npz, max_samples=args.max_train_samples)
    val_dataset = RefinerNpzDataset(args.val_dataset_npz or args.train_dataset_npz, max_samples=args.max_val_samples)
    device = _resolve_device(args.device)

    model = DetailNormalRefiner(base_dim=args.base_dim, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = _make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)

    history: list[dict[str, float | int]] = []
    best_epoch = -1
    best_val = float("inf")
    best_checkpoint_path = output_dir / "best_model.pt"
    latest_checkpoint_path = output_dir / "latest_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = _eval_one_epoch(model, val_loader, device)
        record: dict[str, float | int] = {"epoch": epoch}
        record.update({f"train/{key}": value for key, value in train_metrics.items()})
        record.update({f"val/{key}": value for key, value in val_metrics.items()})
        history.append(record)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "train_dataset_npz": train_dataset.dataset_npz,
            "val_dataset_npz": val_dataset.dataset_npz,
            "history": history,
        }
        torch.save(checkpoint_payload, latest_checkpoint_path)
        if val_metrics["loss_detail_normal_total"] <= best_val:
            best_val = float(val_metrics["loss_detail_normal_total"])
            best_epoch = epoch
            torch.save(checkpoint_payload, best_checkpoint_path)

    best_payload = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_payload["model_state"])
    best_train_metrics = _eval_one_epoch(model, train_loader, device)
    best_val_metrics = _eval_one_epoch(model, val_loader, device)

    train_visuals = _save_visuals(
        output_dir=output_dir / "best_train",
        model=model,
        dataset=train_dataset,
        device=device,
        count=args.visualize_count,
    )
    val_visuals = _save_visuals(
        output_dir=output_dir / "best_val",
        model=model,
        dataset=val_dataset,
        device=device,
        count=args.visualize_count,
    )

    summary = {
        "message": "detail_normal_refiner training completed",
        "device": str(device),
        "train_dataset_npz": train_dataset.dataset_npz,
        "val_dataset_npz": val_dataset.dataset_npz,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "notes": [
            "This is the formal ROI-first detail normal refiner training path.",
            "Teacher normal must come from finer geometry than the coarse prior itself.",
            "Visual exports follow the fixed protocol: RGB, coarse prior normal, refined normal, coarse-vs-refined diff, teacher.",
        ],
        "train_visuals": train_visuals,
        "val_visuals": val_visuals,
    }

    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "best_metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "best_train_metrics": best_train_metrics,
                "best_val_metrics": best_val_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
