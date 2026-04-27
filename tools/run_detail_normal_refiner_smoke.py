from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.detail_normal_refiner_loss import compute_detail_normal_refiner_loss  # noqa: E402
from vggt.models.detail_normal_refiner import DetailNormalRefiner  # noqa: E402
from vggt.utils.normal_refiner import normal_to_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train the detail normal refiner on one ROI pack.")
    parser.add_argument("--dataset-npz", required=True, help="ROI dataset exported by export_detail_normal_refiner_dataset.py")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=80, help="Small overfit step count")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=8, help="Number of ROI samples to use for the smoke subset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, _ = torch.cuda.get_device_capability()
        if major > 9:
            print("CUDA device detected but current PyTorch build does not support this capability; falling back to CPU.")
            return torch.device("cpu")
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        print(f"CUDA probe failed ({exc}); falling back to CPU.")
        return torch.device("cpu")


def _load_batch(payload: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    def _tensor(key: str, *, mask: bool = False) -> torch.Tensor:
        arr = payload[key][indices]
        tensor = torch.from_numpy(arr).to(device)
        if tensor.ndim == 4 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.float()
        if mask:
            return (tensor > 0.5).float()
        return tensor

    return {
        "rgb": _tensor("rgb") / 255.0,
        "human_mask": _tensor("human_mask", mask=True),
        "coarse_prior_normal": _tensor("coarse_prior_normal"),
        "teacher_normal": _tensor("teacher_normal"),
        "teacher_mask": _tensor("teacher_mask", mask=True),
        "hairline_mask": _tensor("hairline_mask", mask=True),
        "ear_band_mask": _tensor("ear_band_mask", mask=True),
    }


def _save_visuals(
    *,
    output_dir: Path,
    rgb: torch.Tensor,
    coarse: torch.Tensor,
    refined: torch.Tensor,
    teacher: torch.Tensor,
    human_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> None:
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    rgb_np = (rgb.permute(0, 2, 3, 1).cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    coarse_np = coarse.permute(0, 2, 3, 1).cpu().numpy()
    refined_np = refined.permute(0, 2, 3, 1).cpu().numpy()
    teacher_np = teacher.permute(0, 2, 3, 1).cpu().numpy()
    human_mask_np = human_mask[:, 0].cpu().numpy() > 0.5
    teacher_mask_np = teacher_mask[:, 0].cpu().numpy() > 0.5

    count = rgb_np.shape[0]
    for idx in range(count):
        coarse_img = normal_to_rgb(coarse_np[idx], human_mask_np[idx])
        refined_img = normal_to_rgb(refined_np[idx], human_mask_np[idx])
        teacher_img = normal_to_rgb(teacher_np[idx], teacher_mask_np[idx])
        diff = np.abs(refined_np[idx] - coarse_np[idx]).mean(axis=-1)
        diff = np.clip(diff / max(1e-6, float(diff.max())), 0.0, 1.0)
        diff_img = np.stack(
            [
                (diff * 255.0).astype(np.uint8),
                np.zeros_like(diff, dtype=np.uint8),
                ((1.0 - diff) * 255.0).astype(np.uint8),
            ],
            axis=-1,
        )

        Image.fromarray(rgb_np[idx]).save(visual_dir / f"{idx:02d}_rgb.png")
        Image.fromarray(coarse_img).save(visual_dir / f"{idx:02d}_coarse_prior_normal.png")
        Image.fromarray(refined_img).save(visual_dir / f"{idx:02d}_refined_normal.png")
        Image.fromarray(teacher_img).save(visual_dir / f"{idx:02d}_teacher_normal.png")
        Image.fromarray(diff_img).save(visual_dir / f"{idx:02d}_coarse_vs_refined_diff.png")
        summary = np.concatenate([rgb_np[idx], coarse_img, refined_img, diff_img, teacher_img], axis=1)
        Image.fromarray(summary).save(visual_dir / f"{idx:02d}_summary_strip.png")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.dataset_npz, allow_pickle=False)
    total_samples = int(payload["rgb"].shape[0])
    if total_samples <= 0:
        raise RuntimeError("Dataset is empty.")
    sample_count = min(int(args.sample_count), total_samples)
    subset_indices = np.arange(total_samples, dtype=np.int64)[:sample_count]

    device = _resolve_device(args.device)
    model = DetailNormalRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    last_batch = None
    last_output = None

    for step in range(args.steps):
        batch_indices = rng.choice(subset_indices, size=min(args.batch_size, sample_count), replace=True)
        batch = _load_batch(payload, batch_indices, device)
        predictions = model(
            rgb=batch["rgb"],
            coarse_normal=batch["coarse_prior_normal"],
            human_mask=batch["human_mask"],
        )
        loss_dict = compute_detail_normal_refiner_loss(predictions, batch)
        loss = loss_dict["loss_detail_normal_total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        record = {"step": float(step), "loss": float(loss.item())}
        for key, value in loss_dict.items():
            record[key] = float(value.item())
        history.append(record)
        last_batch = batch
        last_output = predictions

    if last_batch is None or last_output is None:
        raise RuntimeError("Smoke run produced no batch.")

    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "dataset_npz": str(Path(args.dataset_npz).expanduser().resolve()),
        },
        output_dir / "detail_normal_refiner_smoke.pt",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        rgb=(last_batch["rgb"].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8),
        coarse_prior_normal=last_output["coarse_normal"].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32),
        refined_normal=last_output["refined_normal"].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32),
        normal_residual=last_output["normal_residual"].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32),
        teacher_normal=last_batch["teacher_normal"].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32),
        human_mask=last_batch["human_mask"].detach().cpu().numpy().astype(bool),
        teacher_mask=last_batch["teacher_mask"].detach().cpu().numpy().astype(bool),
    )

    metrics = {
        "dataset_npz": str(Path(args.dataset_npz).expanduser().resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "steps": int(args.steps),
        "sample_count": int(sample_count),
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "best_loss": min(item["loss"] for item in history),
        "final_detail_cosine": history[-1]["loss_detail_normal_cosine"],
        "final_detail_edge": history[-1]["loss_detail_normal_edge"],
        "final_detail_mask_restricted": history[-1]["loss_detail_normal_mask_restricted"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "loss_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    _save_visuals(
        output_dir=output_dir,
        rgb=last_batch["rgb"].detach(),
        coarse=last_output["coarse_normal"].detach(),
        refined=last_output["refined_normal"].detach(),
        teacher=last_batch["teacher_normal"].detach(),
        human_mask=last_batch["human_mask"].detach(),
        teacher_mask=last_batch["teacher_mask"].detach(),
    )

    summary = {
        "message": "detail_normal_refiner smoke run completed",
        "metrics": metrics,
        "notes": [
            "This is a small ROI-first overfit check, not the final long-run refined normal result.",
            "Teacher normal comes from 60v pseudo geometry, not from coarse prior self-distillation.",
            "The saved visuals follow the fixed protocol: RGB, coarse prior normal, refined normal, diff, teacher.",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
