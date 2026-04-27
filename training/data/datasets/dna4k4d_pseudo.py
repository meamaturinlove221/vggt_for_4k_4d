# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from data.base_dataset import BaseDataset


class DNA4K4DPseudoDataset(BaseDataset):
    """
    Fine-tuning dataset for 4K4D cases prepared from:
    - exported scene images/masks
    - VGGT pseudo labels (predictions.npz-derived targets)
    - projected human prior maps

    Each case root is expected to contain:
    - case_manifest.json
    - inputs.npz
    - targets.npz
    """

    def __init__(
        self,
        common_conf,
        case_roots,
        split: str = "train",
        len_train: int = 100000,
        len_test: int = 1000,
        keep_target_first: bool = True,
    ):
        super().__init__(common_conf=common_conf)

        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.keep_target_first = keep_target_first
        self.prior_pose_noise = self._resolve_prior_pose_noise(common_conf)

        if not case_roots:
            raise ValueError("DNA4K4DPseudoDataset requires at least one case root.")

        self.case_specs = []
        self.case_by_seq_name = {}
        self._case_cache = {}

        for raw_root in case_roots:
            case_root = Path(raw_root).expanduser().resolve()
            manifest_path = case_root / "case_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"case_manifest.json not found under {case_root}")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seq_name = manifest.get("case_id", case_root.name)
            input_npz = case_root / manifest.get("inputs_npz", "inputs.npz")
            target_npz = case_root / manifest.get("targets_npz", "targets.npz")

            if not input_npz.is_file():
                raise FileNotFoundError(f"inputs.npz not found: {input_npz}")
            if not target_npz.is_file():
                raise FileNotFoundError(f"targets.npz not found: {target_npz}")

            spec = {
                "seq_name": seq_name,
                "case_root": case_root,
                "manifest_path": manifest_path,
                "input_npz": input_npz,
                "target_npz": target_npz,
                "manifest": manifest,
                "num_views": int(manifest["num_views"]),
                "target_view_index": int(manifest.get("target_view_index", 0)),
            }
            self.case_specs.append(spec)
            self.case_by_seq_name[seq_name] = spec

        self.sequence_list = [spec["seq_name"] for spec in self.case_specs]
        self.sequence_list_len = len(self.sequence_list)
        self.len_train = len_train if split == "train" else len_test

        logging.info(f"DNA4K4DPseudoDataset cases: {self.sequence_list_len}")
        for spec in self.case_specs:
            logging.info(
                "  case=%s views=%d root=%s",
                spec["seq_name"],
                spec["num_views"],
                spec["case_root"],
            )

    @staticmethod
    def _cfg_get(config, key, default=None):
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _resolve_prior_pose_noise(self, common_conf) -> dict:
        cfg = self._cfg_get(common_conf, "prior_pose_noise", None)
        enabled = bool(self._cfg_get(cfg, "enabled", False))
        return {
            "enabled": enabled,
            "prob": float(self._cfg_get(cfg, "prob", 0.5)),
            "cam_xyz_std": float(self._cfg_get(cfg, "cam_xyz_std", 0.01)),
            "cam_xyz_global_std": float(self._cfg_get(cfg, "cam_xyz_global_std", 0.005)),
            "scale_std": float(self._cfg_get(cfg, "scale_std", 0.01)),
            "normal_std": float(self._cfg_get(cfg, "normal_std", 0.03)),
            "visibility_dropout": float(self._cfg_get(cfg, "visibility_dropout", 0.05)),
            "spatial_shift_std_px": float(self._cfg_get(cfg, "spatial_shift_std_px", 1.0)),
            "max_spatial_shift_px": int(self._cfg_get(cfg, "max_spatial_shift_px", 3)),
        }

    def _apply_prior_pose_noise(
        self,
        prior_maps: np.ndarray | None,
        prior_summary_tokens: np.ndarray | None,
        prior_input_meta: dict | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        cfg = self.prior_pose_noise
        if not self.training or not cfg["enabled"]:
            return prior_maps, prior_summary_tokens
        if np.random.rand() > cfg["prob"]:
            return prior_maps, prior_summary_tokens

        channel_groups = (prior_input_meta or {}).get("channel_groups", {})
        dense_groups = channel_groups.get("dense", {})
        summary_groups = channel_groups.get("summary", {})

        maps = None if prior_maps is None else np.asarray(prior_maps, dtype=np.float32).copy()
        summaries = None if prior_summary_tokens is None else np.asarray(prior_summary_tokens, dtype=np.float32).copy()

        dense_spatial_idx = list(dense_groups.get("smplx_spatial", []))
        dense_posed_idx = list(dense_groups.get("smplx_posed_cam", []))
        dense_normal_idx = list(dense_groups.get("smplx_cam_normals", []))
        dense_visibility_idx = list(dense_groups.get("smplx_visibility", []))
        summary_posed_idx = list(summary_groups.get("smplx_posed_cam", []))
        summary_normal_idx = list(summary_groups.get("smplx_cam_normals", []))

        if maps is not None and dense_spatial_idx:
            max_shift = max(0, int(cfg["max_spatial_shift_px"]))
            shift_std = max(0.0, float(cfg["spatial_shift_std_px"]))
            if max_shift > 0 and shift_std > 0:
                for view_idx in range(maps.shape[0]):
                    shift_x = int(np.clip(np.round(np.random.normal(0.0, shift_std)), -max_shift, max_shift))
                    shift_y = int(np.clip(np.round(np.random.normal(0.0, shift_std)), -max_shift, max_shift))
                    if shift_x != 0 or shift_y != 0:
                        maps[view_idx, dense_spatial_idx] = np.roll(
                            maps[view_idx, dense_spatial_idx],
                            shift=(shift_y, shift_x),
                            axis=(1, 2),
                        )

        global_offset = np.random.normal(0.0, cfg["cam_xyz_global_std"], size=(3,)).astype(np.float32)
        view_offsets = None
        scale = np.float32(1.0 + np.random.normal(0.0, cfg["scale_std"]))

        if maps is not None and len(dense_posed_idx) == 3:
            view_offsets = np.random.normal(
                0.0,
                cfg["cam_xyz_std"],
                size=(maps.shape[0], 3),
            ).astype(np.float32)
            maps[:, dense_posed_idx] = (
                maps[:, dense_posed_idx] * scale
                + global_offset[None, :, None, None]
                + view_offsets[:, :, None, None]
            ).astype(np.float32)

        if summaries is not None and len(summary_posed_idx) == 3:
            if view_offsets is None:
                view_offsets = np.random.normal(
                    0.0,
                    cfg["cam_xyz_std"],
                    size=(summaries.shape[0], 3),
                ).astype(np.float32)
            summaries[:, :, summary_posed_idx] = (
                summaries[:, :, summary_posed_idx] * scale
                + global_offset[None, None, :]
                + view_offsets[:, None, :]
            ).astype(np.float32)

        if maps is not None and len(dense_normal_idx) == 3 and cfg["normal_std"] > 0:
            maps[:, dense_normal_idx] += np.random.normal(
                0.0,
                cfg["normal_std"],
                size=maps[:, dense_normal_idx].shape,
            ).astype(np.float32)
            norms = np.linalg.norm(maps[:, dense_normal_idx], axis=1, keepdims=True)
            maps[:, dense_normal_idx] /= np.clip(norms, 1e-6, None)

        if summaries is not None and len(summary_normal_idx) == 3 and cfg["normal_std"] > 0:
            summaries[:, :, summary_normal_idx] += np.random.normal(
                0.0,
                cfg["normal_std"],
                size=summaries[:, :, summary_normal_idx].shape,
            ).astype(np.float32)
            norms = np.linalg.norm(summaries[:, :, summary_normal_idx], axis=2, keepdims=True)
            summaries[:, :, summary_normal_idx] /= np.clip(norms, 1e-6, None)

        if maps is not None and dense_visibility_idx and cfg["visibility_dropout"] > 0:
            keep_prob = float(np.clip(1.0 - cfg["visibility_dropout"], 0.0, 1.0))
            keep_mask = (
                np.random.rand(maps.shape[0], maps.shape[2], maps.shape[3]) < keep_prob
            ).astype(np.float32)
            maps[:, dense_visibility_idx] *= keep_mask[:, None, :, :]
            maps[:, dense_visibility_idx] = np.clip(maps[:, dense_visibility_idx], 0.0, 1.0)

        return maps, summaries

    def _load_case_arrays(self, seq_name: str) -> dict:
        if seq_name in self._case_cache:
            return self._case_cache[seq_name]

        spec = self.case_by_seq_name[seq_name]
        inputs = np.load(spec["input_npz"], allow_pickle=False)
        targets = np.load(spec["target_npz"], allow_pickle=False)

        case = {
            "images": inputs["images"],
            "point_masks": inputs["point_masks"].astype(bool),
            "camera_ids": inputs["camera_ids"],
            "view_roles": inputs["view_roles"],
            "prior_input_meta": spec["manifest"].get("prior_input_meta", {}),
            "depths": targets["depths"].astype(np.float32),
            "extrinsics": targets["extrinsics"].astype(np.float32),
            "intrinsics": targets["intrinsics"].astype(np.float32),
            "cam_points": targets["cam_points"].astype(np.float32),
            "world_points": targets["world_points"].astype(np.float32),
        }

        if "prior_maps" in inputs.files:
            case["prior_maps"] = inputs["prior_maps"].astype(np.float32)
        if "prior_summary_tokens" in inputs.files:
            case["prior_summary_tokens"] = inputs["prior_summary_tokens"].astype(np.float32)
        if "prior_mask" in inputs.files:
            case["prior_mask"] = inputs["prior_mask"].astype(bool)
        if "prior_depths" in targets.files:
            case["prior_depths"] = targets["prior_depths"].astype(np.float32)
        if "prior_points" in targets.files:
            case["prior_points"] = targets["prior_points"].astype(np.float32)
        if "prior_normals" in targets.files:
            case["prior_normals"] = targets["prior_normals"].astype(np.float32)
        if "teacher_normals" in targets.files:
            case["teacher_normals"] = targets["teacher_normals"].astype(np.float32)
        if "teacher_mask" in targets.files:
            case["teacher_mask"] = targets["teacher_mask"].astype(bool)
        if "head_roi_mask" in targets.files:
            case["head_roi_mask"] = targets["head_roi_mask"].astype(bool)
        if "face_roi_mask" in targets.files:
            case["face_roi_mask"] = targets["face_roi_mask"].astype(bool)
        if "hairline_mask" in targets.files:
            case["hairline_mask"] = targets["hairline_mask"].astype(bool)
        if "ear_band_mask" in targets.files:
            case["ear_band_mask"] = targets["ear_band_mask"].astype(bool)

        self._case_cache[seq_name] = case
        return case

    def _sample_ids(self, num_views: int, img_per_seq: int) -> np.ndarray:
        if img_per_seq is None or img_per_seq <= 0:
            img_per_seq = num_views

        img_per_seq = int(img_per_seq)
        if self.keep_target_first:
            target_idx = 0
            available = np.arange(1, num_views, dtype=np.int64)
            if img_per_seq == 1:
                return np.array([target_idx], dtype=np.int64)

            sample_size = img_per_seq - 1
            if len(available) == 0:
                sampled = np.full((sample_size,), target_idx, dtype=np.int64)
            elif sample_size <= len(available):
                sampled = np.random.choice(available, size=sample_size, replace=False)
            else:
                unique = available
                extra = np.random.choice(available, size=sample_size - len(available), replace=True)
                sampled = np.concatenate([unique, extra], axis=0)
            sampled = np.asarray(sampled, dtype=np.int64)
            return np.concatenate([np.array([target_idx], dtype=np.int64), sampled], axis=0)

        indices = np.arange(num_views, dtype=np.int64)
        if img_per_seq <= num_views:
            return np.sort(np.random.choice(indices, size=img_per_seq, replace=False))
        if num_views == 0:
            raise ValueError("Cannot sample from an empty 4K4D pseudo case.")
        extra = np.random.choice(indices, size=img_per_seq - num_views, replace=True).astype(np.int64)
        return np.concatenate([indices, extra], axis=0)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        del aspect_ratio

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            if self.sequence_list_len <= 0:
                raise ValueError("DNA4K4DPseudoDataset has no available cases.")
            seq_index = 0 if seq_index is None else int(seq_index) % self.sequence_list_len
            seq_name = self.sequence_list[seq_index]

        case = self._load_case_arrays(seq_name)
        num_views = case["images"].shape[0]

        if ids is None:
            ids = self._sample_ids(num_views, img_per_seq)
        ids = np.asarray(ids, dtype=np.int64)

        batch = {
            "seq_name": f"dna4k4d_{seq_name}",
            "ids": ids,
            "frame_num": len(ids),
            "images": [case["images"][idx] for idx in ids],
            "depths": [case["depths"][idx] for idx in ids],
            "extrinsics": [case["extrinsics"][idx] for idx in ids],
            "intrinsics": [case["intrinsics"][idx] for idx in ids],
            "cam_points": [case["cam_points"][idx] for idx in ids],
            "world_points": [case["world_points"][idx] for idx in ids],
            "point_masks": [case["point_masks"][idx] for idx in ids],
            "original_sizes": [np.array(case["images"][idx].shape[:2]) for idx in ids],
        }

        selected_prior_maps = None
        selected_prior_summary_tokens = None
        if "prior_maps" in case:
            selected_prior_maps = np.asarray([case["prior_maps"][idx] for idx in ids], dtype=np.float32)
        if "prior_summary_tokens" in case:
            selected_prior_summary_tokens = np.asarray(
                [case["prior_summary_tokens"][idx] for idx in ids],
                dtype=np.float32,
            )
        if selected_prior_maps is not None or selected_prior_summary_tokens is not None:
            selected_prior_maps, selected_prior_summary_tokens = self._apply_prior_pose_noise(
                selected_prior_maps,
                selected_prior_summary_tokens,
                case.get("prior_input_meta", {}),
            )
        if selected_prior_maps is not None:
            batch["prior_maps"] = [selected_prior_maps[idx] for idx in range(len(ids))]
        if selected_prior_summary_tokens is not None:
            batch["prior_summary_tokens"] = [
                selected_prior_summary_tokens[idx] for idx in range(len(ids))
            ]
        if "prior_mask" in case:
            batch["prior_mask"] = [case["prior_mask"][idx] for idx in ids]
        if "prior_depths" in case:
            batch["prior_depths"] = [case["prior_depths"][idx] for idx in ids]
        if "prior_points" in case:
            batch["prior_points"] = [case["prior_points"][idx] for idx in ids]
        if "prior_normals" in case:
            batch["prior_normals"] = [case["prior_normals"][idx] for idx in ids]
        if "teacher_normals" in case:
            batch["teacher_normals"] = [case["teacher_normals"][idx] for idx in ids]
        if "teacher_mask" in case:
            batch["teacher_mask"] = [case["teacher_mask"][idx] for idx in ids]
        if "head_roi_mask" in case:
            batch["head_roi_mask"] = [case["head_roi_mask"][idx] for idx in ids]
        if "face_roi_mask" in case:
            batch["face_roi_mask"] = [case["face_roi_mask"][idx] for idx in ids]
        if "hairline_mask" in case:
            batch["hairline_mask"] = [case["hairline_mask"][idx] for idx in ids]
        if "ear_band_mask" in case:
            batch["ear_band_mask"] = [case["ear_band_mask"][idx] for idx in ids]

        return batch
