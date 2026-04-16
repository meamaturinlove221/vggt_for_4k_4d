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
            "depths": targets["depths"].astype(np.float32),
            "extrinsics": targets["extrinsics"].astype(np.float32),
            "intrinsics": targets["intrinsics"].astype(np.float32),
            "cam_points": targets["cam_points"].astype(np.float32),
            "world_points": targets["world_points"].astype(np.float32),
        }

        if "prior_maps" in inputs.files:
            case["prior_maps"] = inputs["prior_maps"].astype(np.float32)
        if "prior_mask" in inputs.files:
            case["prior_mask"] = inputs["prior_mask"].astype(bool)
        if "prior_depths" in targets.files:
            case["prior_depths"] = targets["prior_depths"].astype(np.float32)
        if "prior_points" in targets.files:
            case["prior_points"] = targets["prior_points"].astype(np.float32)

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

        if "prior_maps" in case:
            batch["prior_maps"] = [case["prior_maps"][idx] for idx in ids]
        if "prior_mask" in case:
            batch["prior_mask"] = [case["prior_mask"][idx] for idx in ids]
        if "prior_depths" in case:
            batch["prior_depths"] = [case["prior_depths"][idx] for idx in ids]
        if "prior_points" in case:
            batch["prior_points"] = [case["prior_points"][idx] for idx in ids]

        return batch
