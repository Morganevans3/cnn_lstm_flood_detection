"""Train/val from raw 5° MODIS + S1 tiles: one center crop per tile-date (no patch shards)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Set

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pipeline.tile_5d_loader import Tile5dLoadContext, center_crop_or_pad, discover_tile_paths


class Tile5dCenterCropDataset(Dataset):
    """
    One sample per (year, period_idx, tile_id): full tile loaded, then center-cropped to img_hw.

    Expects keys where p_idx >= sequence_length - 1 (same as processing notebook).
    """

    def __init__(
        self,
        keys: Sequence[Tuple[int, int, int]],
        ctx: Tile5dLoadContext,
        img_hw: int,
        sequence_length: int,
    ):
        self.keys = list(keys)
        self.ctx = ctx
        self.img_hw = int(img_hw)
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        year, p_idx, tile_id = self.keys[i]
        out = self.ctx.load_sample(year, p_idx, tile_id)
        if out is None:
            # Rare: return zeros if load failed mid-epoch (should filter keys in setup)
            z = torch.zeros(self.sequence_length, 8, self.img_hw, self.img_hw, dtype=torch.float32)
            y = torch.zeros(self.img_hw, self.img_hw, dtype=torch.float32)
            return z, y
        x, y = out
        x, y = center_crop_or_pad(x, y, self.img_hw)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class Tile5dDataModule(pl.LightningDataModule):
    """
    Lightweight training: scans ``modis_dir`` / ``s1_dir`` for GeoTIFF + .npy pairs, no ``patch_shard_*.npz``.

    Set ``training.data_mode: tiles`` in config and point ``paths.modis_5d`` (or legacy ``modis_8day_5d``) / ``paths.s1_labels_5d``.
    """

    def __init__(
        self,
        modis_dir: Path,
        s1_dir: Path,
        img_hw: int,
        sequence_length: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_split: float = 0.2,
        random_state: int = 42,
        max_tile_keys: Optional[int] = None,
        s1_percentile: float = 20.0,
        loyo_val_year: Optional[int] = None,
        loyo_years_filter: Optional[Sequence[int]] = None,
        modis_cache_max: int = 48,
        s1_cache_max: int = 96,
    ):
        super().__init__()
        self.modis_dir = Path(modis_dir)
        self.s1_dir = Path(s1_dir)
        self.img_hw = int(img_hw)
        self.sequence_length = int(sequence_length)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.random_state = random_state
        self.max_tile_keys = max_tile_keys
        self.s1_percentile = s1_percentile
        self.loyo_val_year = loyo_val_year
        self.loyo_years_filter: Optional[Set[int]] = (
            set(int(y) for y in loyo_years_filter) if loyo_years_filter is not None else None
        )
        self.modis_cache_max = int(modis_cache_max)
        self.s1_cache_max = int(s1_cache_max)
        self._ctx: Optional[Tile5dLoadContext] = None
        self.train_ds: Optional[Tile5dCenterCropDataset] = None
        self.val_ds: Optional[Tile5dCenterCropDataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage not in (None, "fit"):
            return
        if self.train_ds is not None:
            return

        modis_files, s1_files, common = discover_tile_paths(self.modis_dir, self.s1_dir)
        keys = sorted(k for k in common if k[1] >= self.sequence_length - 1)
        if self.loyo_years_filter is not None:
            keys = [k for k in keys if k[0] in self.loyo_years_filter]
        if self.max_tile_keys is not None:
            keys = keys[: int(self.max_tile_keys)]

        split_note = (
            f"LOYO val_year={self.loyo_val_year}"
            if self.loyo_val_year is not None
            else f"random val_split={self.val_split}"
        )
        print(
            f"Tile5d: MODIS={len(modis_files)} S1={len(s1_files)} common={len(common)} "
            f"trainable_keys={len(keys)} (one crop/tile-date, img_hw={self.img_hw}; {split_note})",
            flush=True,
        )
        if not keys:
            raise FileNotFoundError(
                f"No overlapping MODIS/S1 keys in {self.modis_dir} / {self.s1_dir} "
                f"(need p_idx >= {self.sequence_length - 1})."
            )

        self._ctx = Tile5dLoadContext(
            modis_files,
            s1_files,
            sequence_length=self.sequence_length,
            s1_percentile=self.s1_percentile,
            modis_cache_max=self.modis_cache_max,
            s1_cache_max=self.s1_cache_max,
        )

        t0 = time.perf_counter()
        if self.loyo_val_year is not None:
            vy = int(self.loyo_val_year)
            train_k = [k for k in keys if k[0] != vy]
            val_k = [k for k in keys if k[0] == vy]
            if not val_k:
                raise ValueError(
                    f"LOYO: no samples for validation year {vy}. "
                    "Check modis/S1 overlap and loyo_cv.train_years in config."
                )
            if not train_k:
                raise ValueError(f"LOYO: no training samples when holding out year {vy}.")
        else:
            train_k, val_k = train_test_split(
                keys,
                test_size=self.val_split,
                random_state=self.random_state,
                shuffle=True,
            )
        print(f"Train/val split in {time.perf_counter() - t0:.2f}s", flush=True)

        self.train_ds = Tile5dCenterCropDataset(train_k, self._ctx, self.img_hw, self.sequence_length)
        self.val_ds = Tile5dCenterCropDataset(val_k, self._ctx, self.img_hw, self.sequence_length)
        print(f"Tile5d: train={len(self.train_ds)} val={len(self.val_ds)}", flush=True)

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
