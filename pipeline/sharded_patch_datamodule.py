"""Train/val from patch_shard_*.npz without loading all patches into RAM."""

from __future__ import annotations

import bisect
import errno
import json
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(it, **kwargs):
        return it


T = TypeVar("T")


def retry_drive_io(fn: Callable[[], T], *, retries: int = 6, base_delay: float = 0.35) -> T:
    """
    Google Drive FUSE on Colab often raises OSError 107 (ENOTCONN) under heavy sequential opens.
    Retry a few times with backoff before giving up.
    """
    last: Optional[BaseException] = None
    for i in range(retries):
        try:
            return fn()
        except OSError as e:
            last = e
            if e.errno not in (errno.ENOTCONN, errno.EIO, errno.ESTALE, errno.EBUSY):
                raise
            time.sleep(base_delay * (i + 1))
    assert last is not None
    raise last


def npz_member_leading_dim(npz_path: Union[str, Path], member: str = "Y") -> int:
    """
    First dimension of an array inside a .npz without loading array data.

    Uses zip + NPY header parsing only (lighter than numpy.load on Drive FUSE; avoids
    NpzFile.close() races that often surface as OSError 107).
    """

    def _read() -> int:
        from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

        p = Path(npz_path)
        with zipfile.ZipFile(p, "r") as zf:
            entry = f"{member}.npy"
            if entry not in zf.namelist():
                raise KeyError(f"{p}: zip has no {entry}")
            with zf.open(entry) as fh:
                version = read_magic(fh)
                if version == (1, 0):
                    shape, _fortran, _dtype = read_array_header_1_0(fh)
                else:
                    shape, _fortran, _dtype = read_array_header_2_0(fh)
        if isinstance(shape, int):
            return int(shape)
        t = tuple(shape)
        return int(t[0]) if t else 0

    return retry_drive_io(_read)


class ShardedNpzPatchDataset(Dataset):
    """
    Random access to patches stored across multiple .npz files (keys X, Y).
    Uses mmap for reads when arrays are uncompressed inside the npz.

    Expects X: (N, T, C, H, W), Y: (N, H, W).

    **max_open_shards** (default 4): Only keep this many shard files mmap-open at
    once (LRU). Opening every shard at once (e.g. 100+ files on Google Drive)
    can exhaust Colab's small **local** disk via FUSE/cache and raise OSError 28.
    """

    def __init__(
        self,
        shard_paths: Sequence[Union[str, Path]],
        indices: Optional[np.ndarray] = None,
        max_open_shards: int = 4,
        shard_lengths: Optional[Sequence[int]] = None,
    ):
        self.paths: List[Path] = [Path(p) for p in shard_paths]
        if not self.paths:
            raise ValueError("No shard paths provided.")

        self.max_open = max(1, int(max_open_shards))
        self._lru: OrderedDict[int, Any] = OrderedDict()

        if shard_lengths is not None:
            if len(shard_lengths) != len(self.paths):
                raise ValueError(
                    f"shard_lengths length {len(shard_lengths)} != number of shards {len(self.paths)}"
                )
            lengths = [int(x) for x in shard_lengths]
            print(
                f"Using shard_lengths from manifest ({len(lengths)} shards); skipping slow probe.",
                flush=True,
            )
        else:
            print(
                f"Probing {len(self.paths)} shard files for patch counts (zip header read; "
                f"safer on Google Drive than numpy.load on every shard).",
                flush=True,
            )
            sys.stdout.flush()
            lengths = []
            t0 = time.perf_counter()
            for p in tqdm(self.paths, desc="Probe shards", mininterval=2.0, file=sys.stdout):
                nx = npz_member_leading_dim(p, "X")
                ny = npz_member_leading_dim(p, "Y")
                if nx != ny:
                    raise ValueError(f"Shard {p}: X and Y length mismatch ({nx} vs {ny})")
                lengths.append(nx)
            print(f"Probe finished in {time.perf_counter() - t0:.1f}s", flush=True)

        self._cum: List[int] = [0]
        for n in lengths:
            self._cum.append(self._cum[-1] + n)
        self._total = self._cum[-1]

        if indices is None:
            self._indices = np.arange(self._total, dtype=np.int64)
        else:
            self._indices = np.asarray(indices, dtype=np.int64)
            if (self._indices < 0).any() or (self._indices >= self._total).any():
                raise ValueError("indices out of range for combined shards.")

    def _get_store(self, shard_i: int) -> Any:
        if shard_i in self._lru:
            self._lru.move_to_end(shard_i)
            return self._lru[shard_i]
        while len(self._lru) >= self.max_open:
            _old_i, old_z = self._lru.popitem(last=False)
            try:
                old_z.close()
            except Exception:
                pass
        z = np.load(self.paths[shard_i], mmap_mode="r")
        self._lru[shard_i] = z
        return z

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, i: int):
        g = int(self._indices[i])
        shard_i = bisect.bisect_right(self._cum, g) - 1
        local = g - self._cum[shard_i]
        z = self._get_store(shard_i)
        x = np.asarray(z["X"][local], dtype=np.float32)
        y = np.asarray(z["Y"][local], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def close(self):
        for _i, z in list(self._lru.items()):
            try:
                z.close()
            except Exception:
                pass
        self._lru.clear()


class ShardedPatchFastaiDataset(Dataset):
    """
    CVPR23 fastai CNNLSTM expects flattened channels per time step: (T*C, H, W).

    Wraps mmap shard rows (X: T,C,H,W; Y: H,W) with optional flips/rotations like tile-date training.
    """

    def __init__(
        self,
        shard_paths: Sequence[Union[str, Path]],
        indices: np.ndarray,
        augment_geom: bool = False,
        seq_len: int = 10,
        in_ch: int = 8,
        max_open_shards: int = 4,
        shard_lengths: Optional[Sequence[int]] = None,
    ):
        self._inner = ShardedNpzPatchDataset(
            shard_paths,
            indices=indices,
            max_open_shards=max_open_shards,
            shard_lengths=shard_lengths,
        )
        self.augment_geom = bool(augment_geom)
        self.seq_len = int(seq_len)
        self.in_ch = int(in_ch)
        self._tc = self.seq_len * self.in_ch

    def __len__(self) -> int:
        return len(self._inner)

    @staticmethod
    def _geom_augment(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            x = torch.flip(x, (-1,))
            y = torch.flip(y, (-1,))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, (-2,))
            y = torch.flip(y, (-2,))
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, (-2, -1))
            y = torch.rot90(y, k, (-2, -1))
        return x.contiguous(), y.contiguous()

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._inner[i]
        if x.dim() != 4:
            raise ValueError(f"Expected X (T,C,H,W), got shape {tuple(x.shape)}")
        t, c, h, w = x.shape
        if t != self.seq_len or c != self.in_ch:
            raise ValueError(
                f"Shard X shape mismatch: got (T,C,H,W)=({t},{c},{h},{w}), "
                f"expected T={self.seq_len}, C={self.in_ch}"
            )
        xf = x.reshape(self._tc, h, w)
        yt = y
        if self.augment_geom:
            xf, yt = self._geom_augment(xf, yt)
        return xf.float(), yt.float()

    def close(self) -> None:
        self._inner.close()


def build_loyo_shard_indices(
    shard_paths: Sequence[Union[str, Path]],
    val_year: int,
    train_years: Optional[Sequence[int]] = None,
    shard_lengths: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Global patch indices (concatenated shard order) for LOYO train/val.

    Each .npz must contain KEY (N, 3) int32: (year, period_idx, tile_id) per patch row,
    aligned with X and Y. Re-run process_5d_tiles.ipynb (KEY-writing version) if missing.
    """
    paths = [Path(p) for p in shard_paths]
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    cum = 0
    vy = int(val_year)
    ty_set: Optional[set] = None
    if train_years is not None:
        ty_set = {int(y) for y in train_years}

    for spi, sp in enumerate(paths):
        z = np.load(sp, mmap_mode="r")
        try:
            if "KEY" not in z.files:
                raise ValueError(
                    f"{sp} has no KEY array. Re-run the updated process_5d_tiles.ipynb "
                    "so each shard stores KEY (year, period_idx, tile_id) per patch for LOYO."
                )
            keys = np.asarray(z["KEY"])
            if keys.ndim != 2 or keys.shape[1] != 3:
                raise ValueError(
                    f"{sp}: KEY must have shape (N, 3) int (year, period_idx, tile_id); got {keys.shape}"
                )
            nloc = int(keys.shape[0])
            if shard_lengths is not None:
                expected = int(shard_lengths[spi])
                if nloc != expected:
                    raise ValueError(
                        f"{sp}: KEY length {nloc} != manifest shard_patch_counts[{spi}]={expected}"
                    )
            ycol = keys[:, 0].astype(np.int64, copy=False)
            glob = cum + np.arange(nloc, dtype=np.int64)
            if ty_set is None:
                tr_m = ycol != vy
            else:
                ty_arr = np.array(sorted(ty_set), dtype=np.int64)
                tr_m = np.isin(ycol, ty_arr) & (ycol != vy)
            va_m = ycol == vy
            if tr_m.any():
                train_parts.append(glob[tr_m])
            if va_m.any():
                val_parts.append(glob[va_m])
            cum += nloc
        finally:
            z.close()

    train_idx = (
        np.concatenate(train_parts, axis=0)
        if train_parts
        else np.zeros(0, dtype=np.int64)
    )
    val_idx = (
        np.concatenate(val_parts, axis=0) if val_parts else np.zeros(0, dtype=np.int64)
    )
    return train_idx, val_idx


class ShardedPatchDataModule(pl.LightningDataModule):
    """Train/val split by patch index without concatenating shards in memory."""

    def __init__(
        self,
        processed_tiles_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_split: float = 0.2,
        random_state: int = 42,
        max_open_shards: int = 4,
        loyo_val_year: Optional[int] = None,
        loyo_train_years: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.processed_tiles_dir = Path(processed_tiles_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.random_state = random_state
        self.max_open_shards = max_open_shards
        self.loyo_val_year = loyo_val_year
        self.loyo_train_years = (
            None if loyo_train_years is None else [int(y) for y in loyo_train_years]
        )
        self.train_ds = None
        self.val_ds = None
        self._full_base: Optional[ShardedNpzPatchDataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage not in (None, "fit"):
            return
        if self.train_ds is not None:
            return
        shards = sorted(self.processed_tiles_dir.glob("patch_shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"No patch_shard_*.npz in {self.processed_tiles_dir}")

        shard_lengths: Optional[List[int]] = None
        mp = self.processed_tiles_dir / "manifest.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                counts = m.get("shard_patch_counts")
                if isinstance(counts, list) and len(counts) == len(shards):
                    shard_lengths = [int(c) for c in counts]
            except (json.JSONDecodeError, OSError, TypeError):
                pass

        self._full_base = ShardedNpzPatchDataset(
            shards,
            max_open_shards=self.max_open_shards,
            shard_lengths=shard_lengths,
        )
        n = len(self._full_base)
        t0 = time.perf_counter()
        if self.loyo_val_year is not None:
            print(
                f"LOYO split: val_year={self.loyo_val_year} "
                f"train_years={self.loyo_train_years or 'all except val'} — scanning KEY in shards...",
                flush=True,
            )
            train_idx, val_idx = build_loyo_shard_indices(
                shards,
                val_year=self.loyo_val_year,
                train_years=self.loyo_train_years,
                shard_lengths=shard_lengths,
            )
            if val_idx.size == 0:
                raise ValueError(
                    f"No validation patches for val_year={self.loyo_val_year}. "
                    "Check shards contain that year in KEY[:, 0]."
                )
            if train_idx.size == 0:
                raise ValueError(
                    f"No training patches after LOYO split (val_year={self.loyo_val_year})."
                )
            print(f"LOYO index build in {time.perf_counter() - t0:.1f}s", flush=True)
            self.train_ds = Subset(self._full_base, train_idx)
            self.val_ds = Subset(self._full_base, val_idx)
        else:
            all_idx = np.arange(n, dtype=np.int64)
            print(
                f"Train/val split: {n} patches (sklearn shuffle; can take minutes if N is huge)...",
                flush=True,
            )
            train_idx, val_idx = train_test_split(
                all_idx,
                test_size=self.val_split,
                random_state=self.random_state,
                shuffle=True,
            )
            print(f"Split in {time.perf_counter() - t0:.1f}s", flush=True)
            self.train_ds = Subset(self._full_base, train_idx.tolist())
            self.val_ds = Subset(self._full_base, val_idx.tolist())
        print(
            f"Sharded patches: total={n} train={len(self.train_ds)} val={len(self.val_ds)} "
            f"(max_open_shards={self.max_open_shards})",
            flush=True,
        )

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
