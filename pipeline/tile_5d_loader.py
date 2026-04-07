"""Load 5° MODIS GeoTIFF stacks + S1 flood labels (.npy) for CNN-LSTM training.

File naming (same as ``process_5d_tiles.ipynb``):

- MODIS: ``Indo_5d_MODIS_<year>_<period>_..._<tile_id>.tif``
- S1: ``Indo_5d_S1_<year>_<period>_..._<tile_id>.npy``

Keys are ``(year, period_idx, tile_id)`` with ``period_idx`` counting 8-day periods
within the calendar year.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None  # type: ignore[misc, assignment]

try:
    import rioxarray
except ImportError:
    rioxarray = None  # type: ignore[misc, assignment]

from scipy.ndimage import zoom

__all__ = [
    "Tile5dLoadContext",
    "audit_load_sample_rate",
    "center_crop_or_pad",
    "discover_tile_paths",
    "filter_keys_by_load",
    "parse_modis_name",
    "parse_s1_name",
]

Key = Tuple[int, int, int]


def audit_load_sample_rate(
    ctx: "Tile5dLoadContext",
    keys: Sequence[Key],
    rng: np.random.Generator,
    *,
    n_sample: int = 500,
) -> Tuple[int, int, int]:
    """
    Try ``load_sample`` on up to ``n_sample`` random keys.

    Returns ``(n_ok, n_none, n_checked)`` where ``n_none`` is the count of keys for which
    loading failed (dataset falls back to zeros — no GeoTIFF I/O, very fast steps).
    """
    keys = list(keys)
    n = len(keys)
    if n == 0:
        return 0, 0, 0
    k = min(int(n_sample), n)
    idx = rng.choice(n, size=k, replace=False)
    n_ok = 0
    for i in idx:
        yr, p_idx, tid = keys[i]
        if ctx.load_sample(yr, p_idx, tid) is not None:
            n_ok += 1
    return n_ok, k - n_ok, k


def filter_keys_by_load(
    ctx: "Tile5dLoadContext",
    keys: Sequence[Key],
    *,
    desc: str = "Verify tile loads",
) -> Tuple[List[Key], int]:
    """
    Keep only keys where ``load_sample`` succeeds. Every training/val step will then hit real rasters.

    Returns ``(keys_ok, n_dropped)``. Uses ``tqdm`` when available. One-time cost per key list (warms LRU cache).
    """
    keys = list(keys)
    good: List[Key] = []
    n_fail = 0
    try:
        from tqdm.auto import tqdm

        it = tqdm(keys, desc=desc, unit="key")
    except ImportError:
        it = keys
    for k in it:
        yr, p_idx, tid = k
        if ctx.load_sample(yr, p_idx, tid) is not None:
            good.append(k)
        else:
            n_fail += 1
    return good, n_fail


def parse_modis_name(fname: str) -> Optional[Key]:
    stem = Path(fname).stem
    if not stem.startswith("Indo_5d_MODIS_"):
        return None
    parts = stem.replace("Indo_5d_MODIS_", "").split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[-1])
    except ValueError:
        return None


def parse_s1_name(fname: str) -> Optional[Key]:
    stem = Path(fname).stem
    if not stem.startswith("Indo_5d_S1_"):
        return None
    parts = stem.replace("Indo_5d_S1_", "").split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[-1])
    except ValueError:
        return None


def discover_tile_paths(
    modis_dir: Path,
    s1_dir: Path,
) -> Tuple[Dict[Key, Path], Dict[Key, Path], Set[Key]]:
    modis_dir = Path(modis_dir)
    s1_dir = Path(s1_dir)
    modis_files = {parse_modis_name(f.name): f for f in modis_dir.glob("*.tif") if parse_modis_name(f.name)}
    s1_files = {parse_s1_name(f.name): f for f in s1_dir.glob("*.npy") if parse_s1_name(f.name)}
    common = set(modis_files.keys()) & set(s1_files.keys())
    return modis_files, s1_files, common


def center_crop_or_pad(
    x: np.ndarray,
    y: np.ndarray,
    img_hw: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop or zero-pad ``x`` (T, C, H, W) and ``y`` (H, W) to ``img_hw``."""
    hw = int(img_hw)
    _, _, H, W = x.shape
    if H < hw or W < hw:
        ph, pw = max(0, hw - H), max(0, hw - W)
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
        y = np.pad(y, ((pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
        H, W = x.shape[2], x.shape[3]
    if H > hw or W > hw:
        ch, cw = (H - hw) // 2, (W - hw) // 2
        x = x[:, :, ch : ch + hw, cw : cw + hw]
        y = y[ch : ch + hw, cw : cw + hw]
    return x, y


def _count_8day_periods(year: int) -> int:
    d = datetime(int(year), 1, 1)
    n = 0
    while d.year == int(year):
        n += 1
        d += timedelta(days=8)
    return n


_PERIOD_COUNT: Dict[int, int] = {}


def _get_8day_periods(year: int) -> int:
    if year not in _PERIOD_COUNT:
        _PERIOD_COUNT[year] = _count_8day_periods(year)
    return _PERIOD_COUNT[year]


def _trim(cache: "OrderedDict", max_size: int) -> None:
    while len(cache) > max_size:
        cache.popitem(last=False)


def _resolve_sequence_keys(
    year: int,
    p_idx: int,
    tile_id: int,
    sequence_length: int,
    modis_files: Dict[Key, Path],
) -> Optional[list[Key]]:
    keys: list[Key] = []
    for t in range(sequence_length):
        pp = p_idx - (sequence_length - 1 - t)
        yy = int(year)
        while pp < 0:
            yy -= 1
            pp += _get_8day_periods(yy)
        while pp >= _get_8day_periods(yy):
            pp -= _get_8day_periods(yy)
            yy += 1
        key = (yy, pp, int(tile_id))
        if key not in modis_files:
            return None
        keys.append(key)
    return keys


class Tile5dLoadContext:
    """LRU-cached MODIS/S1 loads + ``load_sample`` / ``strict_score_fast``."""

    def __init__(
        self,
        modis_files: Dict[Key, Path],
        s1_files: Dict[Key, Path],
        *,
        sequence_length: int = 10,
        s1_percentile: float = 20.0,
        modis_cache_max: int = 64,
        s1_cache_max: int = 128,
        min_valid_s1_pixels: int = 100,
    ):
        self.modis_files = modis_files
        self.s1_files = s1_files
        self.sequence_length = int(sequence_length)
        self.s1_percentile = float(s1_percentile)
        self.modis_cache_max = int(modis_cache_max)
        self.s1_cache_max = int(s1_cache_max)
        self.min_valid_s1_pixels = int(min_valid_s1_pixels)
        self._modis_cache: "OrderedDict[Key, np.ndarray]" = OrderedDict()
        self._s1_cache: "OrderedDict[Key, np.ndarray]" = OrderedDict()

    def _get_modis_array(self, key: Key) -> Optional[np.ndarray]:
        if rioxarray is None or rasterio is None:
            return None
        if key in self._modis_cache:
            self._modis_cache.move_to_end(key)
            return self._modis_cache[key]
        path = self.modis_files.get(key)
        if path is None:
            return None
        try:
            da = rioxarray.open_rasterio(path)
            arr = np.asarray(da).squeeze().astype(np.float32, copy=False)
        except Exception:
            # Corrupt GeoTIFF / I/O — skip tile (same spirit as process_5d_tiles.ipynb)
            return None
        self._modis_cache[key] = arr
        self._modis_cache.move_to_end(key)
        _trim(self._modis_cache, self.modis_cache_max)
        return arr

    def _get_s1_array(self, key: Key) -> Optional[np.ndarray]:
        if key in self._s1_cache:
            self._s1_cache.move_to_end(key)
            return self._s1_cache[key]
        path = self.s1_files.get(key)
        if path is None:
            return None
        try:
            s1 = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        except Exception:
            return None
        self._s1_cache[key] = s1
        self._s1_cache.move_to_end(key)
        _trim(self._s1_cache, self.s1_cache_max)
        return s1

    def load_sample(self, year: int, p_idx: int, tile_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        seq_keys = _resolve_sequence_keys(
            year, p_idx, tile_id, self.sequence_length, self.modis_files
        )
        if seq_keys is None:
            return None

        modis_list: list[np.ndarray] = []
        for key in seq_keys:
            arr = self._get_modis_array(key)
            if arr is None:
                return None
            modis_list.append(arr)

        label_key: Key = (int(year), int(p_idx), int(tile_id))
        if label_key not in self.s1_files:
            return None
        s1 = self._get_s1_array(label_key)
        if s1 is None:
            return None

        modis_stack = np.stack(modis_list, axis=0)
        del modis_list

        if modis_stack.ndim == 4 and modis_stack.shape[1] == 7:
            pass
        elif modis_stack.ndim == 4 and modis_stack.shape[-1] == 7:
            modis_stack = np.transpose(modis_stack, (0, 3, 1, 2))
        else:
            return None

        target_h, target_w = modis_stack.shape[2], modis_stack.shape[3]
        if s1.shape != (target_h, target_w):
            sy, sx = target_h / s1.shape[0], target_w / s1.shape[1]
            s1 = zoom(s1, (sy, sx), order=1)

        valid = np.isfinite(s1) & (s1 > -999)
        if valid.sum() < self.min_valid_s1_pixels:
            return None
        thresh = np.percentile(s1[valid], self.s1_percentile)
        flood = np.where(valid, (s1 < thresh).astype(np.float32), np.nan)
        flood = np.nan_to_num(flood, nan=0.0)

        hand = np.zeros((target_h, target_w), dtype=np.float32)
        hand_expanded = np.repeat(hand[np.newaxis, np.newaxis, :, :], self.sequence_length, axis=0)
        modis_norm = modis_stack / 10000.0
        modis_norm = np.clip(np.nan_to_num(modis_norm, nan=0.0), 0.0, 1.0)
        del modis_stack

        x = np.concatenate([modis_norm, hand_expanded], axis=1)
        del modis_norm, hand_expanded
        y = flood
        gc.collect()
        return x, y

    def strict_score_fast(
        self,
        year: int,
        p_idx: int,
        tile_id: int,
        img_hw: int,
    ) -> Optional[dict]:
        """Lightweight scores for quality ranking (``quality_tile_ranking``)."""
        hw = int(img_hw)
        seq_keys = _resolve_sequence_keys(
            year, p_idx, tile_id, self.sequence_length, self.modis_files
        )
        if seq_keys is None:
            return None
        label_key: Key = (int(year), int(p_idx), int(tile_id))
        if label_key not in self.s1_files:
            return None

        modis_list: list[np.ndarray] = []
        for key in seq_keys:
            arr = self._get_modis_array(key)
            if arr is None:
                return None
            if arr.ndim == 3 and arr.shape[-1] == 7:
                arr = np.transpose(arr, (2, 0, 1))
            if arr.ndim != 3 or arr.shape[0] != 7:
                return None
            modis_list.append(arr)

        modis_stack = np.stack(modis_list, axis=0)
        H, W = modis_stack.shape[2], modis_stack.shape[3]
        ch, cw = max(0, (H - hw) // 2), max(0, (W - hw) // 2)
        patch_m = modis_stack[:, :, ch : ch + hw, cw : cw + hw]
        finite = np.isfinite(patch_m)
        modis_valid_frac = float(np.mean(finite.all(axis=(0, 1))))

        s1 = self._get_s1_array(label_key)
        if s1 is None:
            return None
        if s1.shape != (H, W):
            sy, sx = H / s1.shape[0], W / s1.shape[1]
            s1 = zoom(s1, (sy, sx), order=1)
        patch_s = s1[ch : ch + hw, cw : cw + hw]
        v = np.isfinite(patch_s) & (patch_s > -999)
        if v.sum() < 10:
            return None
        pv = patch_s[v]
        s1_contrast = float(np.std(pv) / (np.median(np.abs(pv)) + 1e-6))

        return {
            "modis_valid_frac": modis_valid_frac,
            "s1_contrast": s1_contrast,
        }
