"""Discover train/val tile-date keys for LOYO and curriculum notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from pipeline.tile_5d_loader import discover_tile_paths

Key = Tuple[int, int, int]

__all__ = [
    "discover_all_loyo_tile_keys",
    "discover_trainable_keys",
    "discover_val_keys_for_year",
]


def _trainable_base_keys(
    modis_dir: Path,
    s1_dir: Path,
    sequence_length: int,
) -> List[Key]:
    _, _, common = discover_tile_paths(modis_dir, s1_dir)
    seq = int(sequence_length)
    return sorted(k for k in common if k[1] >= seq - 1)


def discover_all_loyo_tile_keys(
    modis_dir: Path,
    s1_dir: Path,
    *,
    sequence_length: int,
    loyo_years: Iterable[int],
) -> List[Key]:
    """All overlapping keys in ``loyo_years`` with enough MODIS history for the sequence."""
    ys: Set[int] = {int(y) for y in loyo_years}
    keys = _trainable_base_keys(modis_dir, s1_dir, sequence_length)
    return [k for k in keys if k[0] in ys]


def discover_trainable_keys(
    modis_dir: Path,
    s1_dir: Path,
    *,
    sequence_length: int,
    loyo_val_year: int,
    loyo_years: Iterable[int],
    max_tile_keys: Optional[int] = None,
) -> List[Key]:
    """Training pool for LOYO: same years as ``loyo_years`` except the held-out validation year."""
    ys = {int(y) for y in loyo_years}
    keys = [
        k
        for k in _trainable_base_keys(modis_dir, s1_dir, sequence_length)
        if k[0] in ys and k[0] != int(loyo_val_year)
    ]
    if max_tile_keys is not None:
        keys = keys[: int(max_tile_keys)]
    return keys


def discover_val_keys_for_year(
    modis_dir: Path,
    s1_dir: Path,
    *,
    sequence_length: int,
    val_year: int,
    max_tile_keys: Optional[int] = None,
) -> List[Key]:
    """Validation keys for a single calendar year."""
    vy = int(val_year)
    keys = [k for k in _trainable_base_keys(modis_dir, s1_dir, sequence_length) if k[0] == vy]
    if max_tile_keys is not None:
        keys = keys[: int(max_tile_keys)]
    return keys
