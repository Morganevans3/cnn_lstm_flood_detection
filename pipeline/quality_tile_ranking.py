"""
Rank 5° tile-date keys for curriculum / high-quality training.

Combines MODIS validity + S1 contrast (from ``Tile5dLoadContext.strict_score_fast``) with
optional static HAND/DEM signal when GeoTIFFs exist under ``paths.gee_static``.

Writes **one** CSV + JSON meta (fingerprinted) so Colab does not re-scan every run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

_TLS = threading.local()
_INIT: Optional[Tuple[Any, ...]] = None


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _optional_hand_dem_score(
    static_dir: Optional[Path],
    tile_id: int,
    img_hw: int,
) -> Optional[float]:
    """
    If a HAND (or DEM) GeoTIFF exists for this tile, return normalized spatial variability
    in the center crop (higher = more terrain structure). Else None (caller uses neutral).
    """
    if static_dir is None or not static_dir.is_dir():
        return None
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        return None

    cands: List[Path] = []
    tid = str(int(tile_id))
    for p in static_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        u = p.name.upper()
        if tid in u and ("HAND" in u or "DEM" in u):
            cands.append(p)
    if not cands:
        return None

    path = sorted(cands)[0]
    try:
        with rasterio.open(path) as src:
            H, W = int(src.height), int(src.width)
            hw = int(img_hw)
            if H < 2 or W < 2:
                return None
            if H >= hw and W >= hw:
                ch, cw = (H - hw) // 2, (W - hw) // 2
                win = Window(cw, ch, hw, hw)
                arr = src.read(1, window=win).astype(np.float32)
            else:
                arr = src.read(1).astype(np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size < 64:
            return None
        # variability relative to median scale
        sd = float(np.std(arr))
        med = float(np.median(np.abs(arr))) + 1e-6
        return float(np.clip(sd / med, 0.0, 1.0))
    except Exception:
        return None


def _score_packed_main_ctx(
    ctx: Any,
    p: Tuple[Tuple[int, int, int], int, Optional[Path]],
) -> Optional[Dict[str, Any]]:
    """Score one packed tuple using an existing ``Tile5dLoadContext`` (main thread / seq path)."""
    key, img_hw, static_dir = p
    year, p_idx, tile_id = key
    fast = ctx.strict_score_fast(year, p_idx, tile_id, img_hw)
    if fast is None:
        return None
    m = float(fast.get("modis_valid_frac", 0.0))
    s = float(fast.get("s1_contrast", 0.0))
    dem = _optional_hand_dem_score(static_dir, tile_id, img_hw)
    if dem is None:
        dem_n = 0.5
        w_m, w_s, w_d = 0.45, 0.45, 0.10
    else:
        dem_n = dem
        w_m, w_s, w_d = 0.40, 0.40, 0.20
    s_n = _norm01(s, 0.08, 0.45)
    comp = w_m * m + w_s * s_n + w_d * dem_n
    return {
        "year": year,
        "period_idx": p_idx,
        "tile_id": tile_id,
        "modis_valid_frac": m,
        "s1_contrast": s,
        "dem_hand_score": dem,
        "composite": comp,
        "key": key,
    }


def _worker_score_one(args: Tuple[Tuple[int, int, int], int, Optional[Path]]) -> Optional[Dict[str, Any]]:
    key, img_hw, static_dir = args
    init = _INIT
    if init is None:
        return None
    mf, sf, seq, pct, wm, ws = init
    if not hasattr(_TLS, "ctx"):
        from pipeline.tile_5d_loader import Tile5dLoadContext

        _TLS.ctx = Tile5dLoadContext(
            mf,
            sf,
            sequence_length=seq,
            s1_percentile=pct,
            modis_cache_max=int(wm),
            s1_cache_max=int(ws),
        )
    ctx = _TLS.ctx
    return _score_packed_main_ctx(ctx, args)


def merge_quality_rows(
    primary: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge score rows; ``extra`` overwrites same ``key`` as in ``primary``."""
    byk: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in primary:
        byk[tuple(r["key"])] = r
    for r in extra:
        byk[tuple(r["key"])] = r
    return list(byk.values())


def rank_keys_parallel(
    keys: Sequence[Tuple[int, int, int]],
    *,
    modis_files,
    s1_files,
    sequence_length: int,
    s1_percentile: float,
    img_hw: int,
    modis_cache_max: int = 64,
    s1_cache_max: int = 128,
    static_dir: Optional[Path] = None,
    num_workers: int = 4,
    show_progress: bool = True,
    per_key_timeout_sec: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, int]]]:
    """
    Score keys in parallel (or sequentially if ``num_workers==1``).

    If ``per_key_timeout_sec`` is set, futures that exceed that wall time since submit
    are abandoned for this round and their keys are returned in the second list (worker
    threads may still finish in the pool — unavoidable with threads).

    Returns ``(rows, skipped_keys)`` where ``skipped_keys`` are timeout abandons only.
    """
    global _INIT
    packed = [(k, img_hw, static_dir) for k in keys]
    initargs = (modis_files, s1_files, sequence_length, s1_percentile, modis_cache_max, s1_cache_max)
    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[int, int, int]] = []
    n = len(packed)
    if n == 0:
        return rows, skipped
    nw = max(1, int(num_workers))
    timeout = float(per_key_timeout_sec) if per_key_timeout_sec is not None else None

    from pipeline.tile_5d_loader import Tile5dLoadContext

    if nw == 1:
        ctx = Tile5dLoadContext(
            modis_files,
            s1_files,
            sequence_length=sequence_length,
            s1_percentile=s1_percentile,
            modis_cache_max=modis_cache_max,
            s1_cache_max=s1_cache_max,
        )
        for p in packed:
            key = p[0]
            if timeout and timeout > 0:
                box: List[Optional[Dict[str, Any]]] = [None]

                def run(pp: Tuple[Tuple[int, int, int], int, Optional[Path]] = p) -> None:
                    box[0] = _score_packed_main_ctx(ctx, pp)

                t = threading.Thread(target=run, daemon=True)
                t.start()
                t.join(timeout=timeout)
                if t.is_alive():
                    skipped.append(key)
                    print(
                        f"Quality skip (timeout {timeout}s): {key}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                r = box[0]
            else:
                r = _score_packed_main_ctx(ctx, p)
            if r is not None:
                rows.append(r)
        return rows, skipped

    try:
        _INIT = initargs
        with ThreadPoolExecutor(max_workers=nw) as ex:
            pending: Dict[Any, Tuple[Tuple[int, int, int], int, Optional[Path]]] = {}
            submit_time: Dict[Any, float] = {}
            for p in packed:
                fut = ex.submit(_worker_score_one, p)
                pending[fut] = p
                submit_time[fut] = time.monotonic()

            pbar = None
            if show_progress and tqdm is not None:
                pbar = tqdm(total=n, desc="Quality scoring", unit="key", mininterval=0.2)

            while pending:
                wait_timeout = 0.5 if (timeout and timeout > 0) else None
                done, not_done = wait(
                    pending.keys(), timeout=wait_timeout, return_when=FIRST_COMPLETED
                )
                now = time.monotonic()

                for f in done:
                    p = pending.pop(f)
                    try:
                        r = f.result()
                    except Exception:
                        r = None
                    if r is not None:
                        rows.append(r)
                    if pbar is not None:
                        pbar.update(1)

                if timeout and timeout > 0:
                    for f in list(not_done):
                        if f in pending and not f.done() and now - submit_time[f] > timeout:
                            p = pending.pop(f)
                            skipped.append(p[0])
                            print(
                                f"Quality skip (timeout {timeout}s): {p[0]}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if pbar is not None:
                                pbar.update(1)
            if pbar is not None:
                pbar.close()
    finally:
        _INIT = None
    return rows, skipped


def select_top_balanced_per_year(
    rows: List[Dict[str, Any]],
    *,
    max_total: int = 2000,
) -> List[Tuple[int, int, int]]:
    """
    Take up to ``max_total`` keys: equal quota per calendar year (each year sorted by composite).
    If a year has fewer keys than quota, remaining slots are filled from global surplus (highest composite).
    """
    from collections import defaultdict

    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_year[int(r["year"])].append(r)
    years = sorted(y for y, v in by_year.items() if v)
    if not years:
        return []
    quota = max_total // len(years)
    picked: List[Tuple[int, int, int]] = []
    leftovers: List[Dict[str, Any]] = []

    for y in years:
        part = sorted(by_year[y], key=lambda x: -float(x["composite"]))
        take = part[:quota]
        for r in take:
            picked.append(r["key"])
        for r in part[quota:]:
            leftovers.append(r)

    if len(picked) < max_total and leftovers:
        leftovers.sort(key=lambda x: -float(x["composite"]))
        for r in leftovers:
            if len(picked) >= max_total:
                break
            picked.append(r["key"])
    return picked[:max_total]


def _cache_fingerprint_v1(
    keys: Sequence[Tuple[int, int, int]],
    img_hw: int,
    max_total: int,
) -> str:
    h = hashlib.sha256()
    h.update(f"{int(img_hw)},{int(max_total)}\n".encode())
    for y, p, t in keys:
        h.update(f"{int(y)},{int(p)},{int(t)}\n".encode())
    return h.hexdigest()


def cache_fingerprint(
    keys: Sequence[Tuple[int, int, int]],
    img_hw: int,
    max_total: int,
    per_key_timeout_sec: float = 0.0,
) -> str:
    h = hashlib.sha256()
    h.update(f"{int(img_hw)},{int(max_total)},{float(per_key_timeout_sec)}\n".encode())
    for y, p, t in keys:
        h.update(f"{int(y)},{int(p)},{int(t)}\n".encode())
    return h.hexdigest()


def save_quality_cache(
    cache_dir: Path,
    rows: List[Dict[str, Any]],
    selected_keys: List[Tuple[int, int, int]],
    *,
    img_hw: int,
    max_total: int,
    scan_keys: Sequence[Tuple[int, int, int]],
    per_key_timeout_sec: float = 0.0,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "best_tile_keys_ranked.csv"
    meta_path = cache_dir / "best_tile_keys_ranked.meta.json"
    import csv

    fieldnames = [
        "year",
        "period_idx",
        "tile_id",
        "modis_valid_frac",
        "s1_contrast",
        "dem_hand_score",
        "composite",
        "selected_for_training",
    ]
    sel_set = set(selected_keys)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (-float(x["composite"]), x["year"], x["period_idx"], x["tile_id"])):
            k = r["key"]
            w.writerow(
                {
                    "year": r["year"],
                    "period_idx": r["period_idx"],
                    "tile_id": r["tile_id"],
                    "modis_valid_frac": f"{r['modis_valid_frac']:.6f}",
                    "s1_contrast": f"{r['s1_contrast']:.6f}",
                    "dem_hand_score": "" if r.get("dem_hand_score") is None else f"{float(r['dem_hand_score']):.6f}",
                    "composite": f"{r['composite']:.6f}",
                    "selected_for_training": int(k in sel_set),
                }
            )

    meta = {
        "version": 2,
        "img_hw": int(img_hw),
        "max_total": int(max_total),
        "per_key_timeout_sec": float(per_key_timeout_sec),
        "scan_fp": cache_fingerprint(scan_keys, img_hw, max_total, per_key_timeout_sec),
        "n_scored": len(rows),
        "n_selected": len(selected_keys),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote quality cache:\n  {csv_path}\n  {meta_path}", flush=True)


def load_quality_cache_if_valid(
    cache_dir: Path,
    scan_keys: Sequence[Tuple[int, int, int]],
    *,
    img_hw: int,
    max_total: int,
    per_key_timeout_sec: float = 0.0,
    force_recompute: bool = False,
) -> Optional[List[Tuple[int, int, int]]]:
    if force_recompute:
        return None
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "best_tile_keys_ranked.meta.json"
    csv_path = cache_dir / "best_tile_keys_ranked.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ver = int(meta.get("version", 0))
    if ver not in (1, 2):
        return None
    if int(meta.get("img_hw", -1)) != int(img_hw):
        return None
    if int(meta.get("max_total", -1)) != int(max_total):
        return None
    if ver >= 2 and float(meta.get("per_key_timeout_sec", -1.0)) != float(per_key_timeout_sec):
        return None
    fp = cache_fingerprint(scan_keys, img_hw, max_total, per_key_timeout_sec)
    fp_ok = meta.get("scan_fp") == fp
    if not fp_ok and ver == 1 and float(per_key_timeout_sec) == 0.0:
        fp_ok = meta.get("scan_fp") == _cache_fingerprint_v1(scan_keys, img_hw, max_total)
    if not fp_ok:
        return None
    import csv

    out: List[Tuple[int, int, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("selected_for_training", 0)) == 1:
                out.append((int(row["year"]), int(row["period_idx"]), int(row["tile_id"])))
    print(f"Loaded {len(out)} training keys from cache: {csv_path}", flush=True)
    return out
