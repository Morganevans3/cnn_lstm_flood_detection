"""
Copy 5° MODIS + S1 folders from Google Drive (or project tree) to Colab local disk.

Training from /content/... avoids Drive FUSE random I/O, which is the main cause of
multi-hour "epochs" with no progress. Use rsync when available (resume, partial).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _count_globs(d: Path, pattern: str) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob(pattern))


def _which_rsync() -> Optional[str]:
    for name in ("rsync",):
        p = shutil.which(name)
        if p:
            return p
    return None


def _rsync_dir(src: Path, dst: Path, retries: int = 3) -> bool:
    """rsync -a from src/ to dst/. Creates dst. Returns True on success."""
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    rsync = _which_rsync()
    if not rsync:
        return False
    cmd = [
        rsync,
        "-a",
        "--info=stats2",
        "--partial",
        "--ignore-existing",
        f"{str(src)}/",
        f"{str(dst)}/",
    ]
    for attempt in range(retries):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return True
        err = (proc.stderr or "") + (proc.stdout or "")
        if "Transport endpoint is not connected" in err or "Stale file handle" in err:
            raise OSError(
                "Drive FUSE failed during rsync (transport endpoint / stale handle). "
                "In a new cell run: drive.mount('/content/drive', force_remount=True) "
                "then re-run the sync cell."
            )
        time.sleep(min(8, 2 ** attempt))
    return False


def _copy_tree_python(
    src: Path,
    dst: Path,
    pattern: str,
    *,
    label: str,
    every: int = 200,
) -> None:
    """Copy files matching pattern from src to dst with per-file retry."""
    src, dst = src.resolve(), dst.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Missing source dir: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(pattern))
    n = len(files)
    if n == 0:
        raise FileNotFoundError(f"No files {pattern!r} under {src}")
    t0 = time.perf_counter()
    for i, fp in enumerate(files):
        out = dst / fp.name
        if out.is_file() and out.stat().st_size == fp.stat().st_size:
            continue
        for attempt in range(4):
            try:
                shutil.copy2(fp, out)
                break
            except OSError as e:
                if attempt == 3:
                    raise
                if "Transport endpoint" in str(e) or "Stale file handle" in str(e):
                    raise OSError(
                        f"{e!s}. Remount Drive: "
                        "drive.mount('/content/drive', force_remount=True)"
                    ) from e
                time.sleep(min(8, 2 ** attempt))
        if (i + 1) % every == 0 or (i + 1) == n:
            dt = time.perf_counter() - t0
            print(
                f"  [{label}] copied {i + 1}/{n} files ({dt:.0f}s elapsed)",
                flush=True,
            )


def ensure_local_5d_copy(
    project_root: Path,
    local_root: Path,
    paths_cfg: Dict[str, Any],
    *,
    prefer_rsync: bool = True,
) -> Tuple[Path, Path]:
    """
    Ensure MODIS + S1 5° tile dirs exist under ``local_root`` (fast local disk).

    Reads MODIS path from ``paths_cfg``: ``modis_5d`` (preferred) or legacy ``modis_8day_5d``,
    and ``s1_labels_5d``.
    Returns ``(local_modis_dir, local_s1_dir)``.
    """
    rel_m = str(
        paths_cfg.get("modis_5d")
        or paths_cfg.get("modis_8day_5d")
        or "data/modis_5d"
    )
    rel_s = str(paths_cfg.get("s1_labels_5d", "data/s1_labels_5d"))
    src_m = (project_root / rel_m).resolve()
    src_s = (project_root / rel_s).resolve()
    if not src_m.is_dir():
        raise FileNotFoundError(
            f"MODIS dir not found: {src_m}. Check paths.modis_5d (or paths.modis_8day_5d) in config.yaml."
        )
    if not src_s.is_dir():
        raise FileNotFoundError(
            f"S1 dir not found: {src_s}. Check paths.s1_labels_5d in config.yaml."
        )

    local_root = Path(local_root).resolve()
    dst_m = local_root / Path(rel_m).name
    dst_s = local_root / Path(rel_s).name

    nm, ns = _count_globs(src_m, "*.tif"), _count_globs(src_s, "*.npy")
    if nm == 0:
        raise FileNotFoundError(f"No *.tif in {src_m}")
    if ns == 0:
        raise FileNotFoundError(f"No *.npy in {src_s}")

    os.environ.setdefault("CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    print(
        f"Local 5° cache: {local_root}\n"
        f"  Source MODIS: {src_m} ({nm} tifs)\n"
        f"  Source S1:    {src_s} ({ns} npy)\n"
        f"  Target MODIS: {dst_m}\n"
        f"  Target S1:    {dst_s}",
        flush=True,
    )

    def _need_copy(dst: Path, src_n: int, pat: str) -> bool:
        got = _count_globs(dst, pat)
        return got < max(1, int(src_n * 0.98))

    if not _need_copy(dst_m, nm, "*.tif") and not _need_copy(dst_s, ns, "*.npy"):
        print("  Cache already populated (skipping copy).", flush=True)
        return dst_m, dst_s

    t_all = time.perf_counter()
    if prefer_rsync and _which_rsync():
        print("  Using rsync (resume-friendly)...", flush=True)
        ok_m = _rsync_dir(src_m, dst_m)
        ok_s = _rsync_dir(src_s, dst_s)
        if not ok_m or not ok_s:
            print("  rsync failed or unavailable; falling back to Python copy.", flush=True)
            _copy_tree_python(src_m, dst_m, "*.tif", label="MODIS")
            _copy_tree_python(src_s, dst_s, "*.npy", label="S1")
    else:
        print("  Using Python copy (install rsync in VM for faster resume)...", flush=True)
        _copy_tree_python(src_m, dst_m, "*.tif", label="MODIS")
        _copy_tree_python(src_s, dst_s, "*.npy", label="S1")

    # Verify
    cm, cs = _count_globs(dst_m, "*.tif"), _count_globs(dst_s, "*.npy")
    print(
        f"  Done in {time.perf_counter() - t_all:.0f}s. Local counts: MODIS={cm} S1={cs}",
        flush=True,
    )
    if cm < nm * 0.95 or cs < ns * 0.95:
        print(
            "  WARNING: local file count is below source — "
            "check Drive connection or re-run sync.",
            flush=True,
        )
    return dst_m, dst_s
