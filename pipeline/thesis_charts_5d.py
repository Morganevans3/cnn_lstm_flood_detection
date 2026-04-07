"""Thesis figures from Lightning CSV logs (train_from_5d_data.ipynb cell 8)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _try_import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    return plt, pd, sns


def find_metrics_csv_under(log_root: Path, name_prefix: str) -> List[Tuple[Optional[int], Path]]:
    """
    Find metrics.csv from CSVLogger runs under log_root/name_prefix*/version_*/metrics.csv
    or log_root/**/metrics.csv filtered by parent path containing val_YEAR.
    """
    out: List[Tuple[Optional[int], Path]] = []
    log_root = Path(log_root)
    if not log_root.is_dir():
        return out

    for metrics_path in sorted(log_root.rglob("metrics.csv")):
        rel = str(metrics_path.relative_to(log_root))
        m = re.search(r"val[_-]?(\d{4})", rel, re.I)
        vy = int(m.group(1)) if m else None
        if name_prefix and name_prefix not in str(metrics_path):
            continue
        out.append((vy, metrics_path))

    return out


def load_val_loss_series(metrics_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (epochs_or_steps, val_loss) from Lightning metrics.csv."""
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    if "val_loss" not in df.columns:
        raise ValueError(f"{metrics_csv}: no val_loss column; got {list(df.columns)}")
    vl = df["val_loss"].to_numpy(dtype=np.float64)
    if "epoch" in df.columns:
        x = df["epoch"].to_numpy(dtype=np.float64)
    else:
        x = np.arange(len(vl), dtype=np.float64)
    return x, vl


def save_thesis_charts_from_lightning_logs(
    project_root: Path,
    *,
    logs_subdir: str = "logs",
    fig_subdir: str = "thesis/figures/train_5d_lightning",
    run_name_contains: str = "",
) -> Dict[str, Any]:
    """
    Scan ``project_root/logs`` for Lightning ``metrics.csv``, write bar + loss plots under
    ``project_root/thesis/figures/...``.

    Returns a small dict with paths and per-fold min val_loss for JSON export.
    """
    plt, pd, sns = _try_import_plotting()

    project_root = Path(project_root)
    log_root = project_root / logs_subdir
    fig_dir = project_root / fig_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    found = find_metrics_csv_under(log_root, run_name_contains)

    sns.set_theme(style="whitegrid", context="talk")

    for vy, mpath in found:
        try:
            dfm = pd.read_csv(mpath)
            vlcol = None
            for c in ("val_loss", "val_loss_epoch", "val/loss"):
                if c in dfm.columns:
                    vlcol = c
                    break
            if vlcol is None:
                raise ValueError(f"no val_loss column; got {list(dfm.columns)}")
            vl = dfm[vlcol].to_numpy(dtype=np.float64)
            x = (
                dfm["epoch"].to_numpy(dtype=np.float64)
                if "epoch" in dfm.columns
                else np.arange(len(dfm), dtype=np.float64)
            )
        except Exception as e:
            rows.append({"metrics_csv": str(mpath), "val_year": vy, "error": str(e)})
            continue
        good = np.isfinite(vl)
        if not good.any():
            rows.append({"metrics_csv": str(mpath), "val_year": vy, "error": "no finite val_loss"})
            continue
        vmin = float(np.nanmin(vl[good]))
        imin = int(np.nanargmin(vl[good]))
        rows.append(
            {
                "val_year": vy,
                "metrics_csv": str(mpath),
                "min_val_loss": vmin,
                "epoch_at_min": float(x[imin]) if imin < len(x) else None,
            }
        )

        tag = f"val{vy}" if vy is not None else mpath.parent.name
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, vl, "s-", color="#2c7fb8", label="val_loss")
        ax.set_xlabel("epoch" if "epoch" in dfm.columns else "step")
        ax.set_ylabel("val_loss")
        ax.set_title(f"Validation loss (LOYO holdout {vy})" if vy else "Validation loss")
        ax.legend()
        fig.tight_layout()
        fp = fig_dir / f"loss_curve_{tag}.png"
        fig.savefig(fp, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Bar chart: best val_loss per fold
    summary = [r for r in rows if "min_val_loss" in r and r.get("val_year") is not None]
    summary.sort(key=lambda r: r["val_year"])
    if summary:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xs = [str(int(r["val_year"])) for r in summary]
        ys = [r["min_val_loss"] for r in summary]
        ax.bar(xs, ys, color="#35978f", edgecolor="white")
        ax.set_xlabel("Held-out validation year")
        ax.set_ylabel("Best val_loss (min over epochs)")
        ax.set_title("LOYO folds: minimum validation loss by held-out year")
        fig.tight_layout()
        fp_bar = fig_dir / "bar_min_val_loss_by_fold.png"
        fig.savefig(fp_bar, bbox_inches="tight", dpi=150)
        plt.close(fig)

    out_path = fig_dir / "summary.json"
    meta = {
        "figures_dir": str(fig_dir),
        "folds": rows,
        "bar_chart": str(fig_dir / "bar_min_val_loss_by_fold.png") if summary else None,
    }
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
