"""Lightning callbacks for Colab: visible progress and logs that persist on Drive."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytorch_lightning as pl


class EpochFileLogger(pl.Callback):
    """Append one line per validation epoch to a CSV on disk (survives Colab disconnect)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def on_fit_start(self, trainer: pl.Trainer, pl_module) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "event,utc_time,epoch,train_loss,val_loss\n"
            f"fit_start,{datetime.now(timezone.utc).isoformat()},,,\n",
            encoding="utf-8",
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        cm = trainer.callback_metrics
        vl = cm.get("val_loss_epoch", cm.get("val_loss", ""))
        tl = cm.get("train_loss_epoch", cm.get("train_loss", ""))
        now = datetime.now(timezone.utc).isoformat()
        line = f"epoch_end,{now},{trainer.current_epoch},{tl},{vl}\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        print(
            f"[epoch {trainer.current_epoch}] train_loss={tl!s} val_loss={vl!s}",
            flush=True,
        )


class FirstBatchPrinter(pl.Callback):
    """Print once when the first training batch completes (confirms DataLoader is not stuck)."""

    def __init__(self) -> None:
        self._done = False

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self._done:
            return
        batch_idx = kwargs.get("batch_idx")
        if batch_idx is None and args:
            batch_idx = args[-1]
        if batch_idx != 0:
            return
        self._done = True
        print("First batch done — training loop is running.", flush=True)
