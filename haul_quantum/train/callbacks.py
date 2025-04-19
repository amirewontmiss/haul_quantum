"""
haul_quantum.train.callbacks
============================
Callback classes for the Trainer in haul_quantum.train.loop.

Includes:
- Callback: base hook interface
- EarlyStopping: stop training when a monitored metric ceases improving
- ModelCheckpoint: save model parameters at epoch end
- CSVLogger: log metrics to CSV
- ProgressBar: simple console progress display
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np


class Callback:
    """Base class for Trainer callbacks."""

    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_step_end(self, epoch: int, step: int, logs: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        monitor: metric name to track (e.g. 'loss').
        patience: epochs with no improvement before stopping.
        min_delta: minimum change to qualify as improvement.
        mode: 'min' or 'max' for whether lower/higher is better.
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.wait = 0
        self.stopped_epoch: Optional[int] = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.best is None:
            self.best = current
            return
        improvement = (
            current < self.best - self.min_delta
            if self.mode == "min"
            else current > self.best + self.min_delta
        )
        if improvement:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs["stop_training"] = True


class ModelCheckpoint(Callback):
    """
    Save model parameters after each epoch or when a metric improves.

    Args:
        filepath: path template (e.g. 'checkpoints/epoch_{epoch:02d}.npz').
        monitor: metric name to track.
        save_best_only: if True, only save when metric improves.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "loss",
        save_best_only: bool = False,
        mode: str = "min",
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best: Optional[float] = None

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return True
        is_better = current < self.best if self.mode == "min" else current > self.best
        if is_better:
            self.best = current
        return is_better

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.save_best_only and not self._is_improvement(current):
            return
        params = logs.get("params")
        if params is None:
            return
        path = self.filepath.format(epoch=epoch, **logs)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, params=params)
        logging.info(f"ModelCheckpoint: saved model at {path}")


class CSVLogger(Callback):
    """
    Log metrics to a CSV file.

    Args:
        filename: CSV file path.
        fields: list of column names to log.
    """

    def __init__(self, filename: str, fields: List[str]):
        self.filename = filename
        self.fields = fields
        self.file = open(self.filename, "w", buffering=1)
        header = ",".join(self.fields)
        self.file.write(header + "\n")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        row = [str(logs.get(f, "")) for f in self.fields]
        self.file.write(",".join(row) + "\n")

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        self.file.close()


class ProgressBar(Callback):
    """
    Console-based progress display.

    Args:
        total_epochs: total number of epochs.
        steps_per_epoch: number of steps (batches) per epoch.
    """

    def __init__(self, total_epochs: int, steps_per_epoch: int = 1):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        print(f"Epoch {epoch+1}/{self.total_epochs}")

    def on_step_end(self, epoch: int, step: int, logs: Dict[str, Any]) -> None:
        print(
            f"\r [Step {step+1}/{self.steps_per_epoch}] loss={logs.get('loss', 0):.6f}",
            end="",
        )

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        print()
