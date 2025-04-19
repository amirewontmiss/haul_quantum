"""
haul_quantum.train.loop
=======================
Robust training loop for quantum-classical hybrid models.

Features:
- Supports gradient-based (parameter-shift, finite-difference), gradient-free (SPSA) optimizers.
- Callback mechanism for logging, early stopping, checkpointing, progress.
- Handles both expectation-based and loss-based training.
- Flexible interfaces for custom loss, metric, and data batching.

Key classes:
- Callback (base class for custom hooks)
- EarlyStopping
- ModelCheckpoint
- CSVLogger
- ProgressBar
- Trainer (orchestrates the training process)
"""

import os
import time
import logging
import numpy as np
from typing import Callable, List, Optional, Dict, Any

from .optimizer import Optimizer

# ----------------------------------
# Callback mechanism
# ----------------------------------
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
        monitor: metric name to track (e.g. 'loss', 'val_loss').
        patience: number of epochs with no improvement after which training will be stopped.
        min_delta: minimum change to qualify as improvement.
        mode: 'min' or 'max' depending on whether lower or higher is better.
    """
    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
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
        improvement = (current < self.best - self.min_delta) if self.mode == 'min' else (current > self.best + self.min_delta)
        if improvement:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs['stop_training'] = True


class ModelCheckpoint(Callback):
    """
    Save model parameters after every epoch or when a monitored metric improves.

    Args:
        filepath: path template, e.g. 'checkpoints/epoch_{epoch:02d}.npz'
        monitor: metric name to track
        save_best_only: if True, only save when monitored metric improves
        mode: 'min' or 'max'
    """
    def __init__(
        self,
        filepath: str,
        monitor: str = 'loss',
        save_best_only: bool = False,
        mode: str = 'min'
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best: Optional[float] = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if self.save_best_only and current is None:
            return
        if not self.save_best_only or self._is_improvement(current):
            path = self.filepath.format(epoch=epoch, **logs)
            params = logs.get('params')
            if params is not None:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(path, params=params)
                logging.info(f"ModelCheckpoint: saved model at {path}")

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return True
        is_better = (current < self.best) if self.mode == 'min' else (current > self.best)
        if is_better:
            self.best = current
        return is_better


class CSVLogger(Callback):
    """
    Log metrics to a CSV file.

    Args:
        filename: CSV file path
        fields: list of field names (columns) to log
    """
    def __init__(self, filename: str, fields: List[str]):
        self.filename = filename
        self.fields = fields
        self.file = open(self.filename, 'w', buffering=1)
        header = ','.join(fields)
        self.file.write(header + '\n')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        row = [str(logs.get(f, '')) for f in self.fields]
        self.file.write(','.join(row) + '\n')

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        self.file.close()


class ProgressBar(Callback):
    """
    Simple console-based progress indicator.

    Args:
        total_epochs: number of epochs
        steps_per_epoch: number of steps (batches) per epoch
    """
    def __init__(self, total_epochs: int, steps_per_epoch: int = 1):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        print(f"Epoch {epoch+1}/{self.total_epochs}")

    def on_step_end(self, epoch: int, step: int, logs: Dict[str, Any]) -> None:
        print(f"\r [Step {step+1}/{self.steps_per_epoch}] loss={logs.get('loss', 0):.6f}", end='')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        print()


# ----------------------------------
# Trainer
# ----------------------------------
class Trainer:
    """
    Trainer for variational quantum models.

    Args:
        model_fn: callable(params: np.ndarray) -> Any
        initial_params: np.ndarray initial parameter vector
        loss_fn: callable(output: Any) -> float, can wrap model_fn for custom metrics
        optimizer: instance of Optimizer from train/optimizer.py
        gradient_fn: callable(params: np.ndarray) -> np.ndarray gradients
        max_epochs: maximum number of epochs
        tol: tolerance for early stopping based on parameter change
        callbacks: list of Callback instances
    """
    def __init__(
        self,
        model_fn: Callable[[np.ndarray], Any],
        initial_params: np.ndarray,
        loss_fn: Callable[[Any], float],
        optimizer: Optimizer,
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        max_epochs: int = 100,
        tol: float = 1e-6,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model_fn = model_fn
        self.params = initial_params.astype(float)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.gradient_fn = gradient_fn
        self.max_epochs = max_epochs
        self.tol = tol
        self.callbacks = callbacks or []

    def fit(self) -> np.ndarray:
        logs: Dict[str, Any] = {}
        for cb in self.callbacks:
            cb.on_train_begin(logs)

        prev_params = self.params.copy()
        for epoch in range(self.max_epochs):
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch, logs)

            # forward
            output = self.model_fn(self.params)
            loss = self.loss_fn(output)
            logs['loss'] = loss
            logs['params'] = self.params.copy()

            # backward
            grads = self.gradient_fn(self.params)
            self.params = self.optimizer.step(self.params, grads)

            # callbacks step
            for cb in self.callbacks:
                cb.on_step_end(epoch, 0, logs)

            # epoch end
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

            # check early stopping flag
            if logs.get('stop_training'):
                logging.info(f"Stopping at epoch {epoch} by callback request")
                break

            # parameter change tolerance
            if np.linalg.norm(self.params - prev_params) < self.tol:
                logging.info(f"Parameter change below tol={self.tol}, stopping at epoch {epoch}")
                break

            prev_params = self.params.copy()

        for cb in self.callbacks:
            cb.on_train_end(logs)
        return self.params

