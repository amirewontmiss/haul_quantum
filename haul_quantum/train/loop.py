"""
Generic training loop with tqdm progress bar.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np
from tqdm.auto import trange

from .callbacks import Callback
from .optimizer import Optimizer


class Trainer:
    def __init__(
        self,
        model_fn: Callable[[np.ndarray], Any],
        initial_params: np.ndarray,
        loss_fn: Callable[[Any], float],
        optimizer: Optimizer,
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        max_epochs: int = 100,
        tol: float = 1e-4,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model_fn = model_fn
        self.params = initial_params.copy()
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.grad_fn = gradient_fn
        self.max_epochs = max_epochs
        self.tol = tol
        self.callbacks = callbacks or []

    def fit(self) -> np.ndarray:
        p = self.params
        for epoch in trange(self.max_epochs, desc="Training", unit="epoch"):
            grads = self.grad_fn(p)
            p_next = self.opt.step(p, grads)
            loss = self.loss_fn(self.model_fn(p_next))

            for cb in self.callbacks:
                cb(epoch=epoch, params=p_next, loss=loss)

            if abs(self.loss_fn(self.model_fn(p)) - loss) < self.tol:
                break
            p = p_next
        self.params = p
        return p
