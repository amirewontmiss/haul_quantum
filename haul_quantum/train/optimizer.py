"""
haul_quantum.train.optimizer
============================
Optimizers for quantum-classical hybrid training.

Provides:
- Optimizer (abstract base class)
- GradientDescent: vanilla gradient descent
- AdamOptimizer: adaptive moment estimation
- SPSAOptimizer: gradient-free simultaneous perturbation (stub for later extension)
"""

from typing import Callable, Optional

import numpy as np


class Optimizer:
    """
    Abstract optimizer interface.

    Subclasses must implement:
      - step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray
    """

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Optimizer subclasses must implement step()")


class GradientDescent(Optimizer):
    """
    Vanilla gradient descent optimizer.
    Δθ = -lr * ∇θ L
    """

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self.lr * grads


class AdamOptimizer(Optimizer):
    """
    Adam optimizer: adaptive moment estimation.
    Reference: Kingma & Ba (2015).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            # Initialize moment vectors
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        # Update parameters
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SPSAOptimizer(Optimizer):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA).
    NOTE: This implementation expects a `loss_fn: Callable[[np.ndarray], float]`
    to be provided instead of gradients. In the Trainer, set `gradient_fn` to
    your loss function when using SPSAOptimizer.
    """

    def __init__(
        self,
        loss_fn: Callable[[np.ndarray], float],
        a: float = 0.1,
        c: float = 0.1,
        A: float = 10.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: Optional[int] = None,
    ):
        self.loss_fn = loss_fn
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.k = 0
        self.rng = np.random.default_rng(seed)

    def step(self, params: np.ndarray, grads: np.ndarray = None) -> np.ndarray:
        """
        Update parameters using SPSA gradient approximation:
          Δθ_k = a_k * ĝ_k
        where ĝ_k[i] ≈ [L(θ+ c_k Δ) - L(θ- c_k Δ)] / (2 c_k Δ[i])
        """
        self.k += 1
        ak = self.a / ((self.k + self.A) ** self.alpha)
        ck = self.c / (self.k**self.gamma)
        dim = params.size

        # Draw ±1 perturbation vector
        delta = self.rng.choice([-1, 1], size=dim)
        # Evaluate at perturbed points
        loss_plus = self.loss_fn(params + ck * delta)
        loss_minus = self.loss_fn(params - ck * delta)
        # Approximate gradient
        g_hat = (loss_plus - loss_minus) / (2.0 * ck * delta)
        # Parameter update
        return params - ak * g_hat


# Example usage
if __name__ == "__main__":
    # Suppose L(θ) = θ^2, gradient_fn = lambda θ: 2θ
    gd = GradientDescent(lr=0.1)
    adam = AdamOptimizer(lr=0.05)
    spsa = SPSAOptimizer(loss_fn=lambda x: np.sum(x**2), a=0.2, c=0.1)

    params = np.array([1.0, -0.5])
    grads = 2 * params  # True gradient

    print("GD step:", gd.step(params, grads))
    print("Adam step:", adam.step(params, grads))
    print("SPSA step:", spsa.step(params))
