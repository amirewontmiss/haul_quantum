"""
haul_quantum.qnn.torch_integration
===================================
PyTorch integration for Haul Quantum AI framework.
Provides:
  - QNodeFunction: custom autograd.Function implementing the parameter-shift rule
  - TorchQuantumLayer: a torch.nn.Module wrapping a VQCLayer,
    producing ⟨Z⟩ expectation values as a differentiable layer
"""

import torch
from torch.autograd import Function
import numpy as np
from typing import Callable

from ..core.engine import Engine
from .layers import VQCLayer


class QNodeFunction(Function):
    """
    Autograd Function implementing the parameter-shift rule.

    forward:
      inputs: 
        - params: torch.Tensor of shape (num_parameters,)
        - circuit_fn: Callable[[Engine, np.ndarray], np.ndarray]
        - engine: Engine
      returns: torch.Tensor of expectation values

    backward:
      uses f(θ+π/2) and f(θ−π/2) to compute gradients
    """

    @staticmethod
    def forward(ctx, params: torch.Tensor, circuit_fn: Callable, engine: Engine):
        # save for backward
        ctx.circuit_fn = circuit_fn
        ctx.engine = engine

        # detach to numpy
        params_np = params.detach().cpu().numpy()
        # compute expectation values
        expvals = circuit_fn(engine, params_np)
        # tensorify
        exp_tensor = torch.from_numpy(expvals).to(params.device).to(params.dtype)

        # save state
        ctx.params = params_np
        ctx.exp_shape = expvals.shape

        return exp_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        circuit_fn = ctx.circuit_fn
        engine = ctx.engine
        orig = ctx.params.copy()
        num_p = orig.size
        shift = np.pi / 2

        # compute f(θ+shift) and f(θ-shift)
        grads = np.zeros((num_p, *ctx.exp_shape), dtype=float)
        for i in range(num_p):
            plus = orig.copy();  plus[i] += shift
            minus = orig.copy(); minus[i] -= shift
            f_plus = circuit_fn(engine, plus)
            f_minus = circuit_fn(engine, minus)
            grads[i] = 0.5 * (f_plus - f_minus)

        # chain rule with incoming gradient
        grad_out_np = grad_output.detach().cpu().numpy()
        # if exp_shape is vector, tensordot over axis 1
        grad_params = np.tensordot(grads, grad_out_np, axes=([1], [0])) \
            if grads.ndim > 1 else grads * grad_out_np

        # back to torch
        grad_params_tensor = torch.from_numpy(grad_params).to(grad_output.device).to(grad_output.dtype)
        # no grads for circuit_fn or engine
        return grad_params_tensor, None, None


class TorchQuantumLayer(torch.nn.Module):
    """
    High-level torch.nn.Module wrapping a VQCLayer.
    Returns ⟨Z⟩ expectation values on each qubit.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.vqc = VQCLayer(n_qubits, n_layers)
        self.engine = Engine(n_qubits)
        # initialize trainable parameters
        self.params = torch.nn.Parameter(
            torch.randn(self.vqc.num_parameters, dtype=torch.float32) * 0.01
        )

    def forward(self) -> torch.Tensor:
        # closure converting params → expectation values
        def circuit_fn(engine: Engine, params_np: np.ndarray) -> np.ndarray:
            # build and simulate VQC
            qc = self.vqc.build_circuit(params_np)
            state = qc.simulate()
            # measure ⟨Z⟩ on each qubit
            probs = np.abs(state) ** 2
            exps = np.array([
                sum(((1 - 2 * ((idx >> q) & 1)) * p) for idx, p in enumerate(probs))
                for q in range(self.n_qubits)
            ], dtype=float)
            return exps

        # call custom autograd function
        return QNodeFunction.apply(self.params, circuit_fn, self.engine)

