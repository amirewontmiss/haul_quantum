"""
haul_quantum.sim.statevector
============================
Statevector simulator for Haul Quantum AI framework.

Provides:
 - StatevectorSimulator: initialize |0…0> or custom state
 - apply_gate: apply any Gate to the full statevector
 - simulate: run a list of (Gate, qubits) instructions
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple, Optional
from haul_quantum.core.gates import Gate

class StatevectorSimulator:
    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def zero_state(self) -> np.ndarray:
        """Return the |0…0> statevector."""
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def simulate(
        self,
        instructions: Sequence[Tuple[Gate, Sequence[int]]],
        initial_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply all gates in sequence and return the final statevector.

        :param instructions: list of (Gate, qubit indices) tuples
        :param initial_state: optional custom statevector (shape 2**n)
        """
        state = (
            initial_state.astype(complex, copy=True)
            if initial_state is not None
            else self.zero_state()
        )
        for gate, qubits in instructions:
            state = self.apply_gate(state, gate, qubits)
        return state

    def apply_gate(
        self,
        state: np.ndarray,
        gate: Gate,
        qubits: Sequence[int]
    ) -> np.ndarray:
        """Dispatch to single- or two-qubit apply routines."""
        if gate.num_qubits == 1:
            return self._apply_single(state, gate.matrix, qubits[0])
        elif gate.num_qubits == 2:
            return self._apply_two(state, gate.matrix, qubits)
        else:
            raise NotImplementedError(f"{gate.num_qubits}-qubit gates not supported")

    def _apply_single(
        self,
        state: np.ndarray,
        mat: np.ndarray,
        target: int
    ) -> np.ndarray:
        # Build full operator by kron’ing across all qubits
        op = 1.0
        for idx in range(self.n_qubits - 1, -1, -1):
            op = np.kron(op, mat if idx == target else np.eye(2, dtype=complex))
        return op @ state

    def _apply_two(
        self,
        state: np.ndarray,
        mat: np.ndarray,
        qubits: Sequence[int]
    ) -> np.ndarray:
        q0, q1 = qubits
        if q0 == q1:
            raise ValueError("Two-qubit gate requires distinct qubits")
        # reshape to rank-n tensor
        tensor = state.reshape((2,) * self.n_qubits)
        # bring our qubits to the end
        axes = [i for i in range(self.n_qubits) if i not in qubits] + list(qubits)
        tensor = np.transpose(tensor, axes).reshape(-1, 4)
        # apply gate on last two indices
        tensor = tensor @ mat.T
        # reshape back and invert transpose
        tensor = tensor.reshape((2,) * self.n_qubits)
        inverse_axes = np.argsort(axes)
        return np.transpose(tensor, inverse_axes).reshape(-1)

