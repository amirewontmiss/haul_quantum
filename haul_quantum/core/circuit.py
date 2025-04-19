"""
haul_quantum.core.circuit
=========================
Minimal quantum circuit class with state-vector simulation.

Conventions:
* Qubit 0 is the **least** significant bit (rightmost).
* Statevector ordering: |q_{n-1} … q_0>.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple, Optional

import numpy as np

from .gates import Gate, CNOT

__all__ = ["QuantumCircuit"]


class QuantumCircuit:
    """Chainable circuit builder + simple CPU simulator."""

    # ----------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------
    def __init__(self, n_qubits: int) -> None:
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        self.n_qubits: int = n_qubits
        self.instructions: List[Tuple[Gate, Sequence[int]]] = []

    # ----------------------------------------------------------------------
    # Builder helpers
    # ----------------------------------------------------------------------
    def add(self, gate: Gate, *qubits: int) -> "QuantumCircuit":
        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"{gate.name} expects {gate.num_qubits} qubits; got {len(qubits)}"
            )
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise IndexError(f"Qubit {q} out of range 0…{self.n_qubits - 1}")
        self.instructions.append((gate, qubits))
        return self  # enable chaining

    # one-liners for common gates
    def x(self, q: int):          from .gates import X;  return self.add(X(),  q)
    def y(self, q: int):          from .gates import Y;  return self.add(Y(),  q)
    def z(self, q: int):          from .gates import Z;  return self.add(Z(),  q)
    def h(self, q: int):          from .gates import H;  return self.add(H(),  q)
    def cnot(self, c: int, t: int):                         return self.add(CNOT(), c, t)

    # ----------------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------------
    def simulate(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Return final state-vector after applying all gates."""
        dim = 2 ** self.n_qubits
        state = (
            initial_state.astype(complex, copy=True)
            if initial_state is not None
            else np.zeros(dim, dtype=complex)
        )
        if initial_state is None:
            state[0] = 1.0  # |0…0>

        for gate, qubits in self.instructions:
            state = self._apply_gate(state, gate, qubits)
        return state

    # ----------------------------------------------------------------------
    # Internal apply helpers
    # ----------------------------------------------------------------------
    def _apply_gate(
        self, state: np.ndarray, gate: Gate, qubits: Sequence[int]
    ) -> np.ndarray:
        if gate.num_qubits == 1:
            return self._apply_single(state, gate.matrix, qubits[0])
        if gate.num_qubits == 2:
            return self._apply_two(state, gate.matrix, qubits)
        raise NotImplementedError(">2-qubit gates not yet supported")

    def _apply_single(
        self, state: np.ndarray, mat: np.ndarray, q: int
    ) -> np.ndarray:
        op = 1.0
        # tensor product from MSB to LSB
        for i in range(self.n_qubits - 1, -1, -1):
            op = np.kron(op, mat if i == q else np.eye(2, dtype=complex))
        return op @ state

    def _apply_two(
        self, state: np.ndarray, mat: np.ndarray, qubits: Sequence[int]
    ) -> np.ndarray:
        q0, q1 = qubits
        if q0 == q1:
            raise ValueError("Control and target must differ")
        n = self.n_qubits

        # reshape to rank-n tensor
        tensor = state.reshape((2,) * n)

        # move affected qubits to the end
        axes = list(range(n))
        axes.remove(q0); axes.remove(q1)
        axes.extend([q0, q1])

        tensor = np.transpose(tensor, axes).reshape(-1, 4)
        tensor = tensor @ mat.T
        tensor = tensor.reshape((2,) * n)

        # restore original order
        inverse_axes = np.argsort(axes)
        return np.transpose(tensor, inverse_axes).reshape(-1)

    # ----------------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        header = f"QuantumCircuit(n_qubits={self.n_qubits}, depth={len(self.instructions)})"
        lines = [header]
        for idx, (gate, qs) in enumerate(self.instructions):
            lines.append(f"  {idx:>3}: {gate} @ {qs}")
        return "\n".join(lines)

