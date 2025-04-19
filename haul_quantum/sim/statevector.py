"""
Statevector simulator (dense, CPU).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.gates import Gate


class StatevectorSimulator:
    def __init__(self, n_qubits: int, seed: int | None = None):
        self.n_qubits = n_qubits
        self.rng_seed = seed

    # ── helpers ── #
    def zero_state(self) -> NDArray[np.complex128]:
        state = np.zeros(2**self.n_qubits, dtype=complex)
        state[0] = 1.0
        return state

    def apply_gate(
        self,
        state: NDArray[np.complex128],
        gate: Gate,
        qubits: tuple[int, ...],
    ) -> NDArray[np.complex128]:
        """Apply 1- or contiguous 2-qubit *gate* (qubit-0 = LSB)."""
        if gate.num_qubits == 1:
            mats = [np.eye(2) for _ in range(self.n_qubits)]
            mats[qubits[0]] = gate.matrix
            full = mats[-1]
            for m in reversed(mats[:-1]):
                full = np.kron(full, m)
            return full @ state

        q_low, q_high = min(qubits), max(qubits)
        if q_high != q_low + 1:
            raise NotImplementedError("non-contiguous 2-qubit gates not supported")

        mats = [np.eye(2) for _ in range(self.n_qubits)]
        mats[q_high] = gate.matrix.reshape(2, 2, 2, 2)
        mats[q_low] = None  # marker

        full = None
        for idx in reversed(range(self.n_qubits)):
            if mats[idx] is None:
                continue
            block = mats[idx] if mats[idx].shape == (2, 2) else mats[idx].reshape(4, 4)
            full = block if full is None else np.kron(full, block)
        return full @ state

    # ── public API ── #
    def simulate(self, instructions):
        state = self.zero_state()
        for gate, qubits in instructions:
            state = self.apply_gate(state, gate, qubits)
        return state
