"""
Simple single-qubit noise channels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.gates import Gate, X, Z
from ..sim.statevector import StatevectorSimulator


class NoiseModel:
    def __init__(self, n_qubits: int, seed: int | None = None):
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)

    # ── channels ── #
    def bit_flip(
        self,
        state: np.ndarray,
        p: float,
        qubit: int,
        gate: Optional[Gate] = None,
    ) -> np.ndarray:
        if self.rng.random() < p:
            return StatevectorSimulator(self.n_qubits).apply_gate(state, X(), (qubit,))
        return state

    def phase_flip(
        self,
        state: np.ndarray,
        p: float,
        qubit: int,
        gate: Optional[Gate] = None,
    ) -> np.ndarray:
        if self.rng.random() < p:
            return StatevectorSimulator(self.n_qubits).apply_gate(state, Z(), (qubit,))
        return state

    def depolarizing(self, state: np.ndarray, p: float, qubit: int) -> np.ndarray:
        _ = (p, qubit)  # stub
        return state
