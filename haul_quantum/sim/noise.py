"""
haul_quantum.sim.noise
======================
Monte Carlo noise channels for statevector simulation.

Provides basic quantum noise models:
- bit_flip: flips a target qubit with probability p.
- phase_flip: applies a phase-flip (Z) with probability p.
- depolarizing: applies X, Y, or Z uniformly with probability p.
- amplitude_damping: simulates single-qubit amplitude damping channel
  via stochastic quantum jump/no-jump sampling.

Note: This is trajectory-based Monte Carlo noise. Many runs average
over noise realizations to approximate density-matrix behavior.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Optional

from .statevector import StatevectorSimulator
from ..core.gates import Gate, X, Y, Z

class NoiseModel:
    def __init__(self, n_qubits: int, seed: Optional[int] = None) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits
        self.sim = StatevectorSimulator(n_qubits)
        self.rng = np.random.default_rng(seed)

    def bit_flip(self, state: np.ndarray, p: float, qubit: int) -> np.ndarray:
        """Apply bit-flip error on `qubit` with probability p."""
        if self.rng.random() < p:
            return self.sim.apply_gate(state, X(), [qubit])
        return state

    def phase_flip(self, state: np.ndarray, p: float, qubit: int) -> np.ndarray:
        """Apply phase-flip (Z) on `qubit` with probability p."""
        if self.rng.random() < p:
            return self.sim.apply_gate(state, Z(), [qubit])
        return state

    def depolarizing(self, state: np.ndarray, p: float, qubit: int) -> np.ndarray:
        """Apply depolarizing channel on `qubit` with probability p."""
        if self.rng.random() < p:
            error_gate = self.rng.choice([X(), Y(), Z()])
            return self.sim.apply_gate(state, error_gate, [qubit])
        return state

    def amplitude_damping(self, state: np.ndarray, gamma: float, qubit: int) -> np.ndarray:
        """Simulate single-qubit amplitude damping on `qubit` with strength gamma."""
        # Quantum-jump approach: jump with probability gamma
        if self.rng.random() < gamma:
            # jump: |1> -> |0>
            new_state = np.zeros_like(state)
            for idx, amp in enumerate(state):
                if ((idx >> qubit) & 1) == 1:
                    new_idx = idx & ~(1 << qubit)
                    new_state[new_idx] += amp
            norm = np.linalg.norm(new_state)
            if norm > 0:
                new_state /= norm
            return new_state
        else:
            # no-jump: apply non-unitary damping on |1> amplitude
            mat = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
            return self.sim.apply_gate(state, Gate("AD_no_jump", mat, num_qubits=1), [qubit])

