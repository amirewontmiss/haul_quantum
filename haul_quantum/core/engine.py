"""
Fluent, user-facing engine that wraps QuantumCircuit.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .circuit import QuantumCircuit
from .gates import CNOT, H, I, X


class Engine:
    """Chainable circuit builder with quick helpers."""

    def __init__(self, n_qubits: int, seed: int | None = None):
        self._circ = QuantumCircuit(n_qubits, seed=seed)

    # ── helpers ── #
    def h(self, q: int) -> "Engine":
        self._circ.add(H(), q)
        return self

    def x(self, q: int) -> "Engine":
        self._circ.add(X(), q)
        return self

    def cnot(self, ctrl: int, tgt: int) -> "Engine":
        msb, lsb = max(ctrl, tgt), min(ctrl, tgt)
        self._circ.add(CNOT(), msb, lsb)
        return self

    def i(self, q: int) -> "Engine":
        self._circ.add(I(), q)
        return self

    # ── execution ── #
    def simulate(self):
        return self._circ.simulate()

    def measure(self) -> Dict[str, float]:
        """Return computational-basis probabilities {bitstring: prob}."""
        state = self.simulate()
        probs = np.abs(state) ** 2
        bitstrings = [
            format(i, f"0{self._circ.n_qubits}b")[::-1] for i in range(len(probs))
        ]
        return {b: float(p) for b, p in zip(bitstrings, probs) if p > 1e-12}

    # ── utilities ── #
    def to_qasm(self) -> str:
        from .compiler import to_qasm

        return to_qasm(self._circ)

    def reset(self) -> "Engine":
        """Clear current circuit while preserving qubit count & seed."""
        self._circ = QuantumCircuit(self._circ.n_qubits, seed=self._circ.rng_seed)
        return self
