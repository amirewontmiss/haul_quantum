"""
Light-weight quantum-circuit container.

* Stores a list of (Gate, qubit-tuple) instructions.
* Provides chainable helpers (h, x, cnot, i) for convenience.
* Assumes qubit 0 is the least-significant bit (LSB) in basis ordering.
"""

from __future__ import annotations

from typing import List, Tuple

from .gates import CNOT, Gate, H, I, X


class QuantumCircuit:
    """In-memory circuit representation."""

    def __init__(self, n_qubits: int, *, seed: int | None = None):
        self.n_qubits = n_qubits
        self.instructions: List[Tuple[Gate, Tuple[int, ...]]] = []
        self.rng_seed = seed  # reserved for future RNG uses

    # ── low-level API ── #
    def add(self, gate: Gate, *qubits: int) -> None:
        if len(qubits) != gate.num_qubits:
            raise ValueError("Incorrect qubit count for gate")
        self.instructions.append((gate, tuple(qubits)))

    # ── simulation helper ── #
    def simulate(self):
        """Return statevector via StatevectorSimulator (dense CPU)."""
        from ..sim.statevector import StatevectorSimulator

        return StatevectorSimulator(self.n_qubits).simulate(self.instructions)

    # ── chainable helpers ── #
    def h(self, q: int) -> "QuantumCircuit":
        self.add(H(), q)
        return self

    def x(self, q: int) -> "QuantumCircuit":
        self.add(X(), q)
        return self

    def cnot(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        """Add CNOT with *ctrl* control and *tgt* target (LSB ordering)."""
        msb, lsb = max(ctrl, tgt), min(ctrl, tgt)
        self.add(CNOT(), msb, lsb)
        return self

    def i(self, q: int) -> "QuantumCircuit":
        self.add(I(), q)
        return self
