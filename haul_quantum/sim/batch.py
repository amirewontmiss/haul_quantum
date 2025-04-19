"""
Shot-based sampling simulator.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.circuit import QuantumCircuit
from ..sim.noise import NoiseModel
from ..sim.statevector import StatevectorSimulator


class BatchSimulator:
    def __init__(self, n_qubits: int, seed: int | None = None):
        self.sv_sim = StatevectorSimulator(n_qubits, seed=seed)
        self.noise = NoiseModel(n_qubits, seed=seed)

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        state = self.sv_sim.simulate(circuit.instructions)
        probs = np.abs(state) ** 2
        outcomes = np.random.default_rng().choice(len(probs), size=shots, p=probs)
        hist: Dict[str, int] = {}
        for idx in outcomes:
            bitstring = format(idx, f"0{circuit.n_qubits}b")[::-1]
            hist[bitstring] = hist.get(bitstring, 0) + 1
        return hist
