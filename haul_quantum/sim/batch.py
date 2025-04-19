"""
haul_quantum.sim.batch
======================
Batch (shot-based) simulator with noise integration.

Runs multiple trajectories of a QuantumCircuit under a given NoiseModel,
then measures the final state in the computational basis to build a histogram.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Sequence

from .statevector import StatevectorSimulator
from .noise import NoiseModel
from ..core.circuit import QuantumCircuit
from ..core.gates import Gate

class BatchSimulator:
    def __init__(self, n_qubits: int, seed: int | None = None) -> None:
        """
        :param n_qubits: number of qubits in the circuit
        :param seed: optional RNG seed for reproducibility
        """
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits
        self.sim = StatevectorSimulator(n_qubits)
        self.noise = NoiseModel(n_qubits, seed)
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        noise_params: Dict[str, Dict[int, float]] | None = None
    ) -> Dict[str, int]:
        """
        Execute `shots` trajectories of `circuit` with optional noise, then measure.

        :param circuit: QuantumCircuit to simulate
        :param shots: number of repeated runs
        :param noise_params:
            mapping from noise-method name (e.g. "depolarizing", "bit_flip")
            → dict mapping qubit index → probability.
            e.g. {"depolarizing": {0:0.01, 1:0.02}, "bit_flip": {0:0.005}}
        :returns: histogram mapping bit-string → count
        """
        hist: Dict[str,int] = {}

        for _ in range(shots):
            # start in |0…0>
            state = self.sim.zero_state()

            # apply each gate + noise
            for gate, qubits in circuit.instructions:
                state = self.sim.apply_gate(state, gate, qubits)

                if noise_params:
                    # for each noise channel
                    for channel_name, p_map in noise_params.items():
                        channel = getattr(self.noise, channel_name, None)
                        if channel is None:
                            raise ValueError(f"Unknown noise channel '{channel_name}'")
                        # apply to each qubit that has a p configured
                        for q, p in p_map.items():
                            if q in qubits and p > 0:
                                state = channel(state, p, q)

            # measure once
            probs = np.abs(state)**2
            idx = self.rng.choice(len(probs), p=probs)
            bitstr = format(idx, f"0{self.n_qubits}b")
            hist[bitstr] = hist.get(bitstr, 0) + 1

        return hist

