"""
haul_quantum.core.engine
============================
High-level engine to orchestrate circuits, simulation, noise, batch runs, and compilation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..sim.batch import BatchSimulator
from ..sim.noise import NoiseModel
from ..sim.statevector import StatevectorSimulator
from .circuit import QuantumCircuit
from .compiler import CircuitCompiler


class Engine:
    def __init__(self, n_qubits: int, seed: Optional[int] = None) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.simulator = StatevectorSimulator(n_qubits)
        self.noise_model = NoiseModel(n_qubits, seed)
        self.batch_sim = BatchSimulator(n_qubits, seed)

    # Chainable builder API
    def add(self, gate: Any, *qubits: int) -> Engine:
        self.circuit.add(gate, *qubits)
        return self

    def x(self, q: int) -> Engine:
        from .gates import X

        return self.add(X(), q)

    def y(self, q: int) -> Engine:
        from .gates import Y

        return self.add(Y(), q)

    def z(self, q: int) -> Engine:
        from .gates import Z

        return self.add(Z(), q)

    def h(self, q: int) -> Engine:
        from .gates import H

        return self.add(H(), q)

    def cnot(self, ctrl: int, tgt: int) -> Engine:
        from .gates import CNOT

        return self.add(CNOT(), ctrl, tgt)

    # Pure statevector simulation
    def simulate(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        return self.simulator.simulate(self.circuit.instructions, initial_state)

    # Single-shot simulation with inline noise
    def simulate_with_noise(
        self,
        initial_state: Optional[np.ndarray] = None,
        noise_params: Dict[str, Dict[int, float]] | None = None,
    ) -> np.ndarray:
        state = (
            initial_state.astype(complex, copy=True)
            if initial_state is not None
            else self.simulator.zero_state()
        )
        for gate, qubits in self.circuit.instructions:
            state = self.simulator.apply_gate(state, gate, qubits)
            if noise_params:
                for channel, p_map in noise_params.items():
                    func = getattr(self.noise_model, channel, None)
                    if func:
                        for q, p in p_map.items():
                            if q in qubits:
                                state = func(state, p, q)
        return state

    # Shot-based batch runs
    def run(
        self, shots: int = 1024, noise_params: Dict[str, Dict[int, float]] | None = None
    ) -> Dict[str, int]:
        return self.batch_sim.run(self.circuit, shots, noise_params)

    # Export circuits
    def to_qasm(self, qreg: str = "q", creg: str = "c") -> str:
        return CircuitCompiler.to_qasm(self.circuit, qreg, creg)

    def to_dict(self) -> Dict[str, Any]:
        return CircuitCompiler.to_dict(self.circuit)

    # Utility
    def reset(self) -> Engine:
        self.circuit = QuantumCircuit(self.n_qubits)
        return self
