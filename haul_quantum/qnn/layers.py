from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gates import CNOT, RX, RY, RZ


class VQCLayer:
    """Simple layer: RX → RY → CNOT ladder."""

    def __init__(self, n_qubits: int, n_layers: int = 1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_parameters = n_layers * n_qubits * 2  # rx + ry per qubit

    def build_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.add(RX(params[idx]), q)
                idx += 1
                qc.add(RY(params[idx]), q)
                idx += 1
                qc.add(RZ(np.pi / 4), q)  # RZ now used
            for q in range(self.n_qubits - 1):
                qc.add(CNOT(), q, q + 1)
        return qc
