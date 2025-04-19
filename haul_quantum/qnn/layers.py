"""
haul_quantum.qnn.layers
=======================
Variational Quantum Neural Network (QNN) layer definitions.

Provides:
- VQCLayer: parameterized ansatz with alternating rotations and entanglers.
- (Stubs for QConv, QLSTM, etc. to expand later.)
"""

from __future__ import annotations
from typing import Sequence

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gates import RX, RY, RZ, CNOT

class VQCLayer:
    """
    Variational Quantum Circuit layer.

    Ansatz: for each layer
        - apply RX, RY, RZ on each qubit
        - apply a ring of CNOTs entangling qubits [0→1→2→…→0]

    Total parameters = n_layers * n_qubits * 3
    """

    def __init__(self, n_qubits: int, n_layers: int):
        if n_qubits < 1 or n_layers < 1:
            raise ValueError("n_qubits and n_layers must be >=1")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_parameters = n_layers * n_qubits * 3

    def build_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        """
        Turn a flat parameter vector into a QuantumCircuit.

        :param params: length must equal self.num_parameters
        """
        if len(params) != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} params, got {len(params)}")

        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            # single-qubit rotations
            for q in range(self.n_qubits):
                qc.add(RX(params[idx]), q);   idx += 1
                qc.add(RY(params[idx]), q);   idx += 1
                qc.add(RZ(params[idx]), q);   idx += 1
            # entangling ring
            for q in range(self.n_qubits - 1):
                qc.cnot(q, q + 1)
            # connect last back to first
            qc.cnot(self.n_qubits - 1, 0)
        return qc

    def forward(self, params: Sequence[float], initial_state: np.ndarray | None = None) -> np.ndarray:
        """
        Simulate the circuit produced by `build_circuit` and return final statevector.

        :param params: variational parameters
        :param initial_state: optional custom statevector
        """
        qc = self.build_circuit(params)
        return qc.simulate(initial_state)

