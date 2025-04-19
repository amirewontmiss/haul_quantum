"""
haul_quantum.datasets.qdata
============================
Quantum-native dataset generators.

Provides functions to generate common quantum state datasets:
- basis_states: computational basis circuits or statevectors.
- ghz_states: GHZ state circuits.
- random_pure_states: random Haar-distributed pure states.
- bell_states: standard Bell pair circuits.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

from ..core.circuit import QuantumCircuit
from ..core.gates import H, CNOT, X, Z

def basis_states(n_qubits: int) -> Tuple[List[QuantumCircuit], np.ndarray]:
    """
    Generate all computational basis states for n_qubits.

    Returns:
        circuits: list of QuantumCircuits preparing |0...0>, |0...1>, ..., |1...1>.
        statevectors: array of shape (2**n, 2**n) where each row is the statevector.
    """
    dim = 2 ** n_qubits
    circuits: List[QuantumCircuit] = []
    svs = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            if (idx >> q) & 1:
                qc.add(X(), q)
        circuits.append(qc)
        vec = np.zeros(dim, dtype=complex)
        vec[idx] = 1.0
        svs[idx] = vec
    return circuits, svs

def ghz_states(n_qubits: int) -> Tuple[List[QuantumCircuit], np.ndarray]:
    """
    Generate GHZ states for n_qubits.

    Returns:
        circuits: single-circuit list for GHZ state.
        statevectors: array of shape (1, 2**n_qubits) of GHZ statevector.
    """
    qc = QuantumCircuit(n_qubits)
    qc.add(H(), 0)
    for q in range(n_qubits - 1):
        qc.add(CNOT(), q, q + 1)
    dim = 2 ** n_qubits
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1 / np.sqrt(2)
    vec[-1] = 1 / np.sqrt(2)
    return [qc], np.array([vec])

def bell_states() -> Tuple[List[QuantumCircuit], np.ndarray]:
    """
    Generate the four Bell states for 2 qubits.

    Returns:
        circuits: list of 4 QuantumCircuits for |Φ+>, |Φ->, |Ψ+>, |Ψ->.
        statevectors: array of shape (4, 4) of Bell statevectors.
    """
    # |Φ+>
    qc_phi_plus = QuantumCircuit(2)
    qc_phi_plus.add(H(), 0).add(CNOT(), 0, 1)

    # |Φ->
    qc_phi_minus = QuantumCircuit(2)
    qc_phi_minus.add(H(), 0).add(CNOT(), 0, 1).add(Z(), 0)

    # |Ψ+>
    qc_psi_plus = QuantumCircuit(2)
    qc_psi_plus.add(X(), 1).add(H(), 0).add(CNOT(), 0, 1)

    # |Ψ->
    qc_psi_minus = QuantumCircuit(2)
    qc_psi_minus.add(X(), 1).add(H(), 0).add(CNOT(), 0, 1).add(Z(), 0)

    circuits = [qc_phi_plus, qc_phi_minus, qc_psi_plus, qc_psi_minus]
    svs = [qc.simulate() for qc in circuits]
    return circuits, np.vstack(svs)

def random_pure_states(
    n_qubits: int,
    n_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random Haar-distributed pure statevectors.

    Returns:
        statevectors: array of shape (n_samples, 2**n_qubits)
    """
    dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    svs = []
    for _ in range(n_samples):
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec /= np.linalg.norm(vec)
        svs.append(vec)
    return np.array(svs)

