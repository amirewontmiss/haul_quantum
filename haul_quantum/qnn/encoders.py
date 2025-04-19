"""
haul_quantum.qnn.encoders
=========================
Data encoding strategies for quantum circuits.

Provides:
- basis_encoding: maps classical bits → computational basis via X gates.
- angle_encoding: encodes real vectors via single-qubit rotations.
- amplitude_encoding: generates a normalized statevector from a real/complex vector.
"""

from __future__ import annotations
from typing import Sequence, Literal
import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gates import X, RX, RY, RZ

def basis_encoding(bits: Sequence[int]) -> QuantumCircuit:
    """
    Encode a bitstring into the computational basis.

    :param bits: sequence of 0/1 values, length = n_qubits
    :returns: QuantumCircuit that flips qubits where bit == 1
    """
    n = len(bits)
    qc = QuantumCircuit(n)
    for q, bit in enumerate(bits):
        if bit not in (0, 1):
            raise ValueError("basis_encoding expects bits (0 or 1)")
        if bit == 1:
            qc.add(X(), q)
    return qc

def angle_encoding(
    values: Sequence[float],
    rotation: Literal["X", "Y", "Z"] = "Y"
) -> QuantumCircuit:
    """
    Encode a real-valued vector into qubit rotations.

    :param values: sequence of floats, length = n_qubits
    :param rotation: axis of rotation ("X", "Y", or "Z")
    :returns: QuantumCircuit with one rotation per qubit
    """
    n = len(values)
    qc = QuantumCircuit(n)
    for q, theta in enumerate(values):
        if rotation == "X":
            qc.add(RX(theta), q)
        elif rotation == "Y":
            qc.add(RY(theta), q)
        elif rotation == "Z":
            qc.add(RZ(theta), q)
        else:
            raise ValueError("rotation must be 'X', 'Y', or 'Z'")
    return qc

def amplitude_encoding(state_vector: Sequence[complex]) -> np.ndarray:
    """
    Prepare a normalized statevector for amplitude encoding.

    Note: This returns the initial_state for simulation; you can run
          simulate(initial_state=amplitude_encoding(v)).

    :param state_vector: sequence of length 2**n representing amplitudes
    :returns: normalized NumPy array of shape (2**n,)
    """
    arr = np.array(state_vector, dtype=complex)
    if arr.ndim != 1 or (arr.size & (arr.size - 1)) != 0:
        raise ValueError("Length of state_vector must be a power of 2")
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Cannot encode zero vector")
    return arr / norm

