"""
haul_quantum.core.gates
=======================
Basic quantum gates and gate abstraction for the Haul Quantum AI framework.

Defines:
* Gate – immutable base class for any quantum gate.
* Standard 1-qubit gates: I, X, Y, Z, H, S, T, RX, RY, RZ.
* Standard 2-qubit gates: CNOT, CZ, SWAP.
"""

from __future__ import annotations
from math import cos, sin
from typing import Tuple

import numpy as np

__all__ = [
    "Gate",
    # single-qubit factories
    "I", "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ",
    # two-qubit factories
    "CNOT", "CZ", "SWAP",
]


class Gate:
    """Abstract, immutable gate object."""

    def __init__(
        self,
        name: str,
        matrix: np.ndarray,
        num_qubits: int = 1,
        params: Tuple[float, ...] | None = None,
    ) -> None:
        self.name = name
        self.matrix = np.asarray(matrix, dtype=complex)
        self.num_qubits = num_qubits
        self.params: Tuple[float, ...] = tuple(params) if params else tuple()

        # --- validation ---------------------------------------------------
        dim = 2 ** self.num_qubits
        if self.matrix.shape != (dim, dim):
            raise ValueError(
                f"Gate {name}: matrix shape must be {(dim, dim)}, got {self.matrix.shape}"
            )
        # Quick unitarity check (cheap heuristic).
        if not np.allclose(self.matrix.conj().T @ self.matrix, np.eye(dim), atol=1e-8):
            raise ValueError(f"Gate {name}: matrix is not unitary")

    # ----------------------------------------------------------------------
    # Representations
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        param_str = f"{self.params}" if self.params else ""
        return f"Gate<{self.name}{param_str}, qubits={self.num_qubits}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        return (
            self.name == other.name
            and self.num_qubits == other.num_qubits
            and np.allclose(self.matrix, other.matrix)
        )


# --------------------------------------------------------------------------
# Single-qubit standard gates
# --------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def I() -> Gate:   return Gate("I", _I.copy(), 1)
def X() -> Gate:   return Gate("X", _X.copy(), 1)
def Y() -> Gate:   return Gate("Y", _Y.copy(), 1)
def Z() -> Gate:   return Gate("Z", _Z.copy(), 1)
def H() -> Gate:   return Gate("H", _H.copy(), 1)
def S() -> Gate:   return Gate("S", _S.copy(), 1)
def T() -> Gate:   return Gate("T", _T.copy(), 1)

# --------------------------------------------------------------------------
# Parameterised rotation gates
# --------------------------------------------------------------------------
def RX(theta: float) -> Gate:
    c, s = cos(theta / 2), -1j * sin(theta / 2)
    return Gate("RX", np.array([[c, s], [s, c.conjugate()]], dtype=complex), 1, (theta,))


def RY(theta: float) -> Gate:
    c, s = cos(theta / 2), sin(theta / 2)
    return Gate("RY", np.array([[c, -s], [s, c]], dtype=complex), 1, (theta,))


def RZ(theta: float) -> Gate:
    e_pos = np.exp(-1j * theta / 2)
    e_neg = np.exp(1j * theta / 2)
    return Gate("RZ", np.array([[e_pos, 0], [0, e_neg]], dtype=complex), 1, (theta,))


# --------------------------------------------------------------------------
# Two-qubit gates
# --------------------------------------------------------------------------
_CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=complex,
)
_CZ   = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)

def CNOT() -> Gate: return Gate("CNOT", _CNOT.copy(), 2)
def CZ()   -> Gate: return Gate("CZ",   _CZ.copy(),   2)
def SWAP() -> Gate: return Gate("SWAP", _SWAP.copy(), 2)

