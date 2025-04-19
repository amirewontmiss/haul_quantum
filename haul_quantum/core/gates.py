"""
Gate primitives and parametric rotations.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "X",
    "Y",
    "Z",
    "H",
    "CNOT",
    "I",
    "RX",
    "RY",
    "RZ",
]


# --------------------------------------------------------------------------- #
# Gate container
# --------------------------------------------------------------------------- #
class Gate:
    """Light-weight gate container."""

    def __init__(self, name: str, matrix: np.ndarray, num_qubits: int):
        self.name = name
        self.matrix = matrix
        self.num_qubits = num_qubits
        self.params: list[float] | None = None


# --------------------------------------------------------------------------- #
# Static single-qubit gates
# --------------------------------------------------------------------------- #
def I() -> Gate:  # noqa: E743
    return Gate("I", np.eye(2, dtype=complex), 1)


def X() -> Gate:
    return Gate("X", np.array([[0, 1], [1, 0]], complex), 1)


def Y() -> Gate:
    return Gate("Y", np.array([[0, -1j], [1j, 0]], complex), 1)


def Z() -> Gate:
    return Gate("Z", np.array([[1, 0], [0, -1]], complex), 1)


def H() -> Gate:
    return Gate(
        "H",
        (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], complex),
        1,
    )


# --------------------------------------------------------------------------- #
# Parametric rotations
# --------------------------------------------------------------------------- #
def _rot(axis: str, theta: float) -> np.ndarray:
    ct = math.cos(theta / 2)

    if axis == "X":
        st = -1j * math.sin(theta / 2)
        return np.array([[ct, st], [st, ct]], complex)

    if axis == "Y":
        st = math.sin(theta / 2)
        return np.array([[ct, st], [-st, ct]], complex)

    # Z
    phase = math.e ** (1j * theta / 2)
    return np.array([[phase.conjugate(), 0], [0, phase]], complex)


def RX(theta: float) -> Gate:
    g = Gate(f"RX({theta:.3f})", _rot("X", theta), 1)
    g.params = [theta]
    return g


def RY(theta: float) -> Gate:
    g = Gate(f"RY({theta:.3f})", _rot("Y", theta), 1)
    g.params = [theta]
    return g


def RZ(theta: float) -> Gate:
    g = Gate(f"RZ({theta:.3f})", _rot("Z", theta), 1)
    g.params = [theta]
    return g


# --------------------------------------------------------------------------- #
# Two-qubit gate (control q1, target q0)
# --------------------------------------------------------------------------- #
def CNOT() -> Gate:
    """
    CNOT with control qubit 1 (MSB) and target qubit 0 (LSB).

    Basis order |q1 q0⟩ = |00>, |01>, |10>, |11>.
    """
    return Gate(
        "CNOT",
        np.array(
            [
                [1, 0, 0, 0],  # |00>
                [0, 1, 0, 0],  # |01>
                [0, 0, 0, 1],  # |10> → |11>
                [0, 0, 1, 0],  # |11> → |10>
            ],
            complex,
        ),
        2,
    )
