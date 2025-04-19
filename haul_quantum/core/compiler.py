"""
OpenQASM 2.0 exporter.
"""

from __future__ import annotations

from math import pi
from typing import List, Optional

from .circuit import QuantumCircuit

_GATE_MAP = {"H": "h", "X": "x", "CNOT": "cx"}
_PARAMETRIC = {"RX", "RY", "RZ"}


def _format_gate(
    name: str,
    qubits: List[int],
    params: Optional[List[float]] = None,
) -> str:
    """Return a single QASM instruction line."""
    if name in _GATE_MAP:
        if name == "CNOT":  # two-qubit
            ctrl, tgt = qubits
            return f"cx q[{ctrl}],q[{tgt}];"
        return f"{_GATE_MAP[name]} q[{qubits[0]}];"

    base = name.split("(")[0]
    if base in _PARAMETRIC and params:
        theta = params[0] % (2 * pi)
        return f"u3({theta:.8f},0,0) q[{qubits[0]}];"

    raise ValueError(f"Unsupported gate {name!r}")


def to_qasm(circ: QuantumCircuit) -> str:
    """Convert *circ* to an OpenQASM-2 program string."""
    lines: list[str] = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circ.n_qubits}];",
    ]

    for gate, qubits in circ.instructions:
        lines.append(
            _format_gate(
                gate.name,
                list(qubits),
                getattr(gate, "params", None),
            )
        )

    return "\n".join(lines)
