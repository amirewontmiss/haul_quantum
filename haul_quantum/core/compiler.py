"""
haul_quantum.core.compiler
===========================
Compile a QuantumCircuit into OpenQASM or a JSON intermediate representation.

Currently supports:
 - Declaring quantum and classical registers.
 - 1-qubit gates: x, y, z, h, rx, ry, rz, s, t
 - 2-qubit gates: cx, cz, swap
 - Barrier insertion (optional)

Unrecognized gates will raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any, Dict

from .circuit import QuantumCircuit
from .gates import Gate


class CircuitCompiler:
    @staticmethod
    def to_qasm(
        circuit: QuantumCircuit, qreg_name: str = "q", creg_name: str = "c"
    ) -> str:
        n = circuit.n_qubits
        lines = [
            f"OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg {qreg_name}[{n}];",
            f"creg {creg_name}[{n}];",
        ]

        for gate, qubits in circuit.instructions:
            name = gate.name.lower()
            if name in {"x", "y", "z", "h", "s", "t"} and gate.num_qubits == 1:
                lines.append(f"{name} {qreg_name}[{qubits[0]}];")
            elif name in {"rx", "ry", "rz"} and gate.num_qubits == 1:
                theta = gate.params[0]
                lines.append(f"{name}({theta}) {qreg_name}[{qubits[0]}];")
            elif name == "cnot" or name == "cx":
                lines.append(f"cx {qreg_name}[{qubits[0]}],{qreg_name}[{qubits[1]}];")
            elif name == "cz":
                lines.append(f"cz {qreg_name}[{qubits[0]}],{qreg_name}[{qubits[1]}];")
            elif name == "swap":
                lines.append(f"swap {qreg_name}[{qubits[0]}],{qreg_name}[{qubits[1]}];")
            else:
                raise NotImplementedError(
                    f"QASM export for gate '{gate.name}' not implemented"
                )
        return "\n".join(lines)

    @staticmethod
    def to_dict(circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        JSON-serializable dict of the circuit:
        {
            "n_qubits": int,
            "instructions": [
                {"gate": str, "params": [...], "qubits": [q0, q1, ...]},
                ...
            ]
        }
        """
        inst_list = []
        for gate, qubits in circuit.instructions:
            inst_list.append(
                {
                    "gate": gate.name,
                    "params": list(gate.params),
                    "qubits": list(qubits),
                }
            )
        return {
            "n_qubits": circuit.n_qubits,
            "instructions": inst_list,
        }
