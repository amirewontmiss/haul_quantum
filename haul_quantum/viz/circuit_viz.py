"""
haul_quantum.viz.circuit_viz
=============================
Visualization tools for QuantumCircuit objects.

Functions:
- draw_ascii(circuit): ASCII-art representation.
- plot_circuit(circuit): Matplotlib wire- and gate-diagram.
"""

from __future__ import annotations
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from ..core.circuit import QuantumCircuit

def draw_ascii(circuit: QuantumCircuit) -> str:
    """
    Return an ASCII art depiction of the circuit.

    Each qubit is a row; each time-step is a column.
    """
    n = circuit.n_qubits
    inst = circuit.instructions
    # prepare rows
    rows = [f"q{q}: " for q in range(n)]
    # build columns
    for gate, qubits in inst:
        name = gate.name.center(3)
        for q in range(n):
            if q in qubits:
                rows[q] += f"[{name}]-"
            else:
                rows[q] += " --- -"
    return "\n".join(rows)

def plot_circuit(
    circuit: QuantumCircuit,
    figsize: Tuple[float, float] = (6, 2)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a simple circuit diagram with wires and gate boxes.

    Returns the Matplotlib Figure and Axes.
    """
    n = circuit.n_qubits
    inst = circuit.instructions
    fig, ax = plt.subplots(figsize=figsize)
    # horizontal wires
    for q in range(n):
        y = n - 1 - q
        ax.hlines(y, xmin=-0.5, xmax=len(inst) - 0.5, color="black")
    # gates
    for idx, (gate, qubits) in enumerate(inst):
        for q in qubits:
            y = n - 1 - q
            # draw gate box
            rect = plt.Rectangle((idx - 0.3, y - 0.3), 0.6, 0.6,
                                 facecolor="white", edgecolor="black")
            ax.add_patch(rect)
            ax.text(idx, y, gate.name, ha="center", va="center")
        # draw control-target lines for 2-qubit gates
        if gate.num_qubits == 2:
            q0, q1 = qubits
            y0 = n - 1 - q0
            y1 = n - 1 - q1
            ax.vlines(x=idx, ymin=min(y0, y1), ymax=max(y0, y1),
                      color="black", linestyle="dotted")
            ax.plot(idx, y0, 'o', color="black")
            ax.plot(idx, y1, 'o', color="black")
    ax.set_xlim(-0.5, len(inst) - 0.5)
    ax.set_ylim(-1, n)
    ax.axis("off")
    return fig, ax

