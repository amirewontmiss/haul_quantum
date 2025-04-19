from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_vqc_architecture(
    n_qubits: int, n_layers: int, figsize: Tuple[float, float] = (4, 4)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a schematic of the VQC layer: rotation blocks and entangling rings.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # qubit positions
    y = np.linspace(0, 1, n_qubits)
    # layers horizontally
    x = np.linspace(0, 1, n_layers * 2 + 1)
    # plot wires
    for yi in y:
        ax.hlines(yi, xmin=0, xmax=1, color="gray")
    # plot rotation nodes and entanglers
    layer_idx = 0
    for L in range(n_layers):
        # rotations
        for qi, yi in enumerate(y):
            ax.scatter(
                x[layer_idx],
                yi,
                s=100,
                marker="o",
                label="Rot" if L == 0 and qi == 0 else "",
            )
        layer_idx += 1
        # entanglers
        for qi in range(n_qubits - 1):
            ax.plot(
                [x[layer_idx], x[layer_idx]],
                [y[qi], y[qi + 1]],
                color="black",
                linestyle="-",
            )
            ax.scatter(x[layer_idx], y[qi], s=50, marker="x")
            ax.scatter(x[layer_idx], y[qi + 1], s=50, marker="x")
        layer_idx += 1
    ax.axis("off")
    return fig, ax


def plot_training_loss(
    loss_history: List[float], figsize: Tuple[float, float] = (5, 3)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot loss vs training epochs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(loss_history) + 1), loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    return fig, ax


def plot_bloch_sphere(
    state_vector: np.ndarray, figsize: Tuple[float, float] = (4, 4)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the Bloch sphere representation of a single-qubit state.
    """
    # compute Bloch vector
    a = state_vector
    rho = np.outer(a, a.conj())
    # expectation values
    x = np.real(np.trace(rho @ np.array([[0, 1], [1, 0]])))
    y = np.real(np.trace(rho @ np.array([[0, -1j], [1j, 0]])))
    z = np.real(np.trace(rho @ np.array([[1, 0], [0, -1]])))
    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    # sphere
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.5)
    # Bloch vector
    ax.quiver(0, 0, 0, x, y, z, length=1.0, color="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax
