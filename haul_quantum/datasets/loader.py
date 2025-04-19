"""Utility loaders and classical → quantum encoders."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from ..qnn.encoders import amplitude_encoding, angle_encoding, basis_encoding


def load_xor() -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y


def prepare_quantum_dataset(
    X: np.ndarray,
    *,
    encoding: Literal["basis", "angle", "amplitude"] = "angle",
):
    dispatch = {
        "basis": basis_encoding,
        "angle": lambda row: angle_encoding(row, rotation="Y"),
        "amplitude": amplitude_encoding,
    }
    return [dispatch[encoding](row) for row in X]
