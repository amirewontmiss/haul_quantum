"""
haul_quantum.datasets.loader
=============================
Classical and quantum dataset loaders and converters.

Provides:
- load_xor: simple 2-bit XOR dataset
- load_iris: the Iris dataset (3 classes) with optional train/test split
- load_mnist: MNIST digit dataset (with optional subsampling & preprocessing)
- prepare_quantum_dataset: convert classical features to quantum objects
    – returns lists of QuantumCircuit or statevectors + labels
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..core.circuit import QuantumCircuit
from ..qnn.encoders import amplitude_encoding, angle_encoding, basis_encoding

# ------------------------------------------------------------------------------
# Classical dataset loaders
# ------------------------------------------------------------------------------


def load_xor() -> Tuple[np.ndarray, np.ndarray]:
    """
    2-bit XOR dataset:
      inputs: [[0,0], [0,1], [1,0], [1,1]]
      labels: [0, 1, 1, 0]
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    y = np.array([0, 1, 1, 0], dtype=int)
    return X, y


def load_iris_data(
    test_size: float = 0.3, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Iris dataset and split into train/test.

    Returns:
        X_train, X_test: float arrays shape (n_samples, 4)
        y_train, y_test: integer labels {0,1,2}
    """
    data = load_iris()
    X, y = data.data, data.target
    # scale features to [0, π] for angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)


def load_mnist_data(
    digits: Union[List[int], Tuple[int, ...]] = (0, 1),
    n_samples: Optional[int] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST-like digit data from sklearn (8x8 images).

    Args:
        digits: which digit labels to include (e.g. (0,1,2))
        n_samples: total number of samples to subsample (balanced per class)
        test_size: fraction for test split
        random_state: RNG seed
    Returns:
        X_train, X_test: arrays of shape (n_samples, 64) with pixel values [0,16]
        y_train, y_test: integer labels
    """
    data = load_digits()
    X, y = data.data, data.target
    mask = np.isin(y, digits)
    X, y = X[mask], y[mask]
    if n_samples is not None and n_samples < X.shape[0]:
        # balanced sampling
        rng = np.random.default_rng(random_state)
        per_class = n_samples // len(digits)
        idxs = []
        for d in digits:
            ids = np.where(y == d)[0]
            sel = rng.choice(ids, size=per_class, replace=False)
            idxs.extend(sel.tolist())
        X = X[idxs]
        y = y[idxs]
    # normalize pixel values to [0, π]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)


# ------------------------------------------------------------------------------
# Quantum dataset preparation
# ------------------------------------------------------------------------------


def prepare_quantum_dataset(
    X: np.ndarray, encoding: Literal["basis", "angle", "amplitude"] = "angle"
) -> Tuple[List[Union[QuantumCircuit, np.ndarray]], np.ndarray]:
    """
    Convert classical feature matrix X to quantum inputs.

    Args:
        X: array of shape (n_samples, n_features). Must satisfy:
            - For 'basis': features are 0 or 1.
            - For 'angle': real numbers (will be used as rotation angles).
            - For 'amplitude': length must be 2**n in flattened form.
        encoding: one of 'basis', 'angle', 'amplitude'.
    Returns:
        circuits_or_states: list where each element is either:
            - QuantumCircuit (for 'basis' or 'angle')
            - NumPy statevector (for 'amplitude')
        labels: original y array passed alongside X.
    """
    n_features = X.shape[1]
    outputs = []
    for row in X:
        if encoding == "basis":
            # treat row as bits
            circ = basis_encoding(row.astype(int).tolist())
            outputs.append(circ)
        elif encoding == "angle":
            circ = angle_encoding(row.tolist(), rotation="Y")
            outputs.append(circ)
        elif encoding == "amplitude":
            # concatenate row if needed, ensure power-of-2 length
            state = amplitude_encoding(row.tolist())
            outputs.append(state)
        else:
            raise ValueError("encoding must be 'basis', 'angle', or 'amplitude'")
    return outputs


# ------------------------------------------------------------------------------
# Example usage (to remove in production)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # XOR
    X_xor, y_xor = load_xor()
    circ_xor = prepare_quantum_dataset(X_xor, encoding="basis")
    print("XOR circuits:", circ_xor)

    # Iris
    X_train, X_test, y_train, y_test = load_iris_data()
    circ_train = prepare_quantum_dataset(X_train, encoding="angle")
    print("Iris train circuits:", len(circ_train))

    # MNIST (0 vs 1)
    X_train, X_test, y_train, y_test = load_mnist_data(digits=(0, 1), n_samples=200)
    circ_train = prepare_quantum_dataset(X_train, encoding="angle")
    print("MNIST train circuits:", len(circ_train))
