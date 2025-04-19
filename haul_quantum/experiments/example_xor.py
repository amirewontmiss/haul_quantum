"""
Demonstration: train a tiny VQC on XOR.
"""

import numpy as np

from haul_quantum.datasets.loader import load_xor, prepare_quantum_dataset
from haul_quantum.qnn.layers import VQCLayer
from haul_quantum.train.loop import Trainer
from haul_quantum.train.optimizer import GradientDescent

X, y = load_xor()
_ = prepare_quantum_dataset(X, encoding="basis")  # noqa: F841 (demo only)

vqc = VQCLayer(n_qubits=2, n_layers=1)
params = np.random.uniform(0, 2 * np.pi, vqc.num_parameters)


def model_fn(p):
    qc = vqc.build_circuit(p)
    state = qc.simulate()
    probs = np.abs(state) ** 2
    return probs[0] + probs[1] - probs[2] - probs[3]


def loss_fn(pred):
    y_enc = 2 * y - 1
    return float(np.mean((pred - y_enc) ** 2))


def grad_fn(p):
    eps = 1e-3
    grads = np.zeros_like(p)
    for i in range(len(p)):
        dp = np.zeros_like(p)
        dp[i] = eps
        grads[i] = (loss_fn(model_fn(p + dp)) - loss_fn(model_fn(p - dp))) / (2 * eps)
    return grads


Trainer(
    model_fn=model_fn,
    initial_params=params,
    loss_fn=loss_fn,
    optimizer=GradientDescent(0.1),
    gradient_fn=grad_fn,
    max_epochs=50,
    callbacks=[],
).fit()
