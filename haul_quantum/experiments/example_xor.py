"""
example_xor.py
==============
Demonstration: Training a VQC on the XOR dataset using haul_quantum framework.
"""

import numpy as np

from haul_quantum.datasets.loader import load_xor, prepare_quantum_dataset
from haul_quantum.qnn.layers import VQCLayer
from haul_quantum.train.loop import Trainer
from haul_quantum.train.optimizer import GradientDescent

def main():
    # Load XOR
    X, y = load_xor()
    circuits = prepare_quantum_dataset(X, encoding="basis")

    # Model: simple VQC (2 qubits, 1 layer)
    vqc = VQCLayer(n_qubits=2, n_layers=1)
    initial_params = np.random.uniform(0, 2*np.pi, vqc.num_parameters)

    # Model function: returns expectation value ⟨Z0⟩ - ⟨Z1⟩ as prediction
    def model_fn(params: np.ndarray) -> np.ndarray:
        qc = vqc.build_circuit(params)
        state = qc.simulate()
        probs = np.abs(state)**2
        # expectation on each qubit
        exp0 = probs[0] + probs[1] - (probs[2] + probs[3])
        exp1 = probs[0] + probs[2] - (probs[1] + probs[3])
        # combine into array
        return np.array([exp0, exp1])

    # Encode labels as {-1, +1}
    y_encoded = 2*y - 1

    # Loss: average squared error
    def loss_fn(output: np.ndarray) -> float:
        target = y_encoded
        return np.mean((output - target)**2)

    # Gradient: finite-difference
    def gradient_fn(params: np.ndarray) -> np.ndarray:
        grads = np.zeros_like(params)
        eps = 1e-3
        base_loss = loss_fn(model_fn(params))
        for i in range(len(params)):
            dp = np.zeros_like(params); dp[i] = eps
            grads[i] = (loss_fn(model_fn(params + dp)) - loss_fn(model_fn(params - dp))) / (2*eps)
        return grads

    optimizer = GradientDescent(learning_rate=0.1)
    trainer = Trainer(
        model_fn=model_fn,
        initial_params=initial_params,
        loss_fn=loss_fn,
        optimizer=optimizer,
        gradient_fn=gradient_fn,
        max_epochs=100,
        tol=1e-4,
        callbacks=[]
    )

    final_params = trainer.fit()
    print("Trained params:", final_params)

if __name__ == "__main__":
    main()

