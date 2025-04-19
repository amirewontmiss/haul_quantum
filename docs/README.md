# Haul Quantum AI Framework

[![CI](https://github.com/amirewontmiss/haul_quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/amirewontmiss/haul_quantum/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Haul** is a modular, extensible, hardware-agnostic quantum-classical AI framework. It brings together circuit construction, statevector & noise simulation, hybrid quantum-neural-network layers, training utilities, datasets, and visualization into a single, easy-to-use Python package.

---

## 🚀 Features

* **Core abstractions**
  – `Gate`, `QuantumCircuit`, chainable builder API
* **Simulators**
  – CPU statevector, Monte Carlo noise channels, shot-based batch simulator
* **Compiler & IR**
  – Export to OpenQASM 2.0 & JSON intermediate representation
* **QNN & ML integration**
  – Variational layers (`VQCLayer`), encoders (basis/angle/amplitude), PyTorch QNode support
* **Training utilities**
  – Generic `Trainer`, callbacks (early stopping, checkpointing, CSV logger, progress bar), built-in optimizers
* **Datasets**
  – Classical (XOR, Iris, MNIST) & quantum (basis, GHZ, Bell, random Haar states)
* **Visualization**
  – ASCII & Matplotlib circuit diagrams, VQC architecture schematics, Bloch sphere plots

---

## 📦 Installation

```bash
# From PyPI (coming soon):
pip install haul_quantum

# Or install latest from GitHub:
git clone https://github.com/amirewontmiss/haul_quantum.git
cd haul_quantum
pip install -e .
```

> **Note:** it’s best practice to use a virtual environment.

---

## 🏁 Quickstart

### Build & simulate a Bell state

```python
from haul_quantum.core.engine import Engine

# Create a 2-qubit engine
eng = Engine(n_qubits=2, seed=123)

# Build a Bell circuit
eng.h(0).cnot(0, 1)

# Pure statevector simulation
state = eng.simulate()
print("Bell statevector:", state)
```

### Export to QASM

```python
qasm = eng.to_qasm()
print(qasm)
```

### Variational Quantum Circuit (VQC) training

```python
import numpy as np
from haul_quantum.qnn.layers import VQCLayer
from haul_quantum.datasets.loader import load_xor, prepare_quantum_dataset
from haul_quantum.train.loop import Trainer
from haul_quantum.train.optimizer import GradientDescent

# Load XOR data and prepare circuits
X, y = load_xor()
circuits = prepare_quantum_dataset(X, encoding="basis")

# Build a 2-qubit, 1-layer VQC
vqc = VQCLayer(n_qubits=2, n_layers=1)
params = np.random.uniform(0, 2*np.pi, vqc.num_parameters)

# Define model, loss, gradient functions
def model_fn(p):
    qc = vqc.build_circuit(p)
    state = qc.simulate()
    probs = np.abs(state)**2
    return probs[0] + probs[1] - probs[2] - probs[3]

def loss_fn(pred):
    y_enc = 2*y - 1
    return float(np.mean((pred - y_enc)**2))

def grad_fn(p):
    eps = 1e-3
    grads = np.zeros_like(p)
    base = loss_fn(model_fn(p))
    for i in range(len(p)):
        dp = np.zeros_like(p); dp[i] = eps
        grads[i] = (loss_fn(model_fn(p+dp)) - loss_fn(model_fn(p-dp))) / (2*eps)
    return grads

trainer = Trainer(
    model_fn=model_fn,
    initial_params=params,
    loss_fn=loss_fn,
    optimizer=GradientDescent(learning_rate=0.1),
    gradient_fn=grad_fn,
    max_epochs=50,
    callbacks=[]
)
trained = trainer.fit()
print("Trained parameters:", trained)
```

---

## 📚 Documentation & Examples

* Full API docs and tutorials in [`docs/README.md`](docs/README.md)
* Example scripts in [`haul_quantum/experiments`](haul_quantum/experiments)

---

## 🤝 Contributing

We welcome contributions! Please read our:

* [Code of Conduct](CODE_OF_CONDUCT.md)
* [Contributing Guidelines](CONTRIBUTING.md)

Then:

```bash
git clone https://github.com/amirewontmiss/haul_quantum.git
cd haul_quantum
pip install -e .
pre-commit install
pre-commit run --all-files
```

---

## ⚖️ License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built with ♥ for the quantum-machine learning community.*

