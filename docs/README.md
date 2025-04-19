# Haul Quantum

[![PyPI version](https://img.shields.io/pypi/v/haul-quantum.svg)](https://pypi.org/project/haul-quantum)
[![Python versions](https://img.shields.io/pypi/pyversions/haul-quantum.svg)](https://pypi.org/project/haul-quantum)
[![License](https://img.shields.io/github/license/amirewontmiss/haul_quantum.svg)](LICENSE)

A lightweight, extensible **quantum computing framework** for Python, designed for research and prototyping. Haul Quantum provides:

* **Pure-Python simulation** with statevector backends.
* **Fluent API** via `Engine` or direct `QuantumCircuit`.
* **Chainable gates**: `H`, `X`, `CNOT`, `RX`, `RY`, `RZ`, and more.
* **Noise modeling** and batch simulation support.
* **Torch integration** for hybrid quantum-classical neural nets.

---

## 🚀 Installation

```bash
pip install haul-quantum
```

Or install the latest development build:

```bash
git clone https://github.com/amirewontmiss/haul_quantum.git
cd haul_quantum
pip install -e .[dev]
```

## 🎯 Quick Start

### Using the Engine

```python
from haul_quantum.core.engine import Engine

# Create a 2-qubit circuit, build a Bell state:
eng = Engine(2)
out = eng.h(0).cnot(0,1).simulate()
print(out)  # [0.707+0j, 0.707+0j, 0+0j, 0+0j]

# Measure probabilities:
probs = eng.measure()
print(probs)  # {'00': 0.5, '01': 0.5}
```

### Direct Circuit API

```python
from haul_quantum.core.circuit import QuantumCircuit
from haul_quantum.core.gates import RX, H, CNOT

qc = QuantumCircuit(3)
qc.h(0).rx(1.23)(1).cnot(0,2)
state = qc.simulate()
```

## 📚 API Reference

### `Engine`

| Method            | Description                                       |
| ----------------- | ------------------------------------------------- |
| `Engine(n, seed)` | Create engine with *n* qubits, optional RNG seed. |
| `h(q)`            | Apply Hadamard on qubit *q*. Returns self.        |
| `x(q)`            | Apply Pauli-X on qubit *q*. Returns self.         |
| `cnot(ctrl, tgt)` | Controlled-NOT (control & target) on two qubits.  |
| `rx(theta)(q)`    | Rotation-X by *theta* on qubit *q*.               |
| `simulate()`      | Return full statevector as a NumPy array.         |
| `measure()`       | Return a dict of basis-state probabilities.       |
| `to_qasm()`       | Export to OpenQASM 2.0 string.                    |
| `reset()`         | Clear all gates, preserve qubit count & seed.     |

### `QuantumCircuit`

Same API as `Engine`, but stateless. Useful for circuit transformations, compilation, and exporting without an `Engine` wrapper.

## 🔌 Features

* **Statevector simulator:** Pure NumPy backend, no external dependencies.
* **NoiseModel:** Apply `bit_flip`, `phase_flip`, `depolarizing` channels.
* **Batch simulation:** Collect histograms over many shots.
* **Torch integration:** Wrap circuits as `torch.nn.Module` for hybrid training.

## 🖋️ Contributing

Pull requests and issues welcome! Please read `CONTRIBUTING.md` for guidelines.

## 📄 License

MIT © [amir ewontmiss](https://github.com/amirewontmiss)

