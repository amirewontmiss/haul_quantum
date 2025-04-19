import numpy as np

from haul_quantum.core.circuit import QuantumCircuit
from haul_quantum.sim.batch import BatchSimulator


def test_batch_histogram():
    qc = QuantumCircuit(1)  # identity circuit
    bat = BatchSimulator(1, seed=0)
    hist = bat.run(qc, shots=100)
    assert sum(hist.values()) == 100
    # Only '0' outcome possible for |0>
    assert set(hist.keys()) == {"0"}
