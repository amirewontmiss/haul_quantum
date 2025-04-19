from haul_quantum.core.circuit import QuantumCircuit
from haul_quantum.sim.batch import BatchSimulator


def test_batch_histogram():
    qc = QuantumCircuit(1)
    hist = BatchSimulator(1).run(qc, shots=100)
    assert sum(hist.values()) == 100
