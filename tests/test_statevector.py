import numpy as np
from haul_quantum.sim.statevector import StatevectorSimulator
from haul_quantum.core.circuit import QuantumCircuit
from haul_quantum.core.gates import H, CNOT

def test_sv_matches_circuit():
    qc = QuantumCircuit(2).h(0).cnot(0,1)
    sv = StatevectorSimulator(2).simulate(qc.instructions)
    assert np.allclose(sv, qc.simulate())

