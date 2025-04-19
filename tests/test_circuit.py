import numpy as np

from haul_quantum.core.circuit import QuantumCircuit


def test_bell_state():
    # Our convention: qubit 0 is LSB → basis states ordered as |q1 q0>
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    out = qc.simulate()
    # Expect (|00> + |01>)/√2 → amplitudes at indices 0 and 1
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], complex)
    assert np.allclose(out, expected, atol=1e-8)
