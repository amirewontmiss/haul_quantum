import numpy as np
import pytest
from haul_quantum.core.gates import X, H, RX

def test_x_gate_matrix():
    Xm = X().matrix
    expected = np.array([[0,1],[1,0]], complex)
    assert np.allclose(Xm, expected)

def test_h_gate_unitary():
    Hm = H().matrix
    # H·H = I
    assert np.allclose(Hm @ Hm, np.eye(2), atol=1e-8)

@pytest.mark.parametrize("θ", [0, np.pi/4, np.pi/2, np.pi])
def test_rx_inversion(θ):
    # RX(θ)·RX(−θ) = I
    rx_p = RX(θ).matrix
    rx_m = RX(-θ).matrix
    assert np.allclose(rx_p @ rx_m, np.eye(2), atol=1e-8)

