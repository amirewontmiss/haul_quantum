import numpy as np

from haul_quantum.sim.noise import NoiseModel
from haul_quantum.sim.statevector import StatevectorSimulator


def test_noise_preserves_norm():
    sim = StatevectorSimulator(1)
    clean = sim.zero_state()
    nm = NoiseModel(1, seed=123)
    for channel in ("bit_flip", "phase_flip", "depolarizing"):
        noisy = getattr(nm, channel)(clean, p=0.5, qubit=0)
        # norm must remain 1
        assert np.isclose(np.linalg.norm(noisy), 1.0, atol=1e-6)
