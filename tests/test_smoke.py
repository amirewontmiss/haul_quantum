def test_engine_smoke():
    from haul_quantum.core.engine import Engine
    eng = Engine(2, seed=42)
    eng.h(0).cnot(0,1)
    state = eng.simulate()
    assert abs(state[0] - 1/2**0.5) < 1e-6

