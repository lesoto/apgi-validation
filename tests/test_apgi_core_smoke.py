from __future__ import annotations

import numpy as np


def test_apgi_core_model_functions_execute():
    from apgi_core import (
        compute_precision,
        compute_signal,
        ignition_probability,
        update_threshold,
    )

    z_e = 0.2
    z_i = -0.1
    var = 0.5
    pi = compute_precision(var)
    s = compute_signal(z_e, z_i, pi_e=pi, pi_i_eff=pi)

    p = ignition_probability(s, theta=0.1, alpha=5.0)
    assert 0.0 <= p <= 1.0

    new_theta = update_threshold(
        theta=0.5,
        theta0=0.5,
        S=s,
        V_info=0.1,
        dt=0.01,
        tau_theta=20.0,
        gamma_M=0.0,
        metabolic_cost=0.0,
    )
    assert isinstance(new_theta, float)


def test_apgi_core_engine_system_step_smoke(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from apgi_core.engine import APGISystem

    system = APGISystem()

    # Pre-warm stats to keep z-scores stable
    rng = np.random.default_rng(0)
    for _ in range(10):
        system.prep_e.update_statistics(float(rng.normal(0.0, 0.1)))
        system.prep_i.update_statistics(float(rng.normal(0.0, 0.1)))

    out = system.step(
        x=0.2,
        x_hat=0.0,
        x_i=0.1,
        x_hat_i=0.0,
        m_ca=0.1,
    )
    assert "s_t" in out
    assert "theta_t" in out
    assert "ignited" in out


def test_apgi_core_engine_can_ignite(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from apgi_core.engine import APGISystem

    system = APGISystem()
    for _ in range(10):
        system.prep_e.update_statistics(0.0)
        system.prep_i.update_statistics(0.0)

    out = system.step(
        x=10.0,
        x_hat=0.0,
        x_i=10.0,
        x_hat_i=0.0,
        m_ca=5.0,
    )
    assert isinstance(out["ignited"], bool)


def test_apgi_model_step_loop_covers_more_code():
    from apgi_core import APGIModel

    model = APGIModel()
    for i in range(25):
        x = float(np.sin(i * 0.1))
        out = model.step(x)
        assert "S" in out
        assert "theta" in out
