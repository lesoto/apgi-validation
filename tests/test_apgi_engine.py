"""Tests for APGI Engine module - comprehensive coverage for 0% coverage file."""

import numpy as np
import pytest

from apgi_core import (
    APGIAllostaticLayer,
    APGICoreSignal,
    APGIHierarchy,
    APGIIgnitionMechanism,
    APGILiquidNeuralNetwork,
    APGIPrecisionSystem,
    APGIPreProcessor,
    APGIRecovery,
    APGISystem,
    APGISystemDynamics,
    APGIValidationMetrics,
)
from utils.apgi_config import APGIConfig


class TestAPGIPreProcessor:
    """Test APGI PreProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = APGIPreProcessor()
        assert processor.window_size == 100
        assert processor.errors == []

    def test_init_custom_window(self):
        """Test initialization with custom window size."""
        processor = APGIPreProcessor(window_size=50)
        assert processor.window_size == 50

    def test_compute_prediction_error(self):
        """Test prediction error computation."""
        processor = APGIPreProcessor()
        error = processor.compute_prediction_error(10.0, 7.0)
        assert error == 3.0

        error_negative = processor.compute_prediction_error(5.0, 8.0)
        assert error_negative == -3.0

    def test_update_statistics_empty(self):
        """Test statistics update with empty errors."""
        processor = APGIPreProcessor()
        mu, sigma2 = processor.update_statistics(1.0)
        assert mu == 1.0
        assert sigma2 == 0.0
        assert len(processor.errors) == 1

    def test_update_statistics_multiple(self):
        """Test statistics with multiple errors."""
        processor = APGIPreProcessor(window_size=3)
        processor.update_statistics(1.0)
        processor.update_statistics(2.0)
        processor.update_statistics(3.0)
        mu, sigma2 = processor.update_statistics(4.0)

        # Should only have last 3 values: 2, 3, 4
        assert len(processor.errors) == 3
        assert pytest.approx(mu, 0.01) == 3.0

    def test_standardize(self):
        """Test z-score standardization."""
        processor = APGIPreProcessor()
        z = processor.standardize(10.0, 5.0, 2.0)
        assert pytest.approx(z, 0.0001) == 2.5

    def test_standardize_near_zero_sigma(self):
        """Test standardization with near-zero sigma."""
        processor = APGIPreProcessor()
        z = processor.standardize(5.0, 5.0, 0.0)
        assert z == 0.0  # Should handle division by near-zero


class TestAPGIPrecisionSystem:
    """Test APGI Precision System class."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        cfg = APGIConfig()
        cfg.Pi_e_init = 1.0
        return cfg

    def test_init(self, config):
        """Test initialization."""
        ps = APGIPrecisionSystem(config)
        assert ps.precision == 1.0
        assert ps.cfg == config

    def test_compute_precision(self, config):
        """Test precision computation."""
        ps = APGIPrecisionSystem(config)
        pi = ps.compute_precision(0.25)
        assert pytest.approx(pi, 0.0001) == 4.0

    def test_compute_precision_near_zero(self, config):
        """Test precision with near-zero variance."""
        ps = APGIPrecisionSystem(config)
        pi = ps.compute_precision(0.0)
        assert pi > 1e8  # Should be very large

    def test_effective_interoceptive_precision(self, config):
        """Test effective interoceptive precision."""
        ps = APGIPrecisionSystem(config)
        pi_eff = ps.effective_interoceptive_precision(1.0, 0.5, 1.0)
        expected = 1.0 * np.exp(0.5 * 1.0)
        assert pytest.approx(pi_eff, 0.0001) == expected

    def test_precision_ode(self, config):
        """Test precision ODE computation."""
        ps = APGIPrecisionSystem(config)
        result = ps.precision_ode(
            pi=1.0,
            epsilon=0.5,
            pi_next=1.2,
            pi_prev_psi=0.8,
            alpha=0.1,
            tau=2.0,
            c_down=0.05,
            c_up=0.03,
        )
        # dPi/dt = -Pi/tau + alpha|epsilon| + C_down(Pi_l+1 - Pi_l) + C_up * psi(epsilon_l-1)
        expected = (-1.0 / 2.0) + (0.1 * 0.5) + (0.05 * 0.2) + (0.03 * 0.8)
        assert pytest.approx(result, 0.0001) == expected


class TestAPGICoreSignal:
    """Test APGI Core Signal class."""

    def test_accumulated_signal(self):
        """Test accumulated signal computation."""
        result = APGICoreSignal.accumulated_signal(2.0, 3.0, 1.5, 4.0)
        expected = 2.0 * 3.0 + 1.5 * 4.0
        assert pytest.approx(result, 0.0001) == expected

    def test_accumulated_signal_negative(self):
        """Test with negative inputs (absolute value)."""
        result = APGICoreSignal.accumulated_signal(2.0, -3.0, 1.5, -4.0)
        expected = 2.0 * 3.0 + 1.5 * 4.0
        assert pytest.approx(result, 0.0001) == expected


class TestAPGIIgnitionMechanism:
    """Test APGI Ignition Mechanism class."""

    def test_logistic_ignition(self):
        """Test logistic ignition computation."""
        result = APGIIgnitionMechanism.logistic_ignition(1.0, 0.5, 2.0)
        assert 0 <= result <= 1

    def test_hard_ignition(self):
        """Test hard ignition threshold."""
        # Above threshold should ignite
        assert APGIIgnitionMechanism.hard_ignition(1.0, 0.5) is True
        # Below threshold should not ignite
        assert APGIIgnitionMechanism.hard_ignition(0.3, 0.5) is False

    def test_ignition_margin(self):
        """Test ignition margin computation."""
        result = APGIIgnitionMechanism.ignition_margin(1.0, 0.5)
        assert result == 0.5


class TestAPGISystemDynamics:
    """Test APGI System Dynamics class."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return APGIConfig()

    def test_signal_dynamics(self, config):
        """Test signal dynamics computation."""
        dynamics = APGISystemDynamics(config)
        s_t = 0.5
        result = dynamics.signal_dynamics(
            s_t=s_t, pi_e=1.0, ze=0.5, pi_i=1.0, zi=0.3, beta=0.1, tau_s=0.5, dt=0.01
        )
        assert isinstance(result, float)

    def test_threshold_dynamics(self, config):
        """Test threshold dynamics computation."""
        dynamics = APGISystemDynamics(config)
        theta_t = 1.0
        result = dynamics.threshold_dynamics(
            theta_t=theta_t,
            theta_0=1.0,
            b_prev=0.5,
            ds_dt=0.1,
            gamma=0.1,
            delta=0.5,
            lambda_urg=0.05,
            dt=0.01,
        )
        assert isinstance(result, float)


class TestAPGIAllostaticLayer:
    """Test APGI Allostatic Layer class."""

    def test_threshold_update(self):
        """Test threshold update computation."""
        result = APGIAllostaticLayer.threshold_update(1.0, 0.1, 0.5, 0.8)
        assert isinstance(result, float)

    def test_metabolic_cost(self):
        """Test metabolic cost computation."""
        result = APGIAllostaticLayer.metabolic_cost(10.0, 0.01)
        assert result == 0.1

    def test_landauer_limit(self):
        """Test Landauer limit computation."""
        result = APGIAllostaticLayer.landauer_limit(1.38e-23, 310.0)
        assert result > 0


class TestAPGILiquidNeuralNetwork:
    """Test APGI Liquid Neural Network class."""

    def test_init(self):
        """Test LNN initialization."""
        lnn = APGILiquidNeuralNetwork(size=100)
        assert lnn.size == 100
        assert lnn.x.shape == (100,)

    def test_reservoir_dynamics(self):
        """Test reservoir dynamics."""
        lnn = APGILiquidNeuralNetwork(size=100)
        x = np.zeros(100)
        result = lnn.reservoir_dynamics(x, u=1.0, tau_t=1.0, dt=0.01)
        assert result.shape == (100,)

    def test_signal_readout(self):
        """Test signal readout."""
        lnn = APGILiquidNeuralNetwork(size=100)
        x = np.random.randn(100)
        result = lnn.signal_readout(x)
        assert isinstance(result, float)


class TestAPGIHierarchy:
    """Test APGI Hierarchy class."""

    def test_level_count(self):
        """Test level count computation."""
        result = APGIHierarchy.level_count(tau_max=10.0, tau_min=0.1, overlap=2.0)
        assert result > 0

    def test_cross_level_modulation(self):
        """Test cross-level modulation."""
        result = APGIHierarchy.cross_level_modulation(1.0, 0.5, 0.0, 0.1)
        assert isinstance(result, float)

    def test_bottom_up_cascade(self):
        """Test bottom-up cascade."""
        result = APGIHierarchy.bottom_up_cascade(1.0, 0.6, 0.5, 0.1)
        assert isinstance(result, float)

    def test_phase_signal(self):
        """Test phase signal computation."""
        result = APGIHierarchy.phase_signal(omega=2 * np.pi, t=1.0, phi_0=0.0)
        assert isinstance(result, float)


class TestAPGIRecovery:
    """Test APGI Recovery class."""

    def test_reset_rule(self):
        """Test reset rule."""
        s_t, theta_t = APGIRecovery.reset_rule(1.0, 0.5, rho=0.1, delta=0.5)
        assert s_t == 0.1  # 1.0 * 0.1
        assert theta_t == 1.0  # 0.5 + 0.5


class TestAPGIValidationMetrics:
    """Test APGI Validation Metrics class."""

    def test_power_spectrum(self):
        """Test power spectrum computation."""
        f = np.linspace(1, 100, 1000)
        sigma_l = np.array([1.0, 0.5])
        tau_l = np.array([0.1, 0.05])
        result = APGIValidationMetrics.power_spectrum(f, sigma_l, tau_l)
        assert result.shape == f.shape
        assert np.all(result >= 0)

    def test_hurst_exponent(self):
        """Test Hurst exponent computation."""
        result = APGIValidationMetrics.hurst_exponent(beta_spec=1.0)
        assert result == 1.0  # (1 + 1) / 2


class TestAPGISystem:
    """Test APGI System (complete pipeline orchestrator)."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return APGIConfig()

    def test_init(self, config):
        """Test system initialization."""
        system = APGISystem(config)
        assert system.cfg == config
        assert system.s_t == 0.0
        assert system.theta_t == config.theta_init

    def test_step(self, config):
        """Test system step execution."""
        system = APGISystem(config)
        result = system.step(x=0.5, x_hat=0.0, x_i=0.3, x_hat_i=0.0, m_ca=0.1)
        assert isinstance(result, dict)
        assert "s_t" in result
        assert "b_t" in result
        assert "ignited" in result
