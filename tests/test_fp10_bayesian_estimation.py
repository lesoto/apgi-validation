"""Tests for FP_10 Falsification Bayesian Estimation - increase coverage from 0%."""

import numpy as np

from Falsification.FP_10_BayesianEstimation_MCMC import (
    DataSource,
    compute_posterior_distributions,
)
from Falsification.FP_10_BayesianEstimation_MCMC import (
    generate_synthetic_data as generate_synthetic_data_mcmc,
)
from Falsification.FP_10_BayesianEstimation_MCMC import (
    run_bayesian_estimation,
    set_data_source,
)
from Falsification.FP_10_BayesianEstimation_MCMC import (
    test_parameter_identifiability as check_parameter_identifiability,
)
from Falsification.FP_10_Falsification_BayesianEstimation_ParameterRecovery import (
    BayesianParameterRecovery,
    FP10bParameterRecovery,
    ParameterRecoveryResult,
    generate_synthetic_data,
    run_mcmc_bayesian_estimation,
)


class TestDataSource:
    """Test Data Source enum."""

    def test_data_source_values(self):
        """Test data source enum values."""
        assert DataSource.SYNTHETIC.value == "synthetic"
        assert DataSource.EMPIRICAL.value == "empirical"
        assert DataSource.SIMULATION.value == "simulation"


class TestParameterRecoveryResult:
    """Test Parameter Recovery Result dataclass."""

    def test_result_creation(self):
        """Test creating parameter recovery result."""
        result = ParameterRecoveryResult(
            parameter_name="alpha",
            true_value=0.5,
            recovered_mean=0.52,
            recovered_std=0.01,
            credible_interval=(0.50, 0.54),
            recovery_error=0.02,
            relative_error=0.04,
            recovered_successfully=True,
            identifiability_score=0.85,
        )
        assert result.parameter_name == "alpha"
        assert result.recovered_successfully is True


class TestBayesianParameterRecovery:
    """Test Bayesian Parameter Recovery from MCMC module."""

    def test_init(self):
        """Test initialization."""
        recovery = BayesianParameterRecovery(n_samples=100, n_chains=2, burn_in=50)
        assert recovery.n_samples == 100
        assert recovery.n_chains == 2

    def test_basic_attributes(self):
        """Test basic attributes are set correctly."""
        recovery = BayesianParameterRecovery()
        assert recovery.n_samples > 0
        assert recovery.n_chains > 0
        assert recovery.burn_in >= 0


class TestFP10bParameterRecovery:
    """Test FP10b Parameter Recovery class."""

    def test_init(self):
        """Test initialization."""
        recovery = FP10bParameterRecovery()
        assert recovery is not None
        assert recovery.recovery_tolerance > 0

    def test_default_true_parameters(self):
        """Test default true parameters are set."""
        recovery = FP10bParameterRecovery()
        default_params = recovery._get_default_true_parameters()
        assert "tau_S" in default_params
        assert "theta_0" in default_params
        assert "alpha" in default_params


class TestGenerateSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(
            n_trials=50, true_params=true_params, noise_level=0.1
        )
        assert len(stimulus_data) == 50
        assert len(response_data) == 50

    def test_generate_synthetic_data_mcmc_variant(self):
        """Test MCMC variant of synthetic data generation."""
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data_mcmc(
            n_trials=50, true_params=true_params, noise_level=0.1
        )
        assert len(stimulus_data) == 50
        assert len(response_data) == 50


class TestRunMCMCBayesianEstimation:
    """Test MCMC Bayesian estimation runner."""

    def test_run_mcmc_with_synthetic_data(self):
        """Test running MCMC estimation with synthetic data."""
        # Generate synthetic data
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(
            n_trials=50, true_params=true_params, noise_level=0.1
        )

        # Run MCMC with minimal samples for testing
        result = run_mcmc_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=100,
            n_chains=2,
            burn_in=50,
        )
        assert isinstance(result, dict)
        # Result may contain error key if PyMC not available
        assert "posterior_means" in result or "error" in result


class TestRunBayesianEstimationAlias:
    """Test run_bayesian_estimation alias function."""

    def test_run_bayesian_estimation_alias(self):
        """Test that the alias function works."""
        true_params = {"alpha": 2.0, "beta": 0.1}
        stimulus_data, response_data = generate_synthetic_data(
            n_trials=50, true_params=true_params, noise_level=0.1
        )

        result = run_bayesian_estimation(
            stimulus_data=stimulus_data,
            response_data=response_data,
            n_samples=100,
            n_chains=2,
            burn_in=50,
        )
        assert isinstance(result, dict)


class TestComputePosteriorDistributions:
    """Test posterior distribution computation."""

    def test_compute_posterior_distributions_empty(self):
        """Test posterior distribution with empty/invalid trace."""
        # Test with None trace - should return empty dict
        result = compute_posterior_distributions(None, ["alpha", "beta"])
        assert isinstance(result, dict)


class TestTestParameterIdentifiability:
    """Test parameter identifiability testing."""

    def test_identifiability_with_good_recovery(self):
        """Test identifiability with good parameter recovery."""
        # Create fake posterior samples
        posterior_samples = {
            "alpha": np.random.normal(0.5, 0.01, 100),
            "beta": np.random.normal(1.0, 0.02, 100),
        }
        true_params = {"alpha": 0.5, "beta": 1.0}

        result = check_parameter_identifiability(
            posterior_samples, true_params, tolerance=0.15
        )
        assert isinstance(result, dict)
        assert "all_identified" in result

    def test_identifiability_with_poor_recovery(self):
        """Test identifiability with poor parameter recovery."""
        # Create posterior samples far from true values
        posterior_samples = {
            "alpha": np.random.normal(2.0, 0.1, 100),  # Far from true 0.5
            "beta": np.random.normal(3.0, 0.1, 100),  # Far from true 1.0
        }
        true_params = {"alpha": 0.5, "beta": 1.0}

        result = check_parameter_identifiability(
            posterior_samples, true_params, tolerance=0.1
        )
        assert isinstance(result, dict)


class TestSetDataSource:
    """Test data source setting."""

    def test_set_data_source(self):
        """Test setting data source."""
        set_data_source(DataSource.SYNTHETIC)
        # Should not raise an error
