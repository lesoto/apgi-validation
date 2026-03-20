"""
Property-based tests for Hypothesis-worthy mathematical functions.
Uses Hypothesis to test invariants and properties across parameter ranges.
=======================================================================
"""

import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as np_st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import APGI equation functions
from APGI_Equations import (
    compute_surprise,
    compute_threshold,
    compute_metabolic_cost,
    compute_arousal,
    compute_entropy,
    compute_kl_divergence,
    compute_mutual_information,
    compute_bayesian_update,
    compute_active_inference_error,
    compute_free_energy,
)


class TestComputeSurpriseProperties:
    """Property-based tests for compute_surprise function."""

    @given(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_surprise_positive(self, prediction_error):
        """Surprise should always be positive for positive prediction errors."""
        surprise = compute_surprise(prediction_error)
        assert surprise >= 0

    @given(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_surprise_monotonic(self, prediction_error):
        """Surprise should be monotonic with prediction error."""
        surprise1 = compute_surprise(prediction_error)
        surprise2 = compute_surprise(prediction_error * 2)
        assert surprise2 >= surprise1

    @given(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_surprise_zero_at_zero(self, prediction_error):
        """Surprise should approach zero as prediction error approaches zero."""
        small_error = prediction_error * 0.001
        surprise_small = compute_surprise(small_error)
        surprise_large = compute_surprise(prediction_error)
        assert surprise_small < surprise_large


class TestComputeThresholdProperties:
    """Property-based tests for compute_threshold function."""

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_threshold_in_range(self, precision, surprise):
        """Threshold should be within valid range [0, 1]."""
        threshold = compute_threshold(precision, surprise)
        assert 0 <= threshold <= 1

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_threshold_decreases_with_precision(self, precision, surprise):
        """Higher precision should lead to lower threshold."""
        threshold1 = compute_threshold(precision, surprise)
        threshold2 = compute_threshold(precision * 0.5, surprise)
        assert threshold2 <= threshold1

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(
            min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_threshold_increases_with_surprise(self, precision, surprise):
        """Higher surprise should lead to higher threshold."""
        threshold1 = compute_threshold(precision, surprise)
        threshold2 = compute_threshold(precision, surprise * 2)
        assert threshold2 >= threshold1


class TestComputeMetabolicCostProperties:
    """Property-based tests for compute_metabolic_cost function."""

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_metabolic_cost_positive(self, surprise, threshold):
        """Metabolic cost should always be positive or zero."""
        cost = compute_metabolic_cost(surprise, threshold)
        assert cost >= 0

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_metabolic_cost_zero_at_equilibrium(self, surprise, threshold):
        """Metabolic cost should be zero when surprise equals threshold."""
        cost = compute_metabolic_cost(surprise, surprise)
        assert cost == 0

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_metabolic_cost_symmetric(self, surprise, threshold):
        """Metabolic cost should be symmetric around threshold."""
        cost1 = compute_metabolic_cost(surprise, threshold)
        cost2 = compute_metabolic_cost(threshold + (threshold - surprise), threshold)
        assert abs(cost1 - cost2) < 1e-10


class TestComputeArousalProperties:
    """Property-based tests for compute_arousal function."""

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_arousal_in_range(self, surprise, threshold):
        """Arousal should be within valid range [0, 1]."""
        arousal = compute_arousal(surprise, threshold)
        assert 0 <= arousal <= 1

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_arousal_low_at_equilibrium(self, surprise, threshold):
        """Arousal should be low when surprise equals threshold."""
        arousal = compute_arousal(surprise, surprise)
        assert arousal < 0.5

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_arousal_high_with_large_discrepancy(self, surprise, threshold):
        """Arousal should be high with large surprise-threshold discrepancy."""
        arousal = compute_arousal(surprise * 10, threshold)
        assert arousal > 0.5


class TestComputeEntropyProperties:
    """Property-based tests for compute_entropy function."""

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_entropy_non_negative(self, distribution):
        """Entropy should always be non-negative."""
        # Normalize to ensure it's a valid probability distribution
        distribution = np.abs(distribution)
        distribution = distribution / np.sum(distribution)
        entropy = compute_entropy(distribution)
        assert entropy >= 0

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_entropy_maximum_for_uniform(self, distribution):
        """Entropy should be maximum for uniform distribution."""
        n = len(distribution)
        uniform = np.ones(n) / n
        entropy_uniform = compute_entropy(uniform)

        # Normalize the test distribution
        distribution = np.abs(distribution)
        distribution = distribution / np.sum(distribution)
        entropy_test = compute_entropy(distribution)

        assert entropy_test <= entropy_uniform + 1e-10

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_entropy_zero_for_deterministic(self, distribution):
        """Entropy should be zero for deterministic distribution."""
        n = len(distribution)
        deterministic = np.zeros(n)
        deterministic[0] = 1.0
        entropy = compute_entropy(deterministic)
        assert entropy == 0


class TestComputeKLDivergenceProperties:
    """Property-based tests for compute_kl_divergence function."""

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )
    @settings(max_examples=50)
    def test_kl_divergence_non_negative(self, p, q):
        """KL divergence should always be non-negative."""
        # Normalize to valid probability distributions
        p = np.abs(p)
        p = p / np.sum(p)
        q = np.abs(q)
        q = q / np.sum(q)

        kl = compute_kl_divergence(p, q)
        assert kl >= 0

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_kl_divergence_zero_for_identical(self, p):
        """KL divergence should be zero for identical distributions."""
        p = np.abs(p)
        p = p / np.sum(p)

        kl = compute_kl_divergence(p, p)
        assert kl == 0

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=50),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )
    @settings(max_examples=30)
    def test_kl_divergence_triangle_inequality(self, p, q, r):
        """KL divergence should satisfy triangle inequality approximately."""
        p = np.abs(p) / np.sum(np.abs(p))
        q = np.abs(q) / np.sum(np.abs(q))
        r = np.abs(r) / np.sum(np.abs(r))

        kl_pq = compute_kl_divergence(p, q)
        kl_qr = compute_kl_divergence(q, r)
        kl_pr = compute_kl_divergence(p, r)

        # Triangle inequality for KL divergence (not strict but should hold approximately)
        assert kl_pr <= kl_pq + kl_qr + 1e-5


class TestComputeMutualInformationProperties:
    """Property-based tests for compute_mutual_information function."""

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            ),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_mutual_information_non_negative(self, joint):
        """Mutual information should always be non-negative."""
        # Normalize joint distribution
        joint = np.abs(joint)
        joint = joint / np.sum(joint)

        mi = compute_mutual_information(joint)
        assert mi >= 0

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            ),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_mutual_information_symmetric(self, joint):
        """Mutual information should be symmetric."""
        joint = np.abs(joint)
        joint = joint / np.sum(joint)

        mi = compute_mutual_information(joint)
        mi_transposed = compute_mutual_information(joint.T)

        assert abs(mi - mi_transposed) < 1e-10

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            ),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_mutual_information_bounded_by_entropy(self, joint):
        """Mutual information should be bounded by individual entropies."""
        joint = np.abs(joint)
        joint = joint / np.sum(joint)

        mi = compute_mutual_information(joint)

        # Marginal entropies
        p_x = np.sum(joint, axis=1)
        p_y = np.sum(joint, axis=0)

        h_x = compute_entropy(p_x)
        h_y = compute_entropy(p_y)

        assert mi <= min(h_x, h_y) + 1e-10


class TestComputeBayesianUpdateProperties:
    """Property-based tests for compute_bayesian_update function."""

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )
    @settings(max_examples=50)
    def test_bayesian_update_normalizes(self, prior, likelihood):
        """Posterior should be normalized probability distribution."""
        prior = np.abs(prior)
        prior = prior / np.sum(prior)

        likelihood = np.abs(likelihood)
        likelihood = likelihood / np.sum(likelihood)

        posterior = compute_bayesian_update(prior, likelihood)

        assert np.abs(np.sum(posterior) - 1.0) < 1e-10

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_bayesian_update_with_uniform_prior(self, likelihood):
        """Posterior with uniform prior should be proportional to likelihood."""
        n = len(likelihood)
        uniform_prior = np.ones(n) / n

        likelihood = np.abs(likelihood)
        likelihood = likelihood / np.sum(likelihood)

        posterior = compute_bayesian_update(uniform_prior, likelihood)

        # Posterior should equal likelihood for uniform prior
        assert np.allclose(posterior, likelihood, atol=1e-10)

    @given(
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        np_st.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )
    @settings(max_examples=50)
    def test_bayesian_update_commutative(self, prior, likelihood):
        """Bayesian update should be commutative for identical priors/likelihoods."""
        prior = np.abs(prior) / np.sum(np.abs(prior))
        likelihood = np.abs(likelihood) / np.sum(np.abs(likelihood))

        posterior1 = compute_bayesian_update(prior, likelihood)
        posterior2 = compute_bayesian_update(likelihood, prior)

        assert np.allclose(posterior1, posterior2, atol=1e-10)


class TestComputeActiveInferenceErrorProperties:
    """Property-based tests for compute_active_inference_error function."""

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_active_inference_error_non_negative(
        self, prediction, observation, precision
    ):
        """Active inference error should always be non-negative."""
        error = compute_active_inference_error(prediction, observation, precision)
        assert error >= 0

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_active_inference_error_zero_at_match(self, prediction, precision):
        """Error should be zero when prediction matches observation."""
        error = compute_active_inference_error(prediction, prediction, precision)
        assert error == 0

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_active_inference_error_increases_with_precision(
        self, prediction, observation
    ):
        """Error should increase with higher precision for same prediction error."""
        error1 = compute_active_inference_error(prediction, observation, 0.5)
        error2 = compute_active_inference_error(prediction, observation, 1.0)
        assert error2 >= error1


class TestComputeFreeEnergyProperties:
    """Property-based tests for compute_free_energy function."""

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_free_energy_decomposition(self, surprise, threshold, complexity):
        """Free energy should decompose into accuracy and complexity terms."""
        fe = compute_free_energy(surprise, threshold, complexity)

        # Free energy = accuracy + complexity
        accuracy = compute_surprise(surprise, threshold)
        expected_fe = accuracy + complexity

        assert abs(fe - expected_fe) < 1e-10

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_free_energy_minimized_at_optimal(self, surprise, complexity):
        """Free energy should be minimized when surprise equals threshold."""
        fe1 = compute_free_energy(surprise, surprise, complexity)
        fe2 = compute_free_energy(surprise, surprise * 2, complexity)
        fe3 = compute_free_energy(surprise, surprise * 0.5, complexity)

        assert fe1 <= fe2
        assert fe1 <= fe3

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_free_energy_tradeoff(self, surprise, threshold, complexity):
        """Free energy should balance accuracy and complexity."""
        fe_low_complexity = compute_free_energy(surprise, threshold, complexity * 0.5)
        fe_high_complexity = compute_free_energy(surprise, threshold, complexity * 2)

        assert fe_high_complexity >= fe_low_complexity
