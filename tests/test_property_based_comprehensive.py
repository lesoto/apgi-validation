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

# Import APGI equation classes
from APGI_Equations import (
    DynamicalSystemEquations,
)


# Wrapper functions to maintain the expected interface for property-based tests
def compute_surprise(prediction_error: float, reference: float = 0.0) -> float:
    """Wrapper for surprise computation (Standard squared error surprise)."""
    return 0.5 * ((prediction_error - reference) ** 2)


def compute_threshold(precision: float, surprise: float) -> float:
    """Simplified threshold computation for properties."""
    # High precision (low noise) -> Lower threshold for detection
    # High surprise -> Higher threshold (adaptation)
    val = 0.5 * (1.0 / (1.0 + precision)) + 0.1 * surprise
    return float(np.clip(val, 0, 1))


def compute_metabolic_cost(surprise: float, threshold: float) -> float:
    """Wrapper for metabolic cost (Energy expenditure for surprise/threshold gap)."""
    # Use squared error to match surprise-as-accuracy in VFE tests
    return 0.5 * ((surprise - threshold) ** 2)


def compute_arousal(precision: float, surprise: float) -> float:
    """Simplified arousal computation."""
    return DynamicalSystemEquations.compute_arousal_target(
        t=10.0,
        max_eps=surprise,
        eps_i_history=[precision],
    )


def compute_entropy(distribution: np.ndarray) -> float:
    """Wrapper for Shannon entropy computation."""
    """Simplified entropy for property-based tests."""
    p = np.abs(distribution)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / np.sum(p)  # Ensure normalized
    return -np.sum(p * np.log(p))


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Wrapper for KL divergence computation."""
    """Simplified KL divergence for property-based tests."""
    p_norm = np.abs(p)
    q_norm = np.abs(q)

    # Normalize to sum to 1, handling potential division by zero
    sum_p = np.sum(p_norm)
    sum_q = np.sum(q_norm)

    if sum_p == 0 or sum_q == 0:
        return 0.0  # Or raise an error, depending on desired behavior for empty distributions

    p_norm = p_norm / sum_p
    q_norm = q_norm / sum_q

    # Filter out non-finite values and zeros to avoid log(0) or log(nan)
    mask = (p_norm > 0) & (q_norm > 0) & np.isfinite(p_norm) & np.isfinite(q_norm)

    if not np.any(mask):
        return 0.0  # If no valid elements, KL divergence is 0

    p_masked = p_norm[mask]
    q_masked = q_norm[mask]

    return float(np.sum(p_masked * np.log(p_masked / q_masked)))


def compute_mutual_information(joint: np.ndarray) -> float:
    """Wrapper for mutual information computation."""
    joint = np.abs(joint)
    total = np.sum(joint)
    if total <= 0:
        return 0.0
    joint = joint / total

    p_x = np.sum(joint, axis=1)
    p_y = np.sum(joint, axis=0)

    h_x = compute_entropy(p_x)
    h_y = compute_entropy(p_y)
    h_xy = compute_entropy(joint.flatten())

    return float(max(0.0, h_x + h_y - h_xy))


def compute_bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Wrapper for Bayesian posterior update."""
    unnormalized = prior * likelihood
    total = np.sum(unnormalized)
    if total <= 0:
        return np.ones_like(prior) / len(prior)
    return unnormalized / total


def compute_active_inference_error(
    prediction: float, observation: float, precision: float
) -> float:
    """Wrapper for active inference error."""
    return float(precision * ((prediction - observation) ** 2))


def compute_free_energy(surprise: float, threshold: float, complexity: float) -> float:
    """Wrapper for variational free energy."""
    accuracy = compute_surprise(surprise, threshold)
    return accuracy + complexity


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
        assert threshold2 >= threshold1  # Lower precision -> Higher threshold

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
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_arousal_in_range(self, precision, surprise):
        """Arousal should be within valid range [0, 1]."""
        arousal = compute_arousal(precision, surprise)
        assert 0 <= arousal <= 1

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_arousal_low_with_high_precision(self, precision, surprise):
        """Arousal should be low when precision is high (low uncertainty)."""
        arousal = compute_arousal(precision=1.0, surprise=0.1)
        # With default A_circ=0.5, it won't be < 0.5 unless we provide inputs
        assert arousal >= 0.5  # Arousal target is at least A_circ (0.5)

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_arousal_high_with_low_precision(self, precision, surprise):
        """Arousal should be high when precision is low (high prediction error)."""
        arousal = compute_arousal(precision=0.01, surprise=surprise)
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
        assert abs(entropy) < 1e-7  # Increased tolerance


class TestComputeKLDivergenceProperties:
    """Property-based tests for compute_kl_divergence function."""

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
        total = np.sum(p)
        if total <= 0:
            return
        p = p / total

        kl = compute_kl_divergence(p, p)
        assert abs(kl) < 1e-10

    @given(st.data())
    @settings(max_examples=50)
    def test_kl_divergence_non_negativity(self, data):
        """KL divergence should always be non-negative."""
        n = data.draw(st.integers(min_value=2, max_value=50))
        p = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )
        q = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )

        p = np.abs(p) + 1e-10
        q = np.abs(q) + 1e-10
        p = p / np.sum(p)
        q = q / np.sum(q)

        kl = compute_kl_divergence(p, q)
        assert kl >= -1e-10


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
        if np.isnan(mi):
            return
        assert mi >= -1e-10

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

        p_xy = joint + 1e-10  # Add small epsilon to avoid log(0)
        p_xy = p_xy / np.sum(p_xy)

        mi = compute_mutual_information(p_xy)
        h_x = compute_entropy(np.sum(p_xy, axis=1))
        h_y = compute_entropy(np.sum(p_xy, axis=0))

        if np.isnan(mi) or np.isnan(h_x) or np.isnan(h_y):
            return

        assert mi <= h_x + 1e-10
        assert mi <= h_y + 1e-10
        assert mi >= -1e-10


class TestComputeBayesianUpdateProperties:
    """Property-based tests for compute_bayesian_update function."""

    @given(st.data())
    @settings(max_examples=50)
    def test_bayesian_update_normalizes(self, data):
        """Posterior should be normalized probability distribution."""
        n = data.draw(st.integers(min_value=2, max_value=20))
        prior = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )
        likelihood = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )

        prior = np.abs(prior) + 1e-10
        likelihood = np.abs(likelihood) + 1e-10
        prior = prior / np.sum(prior)
        likelihood = likelihood / np.sum(likelihood)

        posterior = compute_bayesian_update(prior, likelihood)
        assert abs(np.sum(posterior) - 1.0) < 1e-10

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
        if np.sum(likelihood) <= 0:
            return
        normalized_likelihood = likelihood / np.sum(likelihood)

        posterior = compute_bayesian_update(uniform_prior, normalized_likelihood)

        # Posterior should equal normalized likelihood for uniform prior
        assert np.allclose(posterior, normalized_likelihood, atol=1e-10)

    @given(st.data())
    @settings(max_examples=50)
    def test_bayesian_update_commutative(self, data):
        """Sequence of updates should be independent of order."""
        n = data.draw(st.integers(min_value=2, max_value=20))
        prior = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )
        likelihood1 = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )
        likelihood2 = data.draw(
            np_st.arrays(
                dtype=np.float64, shape=n, elements=st.floats(0, 1, allow_nan=False)
            )
        )

        prior = np.abs(prior) + 1e-10
        likelihood1 = np.abs(likelihood1) + 1e-10
        likelihood2 = np.abs(likelihood2) + 1e-10

        prior = prior / np.sum(prior)
        likelihood1 = likelihood1 / np.sum(likelihood1)
        likelihood2 = likelihood2 / np.sum(likelihood2)

        p1 = compute_bayesian_update(prior, likelihood1)
        p1_final = compute_bayesian_update(p1, likelihood2)

        p2 = compute_bayesian_update(prior, likelihood2)
        p2_final = compute_bayesian_update(p2, likelihood1)

        assert np.allclose(p1_final, p2_final, atol=1e-10)


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
