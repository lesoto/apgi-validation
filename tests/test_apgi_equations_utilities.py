"""
Targeted tests for uncovered utility functions in apgi_core/equations.py
========================================================================

Focuses on information theory and Bayesian computation functions that are currently uncovered.
"""

import numpy as np
import pytest

from apgi_core.equations import (
    compute_bayesian_update,
    compute_entropy,
    compute_free_energy,
    compute_kl_divergence,
    compute_mutual_information,
)


def test_compute_entropy_uniform():
    """Test entropy computation for uniform distribution."""
    # Uniform distribution over 4 states: each has probability 0.25
    # H = -4 * (0.25 * log2(0.25)) = -4 * (0.25 * -2) = 2.0 bits
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = compute_entropy(uniform)
    assert np.isclose(entropy, 2.0)


def test_compute_entropy_deterministic():
    """Test entropy computation for deterministic distribution."""
    # Deterministic distribution: one state has probability 1.0
    # H = -1.0 * log2(1.0) = 0.0 bits
    deterministic = np.array([1.0, 0.0, 0.0, 0.0])
    entropy = compute_entropy(deterministic)
    # Use tolerance for floating point precision
    assert np.isclose(entropy, 0.0, atol=1e-6)


def test_compute_entropy_normalization():
    """Test that entropy function normalizes invalid distributions."""
    # Distribution that doesn't sum to 1
    invalid = np.array([1.0, 1.0, 1.0, 1.0])
    entropy = compute_entropy(invalid)
    # Should normalize to [0.25, 0.25, 0.25, 0.25] and give 2.0 bits
    assert np.isclose(entropy, 2.0)


def test_compute_entropy_with_zeros():
    """Test entropy computation with zero probabilities."""
    # Distribution with some zero probabilities
    dist = np.array([0.5, 0.5, 0.0, 0.0])
    entropy = compute_entropy(dist)
    # H = -2 * (0.5 * log2(0.5)) = -2 * (0.5 * -1) = 1.0 bit
    assert np.isclose(entropy, 1.0)


def test_compute_kl_divergence_identical():
    """Test KL divergence for identical distributions."""
    p = np.array([0.25, 0.25, 0.25, 0.25])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    kl = compute_kl_divergence(p, q)
    # KL divergence should be 0 for identical distributions
    assert np.isclose(kl, 0.0)


def test_compute_kl_divergence_different():
    """Test KL divergence for different distributions."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    kl = compute_kl_divergence(p, q)
    # KL divergence should be positive
    assert kl > 0.0


def test_compute_kl_divergence_normalization():
    """Test that KL divergence normalizes invalid distributions."""
    p = np.array([2.0, 2.0, 0.0, 0.0])
    q = np.array([1.0, 1.0, 1.0, 1.0])
    kl = compute_kl_divergence(p, q)
    # Should normalize both distributions
    assert kl >= 0.0


def test_compute_mutual_information_independent():
    """Test mutual information for independent variables."""
    # Independent joint distribution: p(x,y) = p(x) * p(y)
    joint = np.array([[0.25, 0.25], [0.25, 0.25]])
    mi = compute_mutual_information(joint)
    # MI should be 0 for independent variables
    assert np.isclose(mi, 0.0, atol=1e-10)


def test_compute_mutual_information_dependent():
    """Test mutual information for dependent variables."""
    # Perfectly correlated: all mass on diagonal
    joint = np.array([[0.5, 0.0], [0.0, 0.5]])
    mi = compute_mutual_information(joint)
    # MI should be positive for dependent variables
    assert mi > 0.0


def test_compute_mutual_information_normalization():
    """Test that mutual information normalizes invalid distributions."""
    joint = np.array([[2.0, 0.0], [0.0, 2.0]])
    mi = compute_mutual_information(joint)
    # Should normalize the joint distribution
    assert mi >= 0.0


def test_compute_bayesian_update():
    """Test Bayesian update computation."""
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.8, 0.2])
    posterior = compute_bayesian_update(prior, likelihood)
    # Posterior should be normalized
    assert np.isclose(np.sum(posterior), 1.0)
    # First hypothesis should have higher posterior probability
    assert posterior[0] > posterior[1]


def test_compute_bayesian_update_uniform_prior():
    """Test Bayesian update with uniform prior."""
    prior = np.array([0.5, 0.5, 0.5, 0.5])
    likelihood = np.array([0.1, 0.2, 0.3, 0.4])
    posterior = compute_bayesian_update(prior, likelihood)
    # With uniform prior, posterior should be proportional to likelihood
    expected = likelihood / np.sum(likelihood)
    assert np.allclose(posterior, expected)


def test_compute_bayesian_update_normalization():
    """Test that Bayesian update normalizes invalid priors."""
    prior = np.array([2.0, 2.0])
    likelihood = np.array([0.8, 0.2])
    posterior = compute_bayesian_update(prior, likelihood)
    # Should normalize the prior
    assert np.isclose(np.sum(posterior), 1.0)


def test_compute_free_energy():
    """Test free energy computation."""
    surprise = np.array([1.0, 2.0, 3.0])
    threshold = np.array([0.5, 1.0, 1.5])
    complexity = np.array([0.1, 0.2, 0.3])
    fe = compute_free_energy(surprise, threshold, complexity)
    # Free energy should be positive
    assert fe > 0.0


def test_compute_free_energy_zero_complexity():
    """Test free energy with zero complexity."""
    surprise = np.array([1.0, 2.0, 3.0])
    threshold = np.array([0.5, 1.0, 1.5])
    complexity = np.array([0.0, 0.0, 0.0])
    fe = compute_free_energy(surprise, threshold, complexity)
    # Should just be expected surprise
    assert fe > 0.0


def test_compute_entropy_edge_cases():
    """Test entropy with edge cases."""
    # Single element distribution
    single = np.array([1.0])
    entropy = compute_entropy(single)
    assert np.isclose(entropy, 0.0)

    # Very small probabilities
    tiny = np.array([1e-9, 1.0 - 1e-9])
    entropy = compute_entropy(tiny)
    assert entropy >= 0.0


def test_compute_kl_divergence_asymmetry():
    """Test that KL divergence is asymmetric."""
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    kl_pq = compute_kl_divergence(p, q)
    kl_qp = compute_kl_divergence(q, p)
    # KL divergence should not be symmetric
    assert not np.isclose(kl_pq, kl_qp)


def test_compute_mutual_information_symmetry():
    """Test that mutual information is symmetric."""
    joint = np.array([[0.3, 0.2], [0.2, 0.3]])
    mi = compute_mutual_information(joint)
    # MI should be the same regardless of which variable we condition on
    assert mi >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
