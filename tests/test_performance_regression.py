"""
Performance regression tests with @pytest.mark.performance.
Tests for performance-critical operations to detect regressions.
=============================================================
"""

import pytest
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPerformanceRegression:
    """Performance regression tests for critical operations."""

    @pytest.mark.performance
    def test_entropy_computation_performance(self):
        """Test entropy computation performance should not regress."""
        from APGI_Equations import compute_entropy

        # Create test data
        distribution = np.random.rand(1000)
        distribution = distribution / np.sum(distribution)

        # Measure execution time
        start_time = time.time()
        for _ in range(100):
            compute_entropy(distribution)
        elapsed = time.time() - start_time

        # Should complete 100 iterations in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Entropy computation too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_kl_divergence_computation_performance(self):
        """Test KL divergence computation performance."""
        from APGI_Equations import compute_kl_divergence

        p = np.random.rand(100)
        p = p / np.sum(p)
        q = np.random.rand(100)
        q = q / np.sum(q)

        start_time = time.time()
        for _ in range(100):
            compute_kl_divergence(p, q)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"KL divergence computation too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_mutual_information_computation_performance(self):
        """Test mutual information computation performance."""
        from APGI_Equations import compute_mutual_information

        joint = np.random.rand(10, 10)
        joint = joint / np.sum(joint)

        start_time = time.time()
        for _ in range(100):
            compute_mutual_information(joint)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Mutual information computation too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_bayesian_update_performance(self):
        """Test Bayesian update performance."""
        from APGI_Equations import compute_bayesian_update

        prior = np.random.rand(100)
        prior = prior / np.sum(prior)
        likelihood = np.random.rand(100)
        likelihood = likelihood / np.sum(likelihood)

        start_time = time.time()
        for _ in range(100):
            compute_bayesian_update(prior, likelihood)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Bayesian update too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_free_energy_computation_performance(self):
        """Test free energy computation performance."""
        from APGI_Equations import compute_free_energy

        surprise = np.random.rand(100)
        threshold = np.random.rand(100)
        complexity = np.random.rand(100)

        start_time = time.time()
        for _ in range(100):
            compute_free_energy(surprise, threshold, complexity)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Free energy computation too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_data_loading_performance(self):
        """Test data loading performance."""
        import pandas as pd
        import tempfile

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_file = f.name
            # Write 1000 rows
            f.write("col1,col2,col3\n")
            for i in range(1000):
                f.write(f"{i}, {i * 2}, {i * 3}\n")

        try:
            start_time = time.time()
            for _ in range(10):
                pd.read_csv(temp_file)
            elapsed = time.time() - start_time

            assert elapsed < 2.0, f"Data loading too slow: {elapsed:.3f}s"
        finally:
            import os

            os.unlink(temp_file)

    @pytest.mark.performance
    def test_matrix_operations_performance(self):
        """Test matrix operations performance."""
        import numpy as np

        # Large matrix operations
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)

        start_time = time.time()
        for _ in range(100):
            np.dot(A, B)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Matrix multiplication too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_file_io_performance(self):
        """Test file I/O performance."""
        import tempfile
        import json

        # Create test data
        test_data = {"key": list(range(1000))}

        start_time = time.time()
        for _ in range(100):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=True
            ) as f:
                json.dump(test_data, f)
                f.flush()
        elapsed = time.time() - start_time

        assert elapsed < 5.0, f"File I/O too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_random_number_generation_performance(self):
        """Test random number generation performance."""
        start_time = time.time()
        for _ in range(100):
            np.random.rand(1000)
        elapsed = time.time() - start_time

        assert elapsed < 0.5, f"Random number generation too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_statistical_computations_performance(self):
        """Test statistical computations performance."""
        data = np.random.randn(10000)

        start_time = time.time()
        for _ in range(100):
            np.mean(data)
            np.std(data)
            np.median(data)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Statistical computations too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_array_sorting_performance(self):
        """Test array sorting performance."""
        data = np.random.rand(10000)

        start_time = time.time()
        for _ in range(100):
            np.sort(data)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Array sorting too slow: {elapsed:.3f}s"

    @pytest.mark.performance
    def test_string_operations_performance(self):
        """Test string operations performance."""
        strings = [f"test_string_{i}" for i in range(1000)]

        start_time = time.time()
        for _ in range(100):
            "_".join(strings)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"String operations too slow: {elapsed:.3f}s"
