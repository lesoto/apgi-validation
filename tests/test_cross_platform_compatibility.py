"""
Cross-Platform Compatibility Tests
===================================

Tests for ensuring APGI Validation Framework works across
different platforms and environments.
"""

import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path


class TestPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility."""

    def setUp(self):
        """Set up platform information."""
        self.system = platform.system()
        self.python_version = sys.version_info

    def test_python_version_requirement(self):
        """Test that Python version meets minimum requirements."""
        # Require Python 3.10+
        self.assertGreaterEqual(self.python_version.major, 3)
        if self.python_version.major == 3:
            self.assertGreaterEqual(self.python_version.minor, 10)

    def test_path_handling(self):
        """Test path handling across platforms."""
        # Test Path objects work correctly
        test_path = Path("data") / "test" / "file.txt"
        self.assertIsInstance(test_path, Path)
        # Should handle separators correctly
        self.assertIn("data", str(test_path))
        self.assertIn("test", str(test_path))

    def test_file_permissions(self):
        """Test file permission handling."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            # Test file exists and is readable
            self.assertTrue(os.path.exists(temp_path))
            self.assertTrue(os.access(temp_path, os.R_OK))

            # Test file is writable
            self.assertTrue(os.access(temp_path, os.W_OK))
        finally:
            os.unlink(temp_path)

    def test_line_endings(self):
        """Test handling of different line endings."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="") as f:
            f.write("line1\nline2\r\nline3\r")
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                content = f.read()
            # Should read all lines regardless of ending
            self.assertIn("line1", content)
            self.assertIn("line2", content)
            self.assertIn("line3", content)
        finally:
            os.unlink(temp_path)

    def test_unicode_handling(self):
        """Test Unicode string handling."""
        test_strings = [
            "Hello, World!",
            "你好，世界",
            "مرحبا بالعالم",
            "🧠🔬📊",  # Emoji
            "αβγδ",  # Greek
        ]

        for s in test_strings:
            # Should be able to encode/decode
            encoded = s.encode("utf-8")
            decoded = encoded.decode("utf-8")
            self.assertEqual(s, decoded)


class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable handling."""

    def test_env_var_setting(self):
        """Test setting and getting environment variables."""
        test_var = "APGI_TEST_VAR"
        test_value = "test_value_123"

        os.environ[test_var] = test_value
        self.assertEqual(os.environ.get(test_var), test_value)

        # Clean up
        del os.environ[test_var]
        self.assertIsNone(os.environ.get(test_var))

    def test_env_var_case_sensitivity(self):
        """Test environment variable case sensitivity."""
        # Windows is case-insensitive, Unix is case-sensitive
        if platform.system() == "Windows":
            os.environ["TestVar"] = "value1"
            self.assertEqual(os.environ.get("TESTVAR"), "value1")
            del os.environ["TestVar"]
        else:
            os.environ["TestVar"] = "value1"
            os.environ["testvar"] = "value2"
            self.assertEqual(os.environ.get("TestVar"), "value1")
            self.assertEqual(os.environ.get("testvar"), "value2")
            del os.environ["TestVar"]
            del os.environ["testvar"]


class TestNumericPrecision(unittest.TestCase):
    """Test numeric precision across platforms."""

    def test_float_precision(self):
        """Test floating point precision."""
        import math

        # Standard double precision tests
        self.assertAlmostEqual(math.pi, 3.141592653589793, places=15)
        self.assertAlmostEqual(math.e, 2.718281828459045, places=15)

    def test_numpy_precision(self):
        """Test NumPy numeric precision."""
        import numpy as np

        # Test array operations maintain precision
        arr = np.array([1.0, 2.0, 3.0])
        result = np.sum(arr)
        self.assertAlmostEqual(result, 6.0, places=10)

    def test_large_numbers(self):
        """Test handling of large numbers."""
        import numpy as np

        large_val = 1e308
        self.assertTrue(np.isfinite(large_val))
        self.assertFalse(np.isinf(large_val))


class TestThreadingAndConcurrency(unittest.TestCase):
    """Test threading behavior across platforms."""

    def test_thread_creation(self):
        """Test basic thread creation."""
        import threading
        import time

        results = []

        def worker():
            time.sleep(0.01)
            results.append("done")

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 5)

    def test_lock_mechanism(self):
        """Test threading lock behavior."""
        import threading

        lock = threading.Lock()
        counter = [0]  # Use list for mutability

        def increment():
            with lock:
                counter[0] += 1

        threads = [threading.Thread(target=increment) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(counter[0], 100)


class TestSystemResources(unittest.TestCase):
    """Test system resource availability."""

    def test_memory_availability(self):
        """Test that sufficient memory is available."""
        import psutil

        mem = psutil.virtual_memory()
        # Should have at least 500MB available
        available_mb = mem.available / (1024 * 1024)
        self.assertGreater(available_mb, 500)

    def test_disk_space(self):
        """Test disk space availability."""
        import psutil

        disk = psutil.disk_usage("/")
        # Should have at least 100MB free
        free_mb = disk.free / (1024 * 1024)
        self.assertGreater(free_mb, 100)


class TestTimeAndDateHandling(unittest.TestCase):
    """Test time and date handling across platforms."""

    def test_timezone_awareness(self):
        """Test timezone handling."""
        from datetime import datetime, timezone

        utc_now = datetime.now(timezone.utc)
        self.assertIsNotNone(utc_now.tzinfo)

    def test_timestamp_precision(self):
        """Test timestamp precision."""
        import time

        t1 = time.time()
        time.sleep(0.001)  # 1ms
        t2 = time.time()

        diff = t2 - t1
        self.assertGreaterEqual(diff, 0.0005)  # At least 0.5ms
        self.assertLess(diff, 0.1)  # Less than 100ms


if __name__ == "__main__":
    unittest.main(verbosity=2)
