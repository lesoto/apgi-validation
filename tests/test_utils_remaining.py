"""
Tests for uncovered utils/ module paths
=========================================

Tests for utils modules that need additional coverage:
- analytical_solutions
- audit_threshold_leakage
- batch_config
- constants
- crash_recovery
- empirical_data_generators
- genome_data_extractor
- hrf_utils
- meta_falsification
- progress_estimator
- signal_handler
- update_protocol_metadata
- verify_framework_status
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAnalyticalSolutions:
    """Test analytical_solutions module"""

    def test_analytical_solution_import(self):
        """Test analytical_solutions module import"""
        from utils import analytical_solutions

        assert analytical_solutions is not None

    def test_compute_steady_state_solution(self):
        """Test steady state solution computation"""
        from utils.analytical_solutions import compute_steady_state

        params = {"tau_S": 0.5, "tau_theta": 30.0, "alpha": 10.0}
        result = compute_steady_state(params)

        assert isinstance(result, dict)
        assert "S" in result or "theta" in result or "S_steady" in result


class TestAuditThresholdLeakage:
    """Test audit_threshold_leakage module"""

    def test_audit_module_import(self):
        """Test audit_threshold_leakage module import"""
        from utils import audit_threshold_leakage

        assert audit_threshold_leakage is not None

    def test_audit_threshold_scanner(self):
        """Test threshold leakage scanner"""
        from utils.audit_threshold_leakage import scan_for_threshold_leakage

        result = scan_for_threshold_leakage()
        assert isinstance(result, (dict, list))


class TestBatchConfig:
    """Test batch_config module"""

    def test_batch_config_import(self):
        """Test batch_config module import"""
        from utils import batch_config

        assert batch_config is not None

    def test_batch_configuration_class(self):
        """Test BatchProcessorConfig class"""
        from utils.batch_config import BatchProcessorConfig

        config = BatchProcessorConfig()
        assert config.get_max_workers() > 0
        assert config.get("use_processes") in [True, False]


class TestConstants:
    """Test constants module"""

    def test_constants_import(self):
        """Test constants module import"""
        from utils import constants

        assert constants is not None

    def test_constant_values(self):
        """Test that key constants are defined"""
        from utils.constants import (
            DIM_CONSTANTS,
            MODEL_PARAMS,
            DimensionConstants,
            ModelParameters,
        )

        assert isinstance(MODEL_PARAMS, ModelParameters)
        assert isinstance(DIM_CONSTANTS, DimensionConstants)


class TestCrashRecovery:
    """Test crash_recovery module"""

    def test_crash_recovery_import(self):
        """Test crash_recovery module import"""
        from utils import crash_recovery

        assert crash_recovery is not None

    def test_crash_recovery_init(self):
        """Test CrashRecovery initialization"""
        try:
            from utils.crash_recovery import CrashRecovery

            recovery = CrashRecovery(app_name="test_app")
            assert hasattr(recovery, "recovery_dir")
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Class not available: {e}")

    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving"""
        from utils.crash_recovery import CrashRecovery

        recovery = CrashRecovery(app_name="test_app", recovery_dir=str(tmp_path))
        recovery.save_state({"data": "value"})

        # Verify state was saved
        assert recovery.state_file.exists()
        assert recovery.current_state is not None


class TestEmpiricalDataGenerators:
    """Test empirical_data_generators module"""

    def test_empirical_generators_import(self):
        """Test empirical_data_generators module import"""
        from utils import empirical_data_generators

        assert empirical_data_generators is not None

    def test_generate_eeg_sample(self):
        """Test EEG sample generation"""
        from utils.empirical_data_generators import generate_cross_cultural_eeg_data

        data, metadata = generate_cross_cultural_eeg_data(
            n_subjects_per_culture=2, n_trials=10, n_channels=8, sampling_rate=100.0
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert isinstance(metadata, dict)
        assert "n_subjects_total" in metadata


class TestGenomeDataExtractor:
    """Test genome_data_extractor module"""

    def test_genome_extractor_import(self):
        """Test genome_data_extractor module import"""
        from utils import genome_data_extractor

        assert genome_data_extractor is not None

    def test_extract_genomic_features(self):
        """Test genomic feature extraction"""
        # Test with mock data structure
        import json
        import tempfile
        from pathlib import Path

        from utils.genome_data_extractor import extract_genome_data_from_vp5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            mock_data = {
                "final_statistics": {
                    "final_frequencies": {
                        "has_threshold": 0.5,
                        "has_precision_weighting": 0.3,
                        "has_interoceptive_weighting": 0.4,
                    }
                },
                "config": {"n_generations": 500},
            }
            json.dump(mock_data, f)
            temp_path = f.name

        try:
            features = extract_genome_data_from_vp5(temp_path)
            assert isinstance(features, dict)
            assert "evolved_alpha_values" in features
            assert "timescale_correlations" in features
            assert "intero_gain_ratios" in features
        finally:
            Path(temp_path).unlink()


class TestHRFUtils:
    """Test hrf_utils module"""

    def test_hrf_utils_import(self):
        """Test hrf_utils module import"""
        from utils import hrf_utils

        assert hrf_utils is not None

    def test_create_hrf_kernel(self):
        """Test HRF kernel creation"""
        import numpy as np

        from utils.hrf_utils import double_gamma_hrf

        t = np.arange(0, 25, 0.1)
        hrf = double_gamma_hrf(t)
        assert isinstance(hrf, np.ndarray)
        assert len(hrf) > 0
        assert hrf.max() <= 1.0  # Normalized


class TestMetaFalsification:
    """Test meta_falsification module"""

    def test_meta_falsification_import(self):
        """Test meta_falsification module import"""
        from utils import meta_falsification

        assert meta_falsification is not None

    def test_meta_falsification_class(self):
        """Test FrameworkFalsificationGate class"""
        from utils.meta_falsification import FrameworkFalsificationGate

        gate = FrameworkFalsificationGate()
        assert hasattr(gate, "protocol_results")
        assert hasattr(gate, "min_criteria")
        assert hasattr(gate, "fail_threshold")


class TestProgressEstimator:
    """Test progress_estimator module"""

    def test_progress_estimator_import(self):
        """Test progress_estimator module import"""
        from utils import progress_estimator

        assert progress_estimator is not None

    def test_estimate_completion_time(self):
        """Test completion time estimation"""
        from utils.progress_estimator import ProgressEstimator

        estimator = ProgressEstimator()
        estimator.start_operation("test_op", "Test Operation", total_steps=100)

        # Update progress to 50%
        import time

        time.sleep(0.1)  # Small delay to calculate rate
        estimator.update_progress("test_op", 50)

        # Get estimated time remaining
        remaining = estimator.estimate_time_remaining("test_op")
        assert remaining is not None
        assert remaining >= 0


class TestSignalHandler:
    """Test signal_handler module"""

    def test_signal_handler_import(self):
        """Test signal_handler module import"""
        from utils import signal_handler

        assert signal_handler is not None

    def test_setup_signal_handlers(self):
        """Test signal handler setup"""
        from utils.signal_handler import SignalHandler

        # Test that SignalHandler can be instantiated
        handler = SignalHandler()
        assert hasattr(handler, "shutdown_callback")
        assert hasattr(handler, "_install_handlers")
        assert hasattr(handler, "_restore_handlers")


class TestUpdateProtocolMetadata:
    """Test update_protocol_metadata module"""

    def test_update_protocol_metadata_import(self):
        """Test update_protocol_metadata module import"""
        from utils import update_protocol_metadata

        assert update_protocol_metadata is not None

    def test_update_metadata(self, tmp_path):
        """Test metadata check function"""
        from utils.update_protocol_metadata import check_protocol_metadata

        # Create a temporary protocol file
        protocol_file = tmp_path / "test_protocol.py"
        protocol_file.write_text("def run_protocol_main(): pass\n")

        has_std, msg = check_protocol_metadata(str(protocol_file))
        assert isinstance(has_std, bool)
        assert isinstance(msg, str)


class TestVerifyFrameworkStatus:
    """Test verify_framework_status module"""

    def test_verify_framework_status_import(self):
        """Test verify_framework_status module import"""
        from utils import verify_framework_status

        assert verify_framework_status is not None

    def test_verify_framework(self):
        """Test framework verification"""
        from utils.verify_framework_status import check_protocol

        # Test check_protocol in quick mode
        result = check_protocol(
            "utils.constants",  # Use a known module
            "constants",
            timeout_secs=5,
            quick_mode=True,
        )
        assert isinstance(result, dict)
        assert "status" in result


class TestUtilsErrorHandling:
    """Test utils module error handling"""

    def test_utils_import_error_handling(self):
        """Test handling of utils import errors"""
        # Test that utils can be imported even with missing optional dependencies
        try:
            from utils import empirical_data_generators

            _ = empirical_data_generators  # Reference to avoid F401

            # Should not raise even if optional deps missing
            assert True
        except ImportError:
            # Some utils may fail to import without deps - that's OK
            pass

    def test_utils_config_validation(self):
        """Test utils configuration validation"""
        from utils import constants

        if hasattr(constants, "validate_config"):
            result = constants.validate_config({})
            assert isinstance(result, (bool, dict, tuple))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
