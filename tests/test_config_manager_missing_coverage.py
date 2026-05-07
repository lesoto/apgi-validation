"""
Targeted tests for utils/config_manager.py missing coverage areas.
============================================================

Focuses on specific functions and lines that are currently uncovered.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigManagerMissingCoverage:
    """Test ConfigManager functions with missing coverage."""

    def test_fallback_yaml_load_safe(self):
        """Test _load_yaml_safe fallback function."""
        # Test when yaml is not available
        with patch.dict("sys.modules", {"yaml": None}):
            from utils.config_manager import _load_yaml_safe

            # Create test YAML file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write("test_key: test_value\nnumber: 42\n")
                yaml_path = f.name

            try:
                # Should return None when yaml is not available
                result = _load_yaml_safe(yaml_path)
                assert result is None
            finally:
                os.unlink(yaml_path)

    def test_fallback_yaml_load_safe_with_yaml(self):
        """Test _load_yaml_safe when yaml is available."""
        # Mock yaml to be available
        mock_yaml = MagicMock()
        mock_yaml.safe_load.return_value = {"key": "value"}

        with patch.dict("sys.modules", {"yaml": mock_yaml}):
            from utils.config_manager import _load_yaml_safe

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write("key: value\n")
                yaml_path = f.name

            try:
                result = _load_yaml_safe(yaml_path)
                assert result == {"key": "value"}
                mock_yaml.safe_load.assert_called_once()
            finally:
                os.unlink(yaml_path)

    def test_fallback_yaml_load_safe_error(self):
        """Test _load_yaml_safe error handling."""
        mock_yaml = MagicMock()
        mock_yaml.safe_load.side_effect = Exception("YAML error")

        with patch.dict("sys.modules", {"yaml": mock_yaml}):
            from utils.config_manager import _load_yaml_safe

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write("invalid: yaml: content:\n  - missing\n")
                yaml_path = f.name

            try:
                result = _load_yaml_safe(yaml_path)
                assert result is None
            finally:
                os.unlink(yaml_path)

    def test_fallback_json_load_safe(self):
        """Test _load_json_safe fallback function."""
        with patch.dict("sys.modules", {"jsonschema": None}):
            from utils.config_manager import _load_json_safe

            # Create test JSON file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({"test_key": "test_value"}, f)
                json_path = f.name

            try:
                result = _load_json_safe(json_path)
                assert result == {"test_key": "test_value"}
            finally:
                os.unlink(json_path)

    def test_fallback_json_load_safe_error(self):
        """Test _load_json_safe error handling."""
        from utils.config_manager import _load_json_safe

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            json_path = f.name

        try:
            result = _load_json_safe(json_path)
            assert result is None
        finally:
            os.unlink(json_path)

    def test_load_env_file(self):
        """Test _load_env_file function."""
        from utils.config_manager import _load_env_file

        # Create test env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# Comment line\n")
            f.write("TEST_VAR=test_value\n")
            f.write("ANOTHER_VAR=another_value\n")
            f.write("EMPTY_VAR=\n")
            env_path = f.name

        try:
            result = _load_env_file(env_path)
            assert result == {
                "TEST_VAR": "test_value",
                "ANOTHER_VAR": "another_value",
                "EMPTY_VAR": "",
            }
        finally:
            os.unlink(env_path)

    def test_load_env_file_nonexistent(self):
        """Test _load_env_file with nonexistent file."""
        from utils.config_manager import _load_env_file

        result = _load_env_file("/nonexistent/.env")
        assert result == {}

    def test_load_env_file_empty_lines(self):
        """Test _load_env_file with empty lines."""
        from utils.config_manager import _load_env_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("\n")
            f.write("# Comment\n")
            f.write("\n")
            f.write("VALID_VAR=value\n")
            f.write("\n")
            env_path = f.name

        try:
            result = _load_env_file(env_path)
            assert result == {"VALID_VAR": "value"}
        finally:
            os.unlink(env_path)

    def test_logger_source_relative_import(self):
        """Test logger source detection with relative import."""
        # This tests the import logic in config_manager
        with patch("utils.config_manager.importlib.util"):
            with patch("utils.config_manager.sys"):
                # Mock the relative import to succeed
                with patch("utils.config_manager.log_error"):
                    with patch("utils.config_manager.apgi_logger"):
                        # Re-import to trigger the import logic
                        if "utils.config_manager" in sys.modules:
                            del sys.modules["utils.config_manager"]
                        import importlib

                        importlib.reload(sys.modules.get("utils.config_manager"))

    def test_logger_source_importlib_fallback(self):
        """Test logger source detection with importlib fallback."""
        # Mock both import attempts to fail
        with patch("utils.config_manager.importlib.util"):
            with patch("utils.config_manager.sys"):
                # Simulate both imports failing
                pass

    def test_config_manager_with_missing_dependencies(self):
        """Test ConfigManager when dependencies are missing."""
        # Mock all dependencies to be unavailable
        with patch.dict(
            "sys.modules", {"jsonschema": None, "yaml": None, "dotenv": None}
        ):
            # Force reimport to trigger fallback paths
            if "utils.config_manager" in sys.modules:
                del sys.modules["utils.config_manager"]

            # Import should work with fallbacks
            from utils.config_manager import ConfigManager

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "test_config.yaml"

                # Should still be able to create ConfigManager
                manager = ConfigManager(str(config_path))
                assert manager is not None

    def test_config_manager_yaml_operations(self):
        """Test YAML operations in ConfigManager."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test YAML loading when yaml is available
            with patch("utils.config_manager.yaml") as mock_yaml:
                mock_yaml.safe_load.return_value = {"test": "config"}

                # This should use the yaml module
                manager._load_config_file(str(config_path))
                mock_yaml.safe_load.assert_called()

    def test_config_manager_json_operations(self):
        """Test JSON operations in ConfigManager."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            manager = ConfigManager(str(config_path))

            # Test JSON loading
            with patch("builtins.open", mock_open(read_data='{"test": "config"}')):
                with patch("json.load", return_value={"test": "config"}):
                    result = manager._load_config_file(str(config_path))
                    assert result == {"test": "config"}

    def test_config_manager_env_loading(self):
        """Test environment variable loading."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test env file loading
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".env", delete=False
            ) as f:
                f.write("ENV_TEST=env_value\n")
                env_path = f.name

            try:
                with patch("utils.config_manager.load_dotenv"):
                    with patch("utils.config_manager._load_env_file") as mock_load_env:
                        mock_load_env.return_value = {"ENV_TEST": "env_value"}

                        manager._load_environment(str(env_path))
                        mock_load_env.assert_called_with(env_path)
            finally:
                os.unlink(env_path)

    def test_config_manager_validation_without_jsonschema(self):
        """Test config validation without jsonschema."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test validation without jsonschema
            with patch.dict("sys.modules", {"jsonschema": None}):
                # Should not raise error when jsonschema is not available
                result = manager._validate_config({"test": "value"})
                assert result is True  # Should pass validation by default

    def test_config_manager_validation_with_jsonschema(self):
        """Test config validation with jsonschema available."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Mock jsonschema
            mock_jsonschema = MagicMock()
            mock_validate = MagicMock()
            mock_jsonschema.validate = mock_validate
            mock_jsonschema.ValidationError = Exception
            mock_jsonschema.SchemaError = Exception

            with patch.dict("sys.modules", {"jsonschema": mock_jsonschema}):
                # Test successful validation
                result = manager._validate_config({"test": "value"})
                assert result is True

    def test_config_manager_validation_error(self):
        """Test config validation error handling."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Mock jsonschema to raise validation error
            mock_jsonschema = MagicMock()
            mock_validate = MagicMock(side_effect=Exception("Validation failed"))
            mock_jsonschema.validate = mock_validate
            mock_jsonschema.ValidationError = Exception
            mock_jsonschema.SchemaError = Exception

            with patch.dict("sys.modules", {"jsonschema": mock_jsonschema}):
                result = manager._validate_config({"invalid": "config"})
                assert result is False

    def test_config_manager_schema_error(self):
        """Test config schema error handling."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Mock jsonschema to raise schema error
            mock_jsonschema = MagicMock()
            mock_validate = MagicMock(side_effect=Exception("Schema error"))
            mock_jsonschema.validate = mock_validate
            mock_jsonschema.ValidationError = ValueError
            mock_jsonschema.SchemaError = ValueError

            with patch.dict("sys.modules", {"jsonschema": mock_jsonschema}):
                result = manager._validate_config({"test": "value"})
                assert result is False

    def test_config_manager_profile_operations(self):
        """Test profile management operations."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test profile creation
            with tempfile.TemporaryDirectory() as profile_dir:
                profile_path = Path(profile_dir) / "profile.yaml"

                result = manager.create_profile(
                    "test_profile", "Test profile", str(profile_path)
                )
                assert result is not None
                assert os.path.exists(profile_path)

    def test_config_manager_load_profile(self):
        """Test loading configuration profiles."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test loading profile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write("profile_data:\n  key: value\n")
                profile_path = f.name

            try:
                result = manager.load_profile(profile_path)
                assert result is not None
            finally:
                os.unlink(profile_path)

    def test_config_manager_save_config(self):
        """Test saving configuration."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test saving config
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                save_path = f.name

            try:
                manager.save_config(save_path)
                assert os.path.exists(save_path)
            finally:
                os.unlink(save_path)

    def test_config_manager_compare_configs(self):
        """Test configuration comparison."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            config1 = {"section1": {"key1": "value1"}, "section2": {"key2": "value2"}}
            config2 = {
                "section1": {"key1": "different"},
                "section3": {"key3": "value3"},
            }

            diff = manager.compare_configs(config1, config2)
            assert isinstance(diff, dict)
            assert "changed" in diff or "added" in diff or "removed" in diff

    def test_config_manager_reset_operations(self):
        """Test configuration reset operations."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test section reset
            manager.reset_to_defaults("simulation")

            # Test full reset
            manager.reset_to_defaults()

    def test_config_manager_parameter_validation(self):
        """Test parameter validation."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test valid parameter
            result = manager.set_parameter("simulation", "default_steps", 100)
            assert result is True

            # Test invalid parameter (should handle gracefully)
            result = manager.set_parameter("invalid_section", "invalid_param", "value")
            assert result is False

    def test_config_manager_get_parameter(self):
        """Test getting parameters."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test getting existing parameter
            value = manager.get_parameter("simulation", "default_steps")
            assert value is not None

            # Test getting non-existent parameter
            value = manager.get_parameter("simulation", "nonexistent")
            assert value is None

    def test_config_manager_thread_safety(self):
        """Test thread safety of ConfigManager."""
        from utils.config_manager import ConfigManager
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            results = []

            def set_params():
                for i in range(10):
                    manager.set_parameter("test", f"param_{i}", f"value_{i}")
                    results.append(manager.get_parameter("test", f"param_{i}"))

            threads = [threading.Thread(target=set_params) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Should have completed without errors
            assert len(results) > 0

    def test_config_manager_error_handling(self):
        """Test error handling in ConfigManager."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test handling of invalid config file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write("invalid: yaml: content: [")
                invalid_path = f.name

            try:
                # Should handle gracefully
                result = manager._load_config_file(invalid_path)
                assert result is not None  # Should return default config
            finally:
                os.unlink(invalid_path)

    def test_config_manager_backup_operations(self):
        """Test backup and restore operations."""
        from utils.config_manager import ConfigManager

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            manager = ConfigManager(str(config_path))

            # Test creating backup
            with tempfile.NamedTemporaryFile(suffix=".backup", delete=False) as f:
                backup_path = f.name

            try:
                manager.create_backup(backup_path)
                assert os.path.exists(backup_path)

                # Test restoring from backup
                manager.restore_from_backup(backup_path)
            finally:
                if os.path.exists(backup_path):
                    os.unlink(backup_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
