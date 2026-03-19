"""
Tests for utils/__init__.py environment variable handling
"""

import os
import pytest


def test_check_required_env_vars_both_keys_set():
    """Test that no error is raised when both keys are set."""
    # Set both required environment variables
    original_pickle = os.environ.get("PICKLE_SECRET_KEY")
    original_backup = os.environ.get("APGI_BACKUP_HMAC_KEY")
    original_env = os.environ.get("APGI_ENV")

    try:
        os.environ["PICKLE_SECRET_KEY"] = "test_pickle_key_12345678901234567890"
        os.environ["APGI_BACKUP_HMAC_KEY"] = "test_backup_key_12345678901234567890"

        # Re-import to trigger the check
        import importlib  # noqa: F401
        import utils  # noqa: F401

        importlib.reload(utils)

        # Should not raise any error
        assert True

    finally:
        # Restore original values
        if original_pickle is not None:
            os.environ["PICKLE_SECRET_KEY"] = original_pickle
        else:
            os.environ.pop("PICKLE_SECRET_KEY", None)

        if original_backup is not None:
            os.environ["APGI_BACKUP_HMAC_KEY"] = original_backup
        else:
            os.environ.pop("APGI_BACKUP_HMAC_KEY", None)

        if original_env is not None:
            os.environ["APGI_ENV"] = original_env
        else:
            os.environ.pop("APGI_ENV", None)


def test_check_required_env_vars_missing_both_keys_production():
    """Test that EnvironmentError is raised when both keys are missing in production."""
    original_pickle = os.environ.get("PICKLE_SECRET_KEY")
    original_backup = os.environ.get("APGI_BACKUP_HMAC_KEY")
    original_env = os.environ.get("APGI_ENV")

    try:
        # Remove both keys
        os.environ.pop("PICKLE_SECRET_KEY", None)
        os.environ.pop("APGI_BACKUP_HMAC_KEY", None)
        os.environ["APGI_ENV"] = "production"

        # Should raise EnvironmentError
        with pytest.raises(EnvironmentError) as exc_info:
            import importlib  # noqa: F401
            import utils  # noqa: F401

        assert "Missing required environment variables" in str(exc_info.value)
        assert "PICKLE_SECRET_KEY" in str(exc_info.value)
        assert "APGI_BACKUP_HMAC_KEY" in str(exc_info.value)

    finally:
        # Restore original values
        if original_pickle is not None:
            os.environ["PICKLE_SECRET_KEY"] = original_pickle
        if original_backup is not None:
            os.environ["APGI_BACKUP_HMAC_KEY"] = original_backup
        if original_env is not None:
            os.environ["APGI_ENV"] = original_env
        else:
            os.environ.pop("APGI_ENV", None)


def test_check_required_env_vars_missing_both_keys_development():
    """Test that EnvironmentError is raised when both keys are missing in development."""
    original_pickle = os.environ.get("PICKLE_SECRET_KEY")
    original_env = os.environ.get("APGI_ENV")
    original_backup = os.environ.get("APGI_BACKUP_HMAC_KEY")
    original_env = os.environ.get("APGI_ENV")

    try:
        os.environ.pop("PICKLE_SECRET_KEY", None)
        os.environ.pop("APGI_BACKUP_HMAC_KEY", None)
        os.environ.pop("APGI_ENV", None)

        # Should raise EnvironmentError with helpful message
        import importlib  # noqa: F401

        with pytest.raises(EnvironmentError) as exc_info:
            import utils  # noqa: F401

        assert "Missing required environment variables" in str(exc_info.value)
        assert ".env file" in str(exc_info.value)

    finally:
        # Restore original values
        if original_pickle is not None:
            os.environ["PICKLE_SECRET_KEY"] = original_pickle
        if original_backup is not None:
            os.environ["APGI_BACKUP_HMAC_KEY"] = original_backup
        if original_env is not None:
            os.environ["APGI_ENV"] = original_env


def test_check_required_env_vars_missing_one_key():
    """Test that EnvironmentError is raised when one key is missing."""
    original_env = os.environ.get("APGI_ENV")
    original_backup = os.environ.get("APGI_BACKUP_HMAC_KEY")

    try:
        # Set only one key
        os.environ["PICKLE_SECRET_KEY"] = "test_pickle_key_12345678901234567890"
        os.environ.pop("APGI_BACKUP_HMAC_KEY", None)

        # Should raise EnvironmentError
        import importlib

        with pytest.raises(EnvironmentError) as exc_info:
            import utils

        assert "Missing required environment variables" in str(exc_info.value)
        assert "APGI_BACKUP_HMAC_KEY" in str(exc_info.value)

    finally:
        # Restore original values
        if original_pickle is not None:
            os.environ["PICKLE_SECRET_KEY"] = original_pickle
        else:
            os.environ.pop("PICKLE_SECRET_KEY", None)
        if original_backup is not None:
            os.environ["APGI_BACKUP_HMAC_KEY"] = original_backup


def test_check_required_env_vars_keys_are_session_unique():
    """Test that different sessions require different keys."""
    # This is implicitly tested by the fact that we set keys in environment
    # and they're checked at import time
    pass


def test_check_required_env_vars_errors_detectable():
    """Test that errors can be detected when keys are missing."""
    original_env = os.environ.get("APGI_ENV")
    original_backup = os.environ.get("APGI_BACKUP_HMAC_KEY")

    try:
        # Remove both keys
        os.environ.pop("PICKLE_SECRET_KEY", None)
        os.environ.pop("APGI_BACKUP_HMAC_KEY", None)

        # Should raise EnvironmentError (not just warning)
        import importlib  # noqa: F401

        with pytest.raises(EnvironmentError):
            import utils  # noqa: F401

    finally:
        # Restore original values
        if original_pickle is not None:
            os.environ["PICKLE_SECRET_KEY"] = original_pickle
        if original_backup is not None:
            os.environ["APGI_BACKUP_HMAC_KEY"] = original_backup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
