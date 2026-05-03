"""Tests for Error Recovery module - comprehensive coverage."""

from utils.error_recovery import (
    BackupState,
    ErrorRecoveryManager,
    RecoveryStrategy,
    attempt_recovery,
    create_checkpoint,
    restore_from_checkpoint,
)


class TestErrorRecoveryManager:
    """Test Error Recovery Manager."""

    def test_init(self):
        """Test initialization."""
        manager = ErrorRecoveryManager()
        assert manager is not None
        assert hasattr(manager, "strategies")

    def test_register_strategy(self):
        """Test registering a recovery strategy."""
        manager = ErrorRecoveryManager()

        def mock_recovery(error):
            return True

        strategy = RecoveryStrategy(
            name="test", condition=lambda e: True, action=mock_recovery
        )
        manager.register_strategy(strategy)
        assert "test" in manager.strategies

    def test_attempt_recovery_success(self):
        """Test successful recovery attempt."""
        manager = ErrorRecoveryManager()

        def mock_recovery(error):
            return {"recovered": True}

        strategy = RecoveryStrategy(
            name="success",
            condition=lambda e: isinstance(e, ValueError),
            action=mock_recovery,
        )
        manager.register_strategy(strategy)

        result = manager.attempt_recovery(ValueError("test"))
        assert result is not None


class TestBackupState:
    """Test Backup State functionality."""

    def test_create_backup(self):
        """Test creating a backup."""
        state = {"data": [1, 2, 3], "count": 3}
        backup = BackupState.create(state)
        assert backup is not None
        assert backup.state == state

    def test_restore_backup(self):
        """Test restoring from backup."""
        original = {"data": [1, 2, 3]}
        backup = BackupState.create(original)

        modified = {"data": [4, 5, 6]}
        restored = backup.restore(modified)
        assert restored == original


class TestAttemptRecovery:
    """Test standalone recovery function."""

    def test_attempt_recovery_with_strategy(self):
        """Test recovery with valid strategy."""
        strategies = [
            RecoveryStrategy(
                name="value_error",
                condition=lambda e: isinstance(e, ValueError),
                action=lambda e: {"fixed": True},
            )
        ]

        error = ValueError("test error")
        result = attempt_recovery(error, strategies)
        assert result is not None

    def test_attempt_recovery_no_match(self):
        """Test recovery with no matching strategy."""
        strategies = [
            RecoveryStrategy(
                name="type_error",
                condition=lambda e: isinstance(e, TypeError),
                action=lambda e: {"fixed": True},
            )
        ]

        error = ValueError("test error")
        result = attempt_recovery(error, strategies)
        assert result is None


class TestCheckpointFunctions:
    """Test checkpoint operations."""

    def test_create_checkpoint(self, tmp_path):
        """Test creating a checkpoint."""
        state = {"value": 42}
        checkpoint_path = tmp_path / "checkpoint.pkl"

        checkpoint = create_checkpoint(state, checkpoint_path)
        assert checkpoint is not None

    def test_restore_checkpoint(self, tmp_path):
        """Test restoring from checkpoint."""
        state = {"value": 42, "data": "test"}
        checkpoint_path = tmp_path / "checkpoint.pkl"

        create_checkpoint(state, checkpoint_path)
        restored = restore_from_checkpoint(checkpoint_path)

        assert restored == state
