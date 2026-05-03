"""
Comprehensive Tests for Monitoring System Module
=================================================

Target: 100% coverage for utils/monitoring_system.py
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.monitoring_system import (
    Alert,
    AlertChannel,
    AlertRule,
    AlertSeverity,
    HealthCheck,
    MonitoringSystem,
    NotificationManager,
    get_monitoring_system,
    get_status,
    start_monitoring,
    stop_monitoring,
)


class TestAlertSeverity:
    """Test AlertSeverity enum"""

    def test_enum_values(self):
        """Test severity enum values"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"


class TestAlertChannel:
    """Test AlertChannel enum"""

    def test_enum_values(self):
        """Test channel enum values"""
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.LOG.value == "log"
        assert AlertChannel.CONSOLE.value == "console"
        assert AlertChannel.FILE.value == "file"


class TestAlert:
    """Test Alert dataclass"""

    def test_alert_creation(self):
        """Test alert creation"""
        alert = Alert(
            id="test_001",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            metric_name="cpu_percent",
            metric_value=85.0,
            threshold=80.0,
            timestamp=time.time(),
        )
        assert alert.id == "test_001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert"
        assert alert.acknowledged is False
        assert alert.resolved is False

    def test_alert_to_dict(self):
        """Test alert to_dict conversion"""
        alert = Alert(
            id="test_002",
            severity=AlertSeverity.CRITICAL,
            message="Critical test",
            metric_name="memory_percent",
            metric_value=95.0,
            threshold=90.0,
            timestamp=time.time(),
            acknowledged=True,
        )
        d = alert.to_dict()
        assert d["id"] == "test_002"
        assert d["severity"] == "critical"
        assert d["acknowledged"] is True
        assert "metric_name" in d


class TestAlertRule:
    """Test AlertRule dataclass"""

    def test_rule_creation(self):
        """Test rule creation"""
        rule = AlertRule(
            name="High CPU",
            metric_name="cpu_percent",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
        )
        assert rule.name == "High CPU"
        assert rule.condition == ">"
        assert rule.enabled is True

    def test_check_condition_greater_than(self):
        """Test condition check with >"""
        rule = AlertRule(
            name="Test",
            metric_name="cpu",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        assert rule.check_condition(85.0) is True
        assert rule.check_condition(75.0) is False
        assert rule.check_condition(80.0) is False

    def test_check_condition_less_than(self):
        """Test condition check with <"""
        rule = AlertRule(
            name="Test",
            metric_name="memory",
            condition="<",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        assert rule.check_condition(5.0) is True
        assert rule.check_condition(15.0) is False

    def test_check_condition_equal(self):
        """Test condition check with =="""
        rule = AlertRule(
            name="Test",
            metric_name="value",
            condition="==",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        assert rule.check_condition(100.0) is True
        assert rule.check_condition(99.0) is False

    def test_check_condition_greater_equal(self):
        """Test condition check with >="""
        rule = AlertRule(
            name="Test",
            metric_name="cpu",
            condition=">=",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        assert rule.check_condition(85.0) is True
        assert rule.check_condition(80.0) is True
        assert rule.check_condition(75.0) is False

    def test_check_condition_less_equal(self):
        """Test condition check with <="""
        rule = AlertRule(
            name="Test",
            metric_name="memory",
            condition="<=",
            threshold=50.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        assert rule.check_condition(40.0) is True
        assert rule.check_condition(50.0) is True
        assert rule.check_condition(60.0) is False

    def test_can_trigger(self):
        """Test cooldown check"""
        rule = AlertRule(
            name="Test",
            metric_name="cpu",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
            cooldown_seconds=300.0,
            last_triggered=0.0,
        )
        assert rule.can_trigger() is True

        rule.last_triggered = time.time()
        assert rule.can_trigger() is False


class TestHealthCheck:
    """Test HealthCheck class"""

    def test_health_check_creation(self):
        """Test health check creation"""

        def check_func():
            return True

        check = HealthCheck("test_check", check_func, interval_seconds=30.0)
        assert check.name == "test_check"
        assert check.interval_seconds == 30.0
        assert check.timeout_seconds == 5.0
        assert check.consecutive_failures == 0

    def test_health_check_run_success(self):
        """Test health check run with success"""

        def check_func():
            return True

        check = HealthCheck("test", check_func)
        result = check.run()
        assert result is True
        assert check.last_result is True
        assert check.consecutive_failures == 0

    def test_health_check_run_failure(self):
        """Test health check run with failure"""

        def check_func():
            return False

        check = HealthCheck("test", check_func)
        result = check.run()
        assert result is False
        assert check.consecutive_failures == 1

    def test_health_check_run_exception(self):
        """Test health check run with exception"""

        def check_func():
            raise ValueError("Test error")

        check = HealthCheck("test", check_func)
        result = check.run()
        assert result is False
        assert check.consecutive_failures == 1


class TestNotificationManager:
    """Test NotificationManager class"""

    @pytest.fixture
    def manager(self):
        return NotificationManager()

    @pytest.fixture
    def sample_alert(self):
        return Alert(
            id="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test message",
            metric_name="cpu_percent",
            metric_value=85.0,
            threshold=80.0,
            timestamp=time.time(),
        )

    def test_manager_creation(self, manager):
        """Test notification manager creation"""
        assert isinstance(manager.config, dict)
        assert isinstance(manager._webhook_sessions, dict)

    def test_send_console(self, manager, sample_alert, capsys):
        """Test console notification"""
        manager._send_console(sample_alert)
        captured = capsys.readouterr()
        assert "ALERT" in captured.out
        assert "cpu_percent" in captured.out

    def test_send_file(self, manager, sample_alert, tmp_path):
        """Test file notification"""
        log_file = tmp_path / "alerts.jsonl"
        manager.config = {"file": {"path": str(log_file)}}
        manager._send_file(sample_alert)

        assert log_file.exists()
        content = log_file.read_text()
        assert "test_alert" in content

    def test_send_log(self, manager, sample_alert):
        """Test log notification"""
        # Should not raise even if logger unavailable
        manager._send_log(sample_alert)

    def test_send_email_disabled(self, manager, sample_alert):
        """Test email notification when disabled"""
        manager.config = {"email": {"enabled": False}}
        # Should return without sending
        manager._send_email(sample_alert)

    def test_send_email_no_recipients(self, manager, sample_alert):
        """Test email notification with no recipients"""
        manager.config = {"email": {"enabled": True, "to": []}}
        # Should return without sending
        manager._send_email(sample_alert)

    def test_send_webhook(self, manager, sample_alert):
        """Test webhook notification"""
        manager.config = {"webhooks": ["http://localhost:9999/test"]}
        # Should handle connection error gracefully
        manager._send_webhook(sample_alert)

    def test_send_notification(self, manager, sample_alert):
        """Test send notification to multiple channels"""
        channels = [AlertChannel.CONSOLE, AlertChannel.LOG]
        # Should not raise
        manager.send_notification(sample_alert, channels)


class TestMonitoringSystem:
    """Test MonitoringSystem class"""

    @pytest.fixture
    def monitoring(self):
        return MonitoringSystem()

    def test_monitoring_creation(self, monitoring):
        """Test monitoring system creation"""
        assert isinstance(monitoring.alert_rules, list)
        assert isinstance(monitoring.health_checks, list)
        assert isinstance(monitoring.active_alerts, dict)
        assert isinstance(monitoring.alert_history, list)

    def test_add_alert_rule(self, monitoring):
        """Test adding alert rule"""
        rule = AlertRule(
            name="Test Rule",
            metric_name="cpu_percent",
            condition=">",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.LOG],
        )
        monitoring.add_alert_rule(rule)
        assert len(monitoring.alert_rules) > 0
        assert monitoring.alert_rules[-1].name == "Test Rule"

    def test_add_health_check(self, monitoring):
        """Test adding health check"""

        def check_func():
            return True

        check = HealthCheck("test_health", check_func)
        monitoring.add_health_check(check)
        assert len(monitoring.health_checks) > 0

    def test_check_alert_rules(self, monitoring):
        """Test checking alert rules"""
        rule = AlertRule(
            name="CPU High",
            metric_name="cpu_percent",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
            cooldown_seconds=0,  # No cooldown for testing
        )
        monitoring.add_alert_rule(rule)

        metrics = {"cpu_percent": 85.0}
        monitoring.check_alert_rules(metrics)
        # Should have triggered alert
        assert len(monitoring.alert_history) > 0

    def test_check_alert_rules_no_trigger(self, monitoring):
        """Test alert rules not triggered"""
        rule = AlertRule(
            name="CPU High",
            metric_name="cpu_percent",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )
        monitoring.add_alert_rule(rule)

        metrics = {"cpu_percent": 70.0}
        monitoring.check_alert_rules(metrics)
        # Should not have triggered alert
        assert len(monitoring.alert_history) == 0

    def test_run_health_checks(self, monitoring):
        """Test running health checks"""

        def check_func():
            return True

        check = HealthCheck("test", check_func)
        monitoring.add_health_check(check)

        results = monitoring.run_health_checks()
        assert "test" in results
        assert results["test"] is True

    def test_acknowledge_alert(self, monitoring):
        """Test acknowledging alert"""
        alert = Alert(
            id="test_ack",
            severity=AlertSeverity.WARNING,
            message="Test",
            metric_name="cpu",
            metric_value=85.0,
            threshold=80.0,
            timestamp=time.time(),
        )
        monitoring.active_alerts["test_ack"] = alert

        result = monitoring.acknowledge_alert("test_ack")
        assert result is True
        assert monitoring.active_alerts["test_ack"].acknowledged is True

    def test_acknowledge_nonexistent_alert(self, monitoring):
        """Test acknowledging non-existent alert"""
        result = monitoring.acknowledge_alert("nonexistent")
        assert result is False

    def test_resolve_alert(self, monitoring):
        """Test resolving alert"""
        alert = Alert(
            id="test_res",
            severity=AlertSeverity.WARNING,
            message="Test",
            metric_name="cpu",
            metric_value=85.0,
            threshold=80.0,
            timestamp=time.time(),
        )
        monitoring.active_alerts["test_res"] = alert

        result = monitoring.resolve_alert("test_res")
        assert result is True
        assert "test_res" not in monitoring.active_alerts

    def test_resolve_nonexistent_alert(self, monitoring):
        """Test resolving non-existent alert"""
        result = monitoring.resolve_alert("nonexistent")
        assert result is False

    def test_get_system_status(self, monitoring):
        """Test getting system status"""
        status = monitoring.get_system_status()
        assert "timestamp" in status
        assert "monitoring_active" in status
        assert "active_alerts_count" in status

    def test_get_metrics_history(self, monitoring):
        """Test getting metrics history"""
        # Add some mock history
        monitoring.metrics_history["test_metric"] = [
            {"timestamp": time.time(), "value": 50.0}
        ]

        history = monitoring.get_metrics_history("test_metric", hours=24)
        assert len(history) == 1
        assert history[0]["value"] == 50.0

    def test_export_status_report(self, monitoring, tmp_path):
        """Test exporting status report"""
        report_file = tmp_path / "status_report.json"
        monitoring.export_status_report(str(report_file))

        assert report_file.exists()
        import json

        with open(report_file) as f:
            data = json.load(f)
        assert "generated_at" in data
        assert "status" in data

    def test_load_config(self, monitoring, tmp_path):
        """Test loading config"""
        config_file = tmp_path / "config.json"
        config_data = {"notifications": {"email": {"enabled": True}}}
        with open(config_file, "w") as f:
            import json

            json.dump(config_data, f)

        monitoring._load_config(str(config_file))
        assert monitoring.notification_manager.config == config_data["notifications"]

    def test_start_stop_monitoring(self, monitoring):
        """Test starting and stopping monitoring"""
        # Start monitoring
        monitoring.start_monitoring(interval_seconds=1.0)
        assert monitoring._monitoring_active is True
        assert monitoring._monitor_thread is not None
        assert monitoring._monitor_thread.is_alive()

        # Stop monitoring
        monitoring.stop_monitoring()
        assert monitoring._monitoring_active is False


class TestGlobalFunctions:
    """Test global convenience functions"""

    def test_get_monitoring_system(self):
        """Test get_monitoring_system singleton"""
        m1 = get_monitoring_system()
        m2 = get_monitoring_system()
        assert m1 is m2

    def test_get_status(self):
        """Test get_status function"""
        status = get_status()
        assert isinstance(status, dict)
        assert "timestamp" in status

    def test_start_stop_monitoring_global(self):
        """Test global start/stop functions"""
        start_monitoring(interval_seconds=1.0)
        status = get_status()
        assert status["monitoring_active"] is True

        stop_monitoring()
        status = get_status()
        assert status["monitoring_active"] is False
