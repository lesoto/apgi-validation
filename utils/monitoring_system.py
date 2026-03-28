"""
APGI Monitoring and Alerting System
====================================

Real-time system monitoring with configurable alerts,
notification channels, and health checks.
"""

import json
import smtplib
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Available notification channels."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class Alert:
    """Alert data structure."""

    id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts."""

    name: str
    metric_name: str
    condition: str  # '>', '<', '==', '>=', '<='
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_seconds: float = 300.0  # 5 minutes
    enabled: bool = True
    last_triggered: float = 0.0

    def check_condition(self, value: float) -> bool:
        """Check if value triggers the alert condition."""
        conditions = {
            ">": value > self.threshold,
            "<": value < self.threshold,
            "==": value == self.threshold,
            ">=": value >= self.threshold,
            "<=": value <= self.threshold,
        }
        return conditions.get(self.condition, False)

    def can_trigger(self) -> bool:
        """Check if alert can be triggered (respects cooldown)."""
        return time.time() - self.last_triggered >= self.cooldown_seconds


class HealthCheck:
    """System health check."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval_seconds: float = 60.0,
        timeout_seconds: float = 5.0,
    ):
        self.name = name
        self.check_func = check_func
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.last_result: Optional[bool] = None
        self.last_check_time: float = 0.0
        self.consecutive_failures = 0

    def run(self) -> bool:
        """Execute health check."""
        try:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.check_func)
                result = future.result(timeout=self.timeout_seconds)

            self.last_result = result
            self.last_check_time = time.time()

            if result:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

            return result

        except Exception as e:
            self.last_result = False
            self.consecutive_failures += 1
            if apgi_logger:
                apgi_logger.logger.error(f"Health check {self.name} failed: {e}")
            return False


class NotificationManager:
    """Manages alert notifications across multiple channels."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._webhook_sessions: Dict[str, Any] = {}

    def send_notification(self, alert: Alert, channels: List[AlertChannel]):
        """Send notification through specified channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)
                elif channel == AlertChannel.LOG:
                    self._send_log(alert)
                elif channel == AlertChannel.CONSOLE:
                    self._send_console(alert)
                elif channel == AlertChannel.FILE:
                    self._send_file(alert)
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(
                        f"Failed to send {channel.value} notification: {e}"
                    )

    def _send_email(self, alert: Alert):
        """Send email notification."""
        email_config = self.config.get("email", {})
        if not email_config.get("enabled", False):
            return

        smtp_host = email_config.get("smtp_host", "localhost")
        smtp_port = email_config.get("smtp_port", 587)
        from_addr = email_config.get("from", "alerts@apgi.local")
        to_addrs = email_config.get("to", [])

        if not to_addrs:
            return

        msg = MIMEText(
            f"APGI Alert: {alert.severity.value.upper()}\n\n"
            f"Message: {alert.message}\n"
            f"Metric: {alert.metric_name} = {alert.metric_value}\n"
            f"Threshold: {alert.threshold}\n"
            f"Time: {datetime.fromtimestamp(alert.timestamp)}"
        )
        msg["Subject"] = f"[APGI {alert.severity.value.upper()}] {alert.metric_name}"
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if email_config.get("use_tls", True):
                    server.starttls()
                if email_config.get("username"):
                    server.login(
                        email_config["username"], email_config.get("password", "")
                    )
                server.send_message(msg)
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Email send failed: {e}")

    def _send_webhook(self, alert: Alert):
        """Send webhook notification."""
        webhooks = self.config.get("webhooks", [])
        for webhook_url in webhooks:
            try:
                import requests

                payload = alert.to_dict()
                requests.post(
                    webhook_url,
                    json=payload,
                    timeout=10,
                    headers={"Content-Type": "application/json"},
                )
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Webhook failed for {webhook_url}: {e}")

    def _send_log(self, alert: Alert):
        """Send notification to log."""
        if not apgi_logger:
            return

        log_msg = (
            f"ALERT [{alert.severity.value.upper()}] {alert.metric_name}: "
            f"{alert.metric_value} (threshold: {alert.threshold}) - {alert.message}"
        )

        if alert.severity == AlertSeverity.EMERGENCY:
            apgi_logger.logger.critical(log_msg)
        elif alert.severity == AlertSeverity.CRITICAL:
            apgi_logger.logger.critical(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            apgi_logger.logger.warning(log_msg)
        else:
            apgi_logger.logger.info(log_msg)

    def _send_console(self, alert: Alert):
        """Send notification to console."""
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(
            f"\n[{timestamp}] ALERT [{alert.severity.value.upper()}] {alert.metric_name}\n"
            f"  Value: {alert.metric_value:.2f} (threshold: {alert.threshold})\n"
            f"  Message: {alert.message}\n"
        )

    def _send_file(self, alert: Alert):
        """Write alert to file."""
        file_config = self.config.get("file", {})
        log_file = file_config.get("path", "logs/alerts.jsonl")

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a") as f:
            f.write(json.dumps(alert.to_dict(), default=str) + "\n")


class MonitoringSystem:
    """Main monitoring and alerting system."""

    def __init__(self, config_path: Optional[str] = None):
        self.alert_rules: List[AlertRule] = []
        self.health_checks: List[HealthCheck] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.notification_manager = NotificationManager()

        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics_lock = threading.Lock()

        # Load config if provided
        if config_path:
            self._load_config(config_path)

        # Setup default alert rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default monitoring rules."""
        if not PSUTIL_AVAILABLE:
            return

        # CPU usage alert
        self.add_alert_rule(
            AlertRule(
                name="High CPU Usage",
                metric_name="cpu_percent",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
                cooldown_seconds=300,
            )
        )

        # Memory usage alert
        self.add_alert_rule(
            AlertRule(
                name="High Memory Usage",
                metric_name="memory_percent",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
                cooldown_seconds=300,
            )
        )

        # Disk usage alert
        self.add_alert_rule(
            AlertRule(
                name="Low Disk Space",
                metric_name="disk_percent",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.FILE],
                cooldown_seconds=600,
            )
        )

    def _load_config(self, config_path: str):
        """Load monitoring configuration from file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.notification_manager.config = config.get("notifications", {})
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.warning(f"Failed to load monitoring config: {e}")

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)

    def add_health_check(self, check: HealthCheck):
        """Add a health check."""
        self.health_checks.append(check)

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        if not PSUTIL_AVAILABLE:
            return {}

        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_percent": psutil.disk_usage("/").percent,
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
        }

        # Add to history
        with self._metrics_lock:
            for key, value in metrics.items():
                if key != "timestamp":
                    if key not in self.metrics_history:
                        self.metrics_history[key] = []
                    self.metrics_history[key].append(
                        {"timestamp": metrics["timestamp"], "value": value}
                    )
                    # Keep only last 1000 entries
                    self.metrics_history[key] = self.metrics_history[key][-1000:]

        return metrics

    def check_alert_rules(self, metrics: Dict[str, float]):
        """Check all alert rules against current metrics."""
        for rule in self.alert_rules:
            if not rule.enabled or not rule.can_trigger():
                continue

            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue

            if rule.check_condition(metric_value):
                # Trigger alert
                alert = Alert(
                    id=f"{rule.name}_{int(time.time())}",
                    severity=rule.severity,
                    message=f"{rule.name}: {rule.metric_name} is {metric_value:.1f}",
                    metric_name=rule.metric_name,
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    timestamp=time.time(),
                )

                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                rule.last_triggered = time.time()

                # Send notifications
                self.notification_manager.send_notification(alert, rule.channels)

    def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        for check in self.health_checks:
            results[check.name] = check.run()
        return results

    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start continuous monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                try:
                    # Collect metrics
                    metrics = self.collect_system_metrics()

                    # Check alert rules
                    self.check_alert_rules(metrics)

                    # Run health checks
                    self.run_health_checks()

                    # Sleep until next check
                    time.sleep(interval_seconds)

                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.error(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

        if apgi_logger:
            apgi_logger.logger.info(
                f"Started monitoring system (interval: {interval_seconds}s)"
            )

    def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring_active = False
        if apgi_logger:
            apgi_logger.logger.info("Stopped monitoring system")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            del self.active_alerts[alert_id]
            return True
        return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        return {
            "timestamp": time.time(),
            "monitoring_active": self._monitoring_active,
            "active_alerts_count": len(self.active_alerts),
            "active_alerts": [a.to_dict() for a in self.active_alerts.values()],
            "health_checks": len(self.health_checks),
            "alert_rules": len(self.alert_rules),
        }

    def get_metrics_history(self, metric_name: str, hours: float = 24.0) -> List[Dict]:
        """Get historical metrics for a specific metric."""
        cutoff_time = time.time() - (hours * 3600)

        with self._metrics_lock:
            history = self.metrics_history.get(metric_name, [])
            return [h for h in history if h["timestamp"] > cutoff_time]

    def export_status_report(self, filepath: str):
        """Export system status report to file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "status": self.get_system_status(),
            "alert_history": [a.to_dict() for a in self.alert_history[-100:]],
            "metrics_summary": {
                name: {
                    "latest": values[-1]["value"] if values else None,
                    "count": len(values),
                }
                for name, values in self.metrics_history.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)


# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Get or create global monitoring system."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system


def start_monitoring(interval_seconds: float = 30.0):
    """Start system monitoring."""
    get_monitoring_system().start_monitoring(interval_seconds)


def stop_monitoring():
    """Stop system monitoring."""
    get_monitoring_system().stop_monitoring()


def get_status() -> Dict[str, Any]:
    """Get current monitoring status."""
    return get_monitoring_system().get_system_status()


if __name__ == "__main__":
    # Demo monitoring
    print("APGI Monitoring and Alerting System Demo")
    print("=" * 40)

    monitoring = get_monitoring_system()

    # Add custom health check
    def check_database():
        return True  # Simulated check

    monitoring.add_health_check(
        HealthCheck("database", check_database, interval_seconds=10)
    )

    # Start monitoring
    print("\n1. Starting monitoring...")
    monitoring.start_monitoring(interval_seconds=5.0)

    # Get status
    print("\n2. System status:")
    print(f"   {get_status()}")

    print("\n3. Monitoring active for 10 seconds...")
    time.sleep(10)

    # Stop monitoring
    print("\n4. Stopping monitoring...")
    stop_monitoring()

    print("\nMonitoring system ready!")
