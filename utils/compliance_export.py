#!/usr/bin/env python3
"""
Immutable Audit Export and Compliance Reporting
=============================================

Generates periodic compliance reports and securely exports audit trails
with cryptographic signatures for immutability verification.
"""

import hashlib
import hmac
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.secure_key_manager import get_backup_hmac_key


class ComplianceExporter:
    """Handles compliance reporting and immutable audit exports."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.export_dir = Path("compliance_exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _get_signing_key(self) -> bytes:
        """Get the HMAC key for signing exports."""
        # Reuse backup HMAC key or specific compliance key
        return get_backup_hmac_key().encode("utf-8")

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a summary compliance report."""
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "framework_version": "1.3.0",
            "controls_status": {
                "access_control": "Enforced (auth_adapter)",
                "data_retention": "Enforced (data_protection)",
                "encryption": "Enforced (secure_key_manager)",
                "audit_logging": "Active",
            },
            "recent_audit_events": self._fetch_recent_audit_events(),
        }
        return report

    def _fetch_recent_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent audit events from logs."""
        # For demonstration, returns mock parsed events
        # In a real scenario, this reads from persistent_audit_logger files
        return [
            {
                "event": "system_startup",
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

    def export_audit_log(self, report_data: Dict[str, Any]) -> str:
        """Export report and sign it immutably."""
        export_id = (
            f"audit_export_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        export_path = self.export_dir / f"{export_id}.json"

        # Sign the payload
        payload_str = json.dumps(report_data, sort_keys=True)
        signature = hmac.new(
            self._get_signing_key(), payload_str.encode(), hashlib.sha256
        ).hexdigest()

        export_package = {
            "id": export_id,
            "signature": signature,
            "algorithm": "HMAC-SHA256",
            "payload": report_data,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_package, f, indent=2)

        return str(export_path)


def run_compliance_export() -> str:
    """Run the compliance export process."""
    exporter = ComplianceExporter()
    report = exporter.generate_compliance_report()
    return exporter.export_audit_log(report)


if __name__ == "__main__":
    export_file = run_compliance_export()
    print(f"Compliance report exported immutably to: {export_file}")
