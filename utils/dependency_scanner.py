"""
Dependency vulnerability scanner.
Regularly scans project dependencies for known security vulnerabilities.
============================================================================================
"""

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class DependencyScanner:
    """Scanner for dependency vulnerabilities."""

    def __init__(self, project_root: str = None):
        """
        Initialize dependency scanner.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = (
            Path(project_root).resolve() if project_root else Path.cwd().resolve()
        )
        self.requirements_file = self._validate_and_get_requirements_path()
        self.logger = logging.getLogger("dependency_scanner")

    def _validate_and_get_requirements_path(self) -> Path:
        """
        Validate and return the requirements.txt path.

        Returns:
            Path to requirements.txt within project root

        Raises:
            ValueError: If path validation fails
        """
        requirements_path = self.project_root / "requirements.txt"

        # Validate the path is within project root (prevent path traversal)
        try:
            requirements_path.resolve().relative_to(self.project_root)
        except ValueError:
            raise ValueError(
                f"Requirements file path {requirements_path} is outside project root"
            )

        return requirements_path

    def scan_with_pip_audit(self) -> Dict[str, Any]:
        """
        Scan dependencies using pip-audit.

        Returns:
            Scan results dictionary
        """
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--format",
                    "json",
                    "--requirement",
                    str(self.requirements_file),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                return {
                    "scanner": "pip-audit",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "vulnerabilities_found": 0,
                    "details": json.loads(result.stdout) if result.stdout else [],
                }
            else:
                # Parse vulnerabilities from stdout even on non-zero return code
                try:
                    if result.stdout:
                        details = json.loads(result.stdout)
                        vulnerabilities_count = len(details)
                    else:
                        # No output means scan failed to produce results
                        details = []
                        vulnerabilities_count = -1
                except json.JSONDecodeError:
                    details = []
                    vulnerabilities_count = -1

                return {
                    "scanner": "pip-audit",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": result.stderr if result.stderr else None,
                    "vulnerabilities_found": vulnerabilities_count,
                    "details": details,
                }

        except FileNotFoundError:
            self.logger.warning(
                "pip-audit not found, install with: pip install pip-audit"
            )
            return {
                "scanner": "pip-audit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "pip-audit not installed",
                "vulnerabilities_found": -1,
            }
        except subprocess.TimeoutExpired:
            return {
                "scanner": "pip-audit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Scan timed out",
                "vulnerabilities_found": -1,
            }
        except Exception as e:
            return {
                "scanner": "pip-audit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "vulnerabilities_found": -1,
            }

    def scan_with_safety(self) -> Dict[str, Any]:
        """
        Scan dependencies using safety.

        Returns:
            Scan results dictionary
        """
        try:
            result = subprocess.run(
                ["safety", "check", "--file", str(self.requirements_file), "--json"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                return {
                    "scanner": "safety",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "vulnerabilities_found": len(vulnerabilities),
                    "details": vulnerabilities,
                }
            else:
                return {
                    "scanner": "safety",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": result.stderr,
                    "vulnerabilities_found": -1,
                }

        except FileNotFoundError:
            self.logger.warning("safety not found, install with: pip install safety")
            return {
                "scanner": "safety",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "safety not installed",
                "vulnerabilities_found": -1,
            }
        except subprocess.TimeoutExpired:
            return {
                "scanner": "safety",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Scan timed out",
                "vulnerabilities_found": -1,
            }
        except Exception as e:
            return {
                "scanner": "safety",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "vulnerabilities_found": -1,
            }

    def scan_with_bandit(self) -> Dict[str, Any]:
        """
        Scan code for security issues using bandit.

        Returns:
            Scan results dictionary
        """
        try:
            result = subprocess.run(
                [
                    "bandit",
                    "-r",
                    str(self.project_root),
                    "-f",
                    "json",
                    "-x",
                    "tests/",
                    "-x",
                    ".git/",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0 or result.returncode == 1:
                # Exit code 1 means issues found but scan succeeded
                try:
                    results = json.loads(result.stdout)
                    return {
                        "scanner": "bandit",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "issues_found": len(results.get("results", [])),
                        "details": results,
                    }
                except json.JSONDecodeError:
                    return {
                        "scanner": "bandit",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "issues_found": 0,
                        "details": {},
                    }
            else:
                return {
                    "scanner": "bandit",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": result.stderr,
                    "issues_found": -1,
                }

        except FileNotFoundError:
            self.logger.warning("bandit not found, install with: pip install bandit")
            return {
                "scanner": "bandit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "bandit not installed",
                "issues_found": -1,
            }
        except subprocess.TimeoutExpired:
            return {
                "scanner": "bandit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Scan timed out",
                "issues_found": -1,
            }
        except Exception as e:
            return {
                "scanner": "bandit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "issues_found": -1,
            }

    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """
        Run comprehensive scan using all available scanners.

        Returns:
            Combined scan results
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "scans": {},
            "summary": {
                "total_vulnerabilities": 0,
                "total_issues": 0,
                "scanners_run": 0,
                "scanners_failed": 0,
            },
        }

        # Run pip-audit
        pip_audit_result = self.scan_with_pip_audit()
        results["scans"]["pip_audit"] = pip_audit_result
        results["summary"]["scanners_run"] += 1

        if pip_audit_result.get("vulnerabilities_found", 0) >= 0:
            results["summary"]["total_vulnerabilities"] += pip_audit_result.get(
                "vulnerabilities_found", 0
            )
        else:
            results["summary"]["scanners_failed"] += 1  # type: ignore[assignment]

        # Run safety
        safety_result = self.scan_with_safety()
        results["scans"]["safety"] = safety_result
        results["summary"]["scanners_run"] += 1

        if safety_result.get("vulnerabilities_found", 0) >= 0:
            results["summary"]["total_vulnerabilities"] += safety_result.get(
                "vulnerabilities_found", 0
            )
        else:
            results["summary"]["scanners_failed"] += 1  # type: ignore[assignment]

        # Run bandit
        bandit_result = self.scan_with_bandit()
        results["scans"]["bandit"] = bandit_result
        results["summary"]["scanners_run"] += 1

        if bandit_result.get("issues_found", 0) >= 0:
            results["summary"]["total_issues"] += bandit_result.get("issues_found", 0)
        else:
            results["summary"]["scanners_failed"] += 1  # type: ignore[assignment]

        return results

    def save_scan_report(
        self, results: Dict[str, Any], output_file: str | Path | None = None
    ) -> None:
        """
        Save scan results to file.

        Args:
            results: Scan results dictionary
            output_file: Path to output file
        """
        if output_file is None:
            output_file_path = self.project_root / "dependency_scan_report.json"
        else:
            output_file_path = Path(output_file)

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Scan report saved to {output_file_path}")

    def get_vulnerability_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of scan results.

        Args:
            results: Scan results dictionary

        Returns:
            Formatted summary string
        """
        summary_lines = [
            "=" * 60,
            "DEPENDENCY VULNERABILITY SCAN REPORT",
            "=" * 60,
            f"Scan Time: {results['timestamp']}",
            f"Project: {results['project_root']}",
            "",
            "SUMMARY:",
            f"  Total Vulnerabilities: {results['summary']['total_vulnerabilities']}",
            f"  Total Code Issues: {results['summary']['total_issues']}",
            f"  Scanners Run: {results['summary']['scanners_run']}",
            f"  Scanners Failed: {results['summary']['scanners_failed']}",
            "",
        ]

        # Add details from each scanner
        for scanner_name, scan_result in results["scans"].items():
            summary_lines.append(f"{scanner_name.upper()}:")

            if "error" in scan_result:
                summary_lines.append(f"  ERROR: {scan_result['error']}")
            else:
                if scanner_name == "bandit":
                    summary_lines.append(
                        f"  Issues Found: {scan_result.get('issues_found', 0)}"
                    )
                else:
                    summary_lines.append(
                        f"  Vulnerabilities Found: {scan_result.get('vulnerabilities_found', 0)}"
                    )

                if scan_result.get("details"):
                    details = scan_result["details"]
                    if isinstance(details, list) and len(details) > 0:
                        summary_lines.append(f"  Details: {len(details)} items found")

            summary_lines.append("")

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)


def run_dependency_scan(
    project_root: str = None, save_report: bool = True
) -> Dict[str, Any]:
    """
    Run dependency vulnerability scan.

    Args:
        project_root: Root directory of the project
        save_report: Whether to save scan report to file

    Returns:
        Scan results
    """
    scanner = DependencyScanner(project_root)
    results = scanner.run_comprehensive_scan()

    if save_report:
        scanner.save_scan_report(results)

    return results


def main():
    """Main entry point for dependency scanning."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan dependencies for vulnerabilities"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save scan report to file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed scan results",
    )

    args = parser.parse_args()

    # Run scan
    results = run_dependency_scan(
        project_root=args.project_root, save_report=not args.no_save
    )

    # Print summary
    scanner = DependencyScanner(args.project_root)
    print(scanner.get_vulnerability_summary(results))

    if args.verbose:
        print("\nDETAILED RESULTS:")
        print(json.dumps(results, indent=2))

    # Exit with error code if vulnerabilities found
    if results["summary"]["total_vulnerabilities"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
