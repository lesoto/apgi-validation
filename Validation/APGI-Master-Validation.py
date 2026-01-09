"""
APGI Master Validation Pipeline
================================

Executes all 8 protocols and applies hierarchical falsification decision tree
per validation roadmap page 26.

Falsification Logic:
- Primary tests (Protocols 1, 3): Failure → Framework rejected
- Secondary tests (2+ failures): Major revision required
- Tertiary tests (3+ failures): Scope restriction
"""

import json
from pathlib import Path
from typing import Dict, List


class APGIMasterValidator:
    def __init__(self):
        self.protocol_results = {}
        self.falsification_status = {"primary": [], "secondary": [], "tertiary": []}

    def run_all_protocols(self):
        """Execute all 8 protocols in sequence"""
        # Protocol tier classification
        protocol_tiers = {
            1: "primary",  # Primary tests: Failure → Framework rejected
            2: "secondary",  # Secondary tests: 2+ failures → Major revision
            3: "primary",  # Primary tests: Failure → Framework rejected
            4: "secondary",  # Secondary tests: 2+ failures → Major revision
            5: "tertiary",  # Tertiary tests: 3+ failures → Scope restriction
            6: "tertiary",  # Tertiary tests: 3+ failures → Scope restriction
            7: "tertiary",  # Tertiary tests: 3+ failures → Scope restriction
            8: "secondary",  # Secondary tests: 2+ failures → Major revision
        }

        for protocol_num in range(1, 9):
            try:
                # Import protocol module
                module_name = f"APGI-Protocol-{protocol_num}"
                protocol_module = __import__(module_name)

                # Run protocol validation
                if hasattr(protocol_module, "run_validation"):
                    result = protocol_module.run_validation()
                elif hasattr(protocol_module, "main"):
                    result = protocol_module.main()
                else:
                    # Try to find main validation function
                    validation_functions = [
                        attr
                        for attr in dir(protocol_module)
                        if callable(getattr(protocol_module, attr))
                        and "validation" in attr.lower()
                    ]
                    if validation_functions:
                        result = getattr(protocol_module, validation_functions[0])()
                    else:
                        result = {"status": "NO_VALIDATION_FUNCTION", "passed": False}

                # Store result
                self.protocol_results[f"protocol_{protocol_num}"] = result

                # Determine if protocol passed
                passed = (
                    result.get("passed", True) if isinstance(result, dict) else True
                )
                tier = protocol_tiers[protocol_num]

                self.falsification_status[tier].append(
                    {"protocol": protocol_num, "passed": passed, "result": result}
                )

                print(f"Protocol {protocol_num}: {'PASSED' if passed else 'FAILED'}")

            except Exception as e:
                # Protocol execution failed
                error_result = {
                    "status": "EXECUTION_ERROR",
                    "error": str(e),
                    "passed": False,
                }

                self.protocol_results[f"protocol_{protocol_num}"] = error_result
                tier = protocol_tiers[protocol_num]
                self.falsification_status[tier].append(
                    {"protocol": protocol_num, "passed": False, "result": error_result}
                )

                print(f"Protocol {protocol_num}: ERROR - {e}")

    def apply_decision_tree(self) -> str:
        """
        Apply hierarchical falsification logic

        Returns:
            'VALIDATED', 'MAJOR_REVISION', 'SCOPE_RESTRICTION', or 'REJECTED'
        """
        # Count failures at each tier
        primary_failures = len(
            [r for r in self.falsification_status["primary"] if not r["passed"]]
        )
        secondary_failures = len(
            [r for r in self.falsification_status["secondary"] if not r["passed"]]
        )
        tertiary_failures = len(
            [r for r in self.falsification_status["tertiary"] if not r["passed"]]
        )

        # Decision tree
        if primary_failures >= 1:
            return "REJECTED"
        elif secondary_failures >= 2:
            return "MAJOR_REVISION"
        elif tertiary_failures >= 3:
            return "SCOPE_RESTRICTION"
        else:
            return "VALIDATED"

    def generate_master_report(self) -> Dict:
        """Comprehensive validation report"""
        return {
            "protocol_results": self.protocol_results,
            "falsification_status": self.falsification_status,
            "overall_decision": self.apply_decision_tree(),
        }


if __name__ == "__main__":
    # Run master validation
    validator = APGIMasterValidator()

    print("Starting APGI Master Validation Pipeline...")
    print("=" * 50)

    # Execute all protocols
    validator.run_all_protocols()

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    # Generate and display report
    report = validator.generate_master_report()

    print(f"Overall Decision: {report['overall_decision']}")

    # Print tier summaries
    for tier, results in report["falsification_status"].items():
        failures = len([r for r in results if not r["passed"]])
        total = len(results)
        print(f"{tier.capitalize()} tier: {failures}/{total} failed")

    # Save detailed report
    with open("APGI-Master-Validation-Report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: APGI-Master-Validation-Report.json")
