"""
APGI Mutation Testing Module
============================

Implements mutation testing to verify test effectiveness and detect weak assertions.
Uses mutmut-inspired mutation operators adapted for scientific computing.

Mutation Operators:
- Arithmetic: + → -, * → /, etc.
- Comparison: > → >=, == → !=, etc.
- Boundary: Constant modification
- Logical: and → or, not removal
- Scientific: Statistical test modifications
"""

import ast
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional


class MutationType(Enum):
    """Types of mutations that can be applied."""

    # Arithmetic mutations
    ADD_TO_SUB = auto()
    SUB_TO_ADD = auto()
    MUL_TO_DIV = auto()
    DIV_TO_MUL = auto()
    POW_TO_MUL = auto()

    # Comparison mutations
    GT_TO_GE = auto()
    GE_TO_GT = auto()
    LT_TO_LE = auto()
    LE_TO_LT = auto()
    EQ_TO_NE = auto()
    NE_TO_EQ = auto()

    # Logical mutations
    AND_TO_OR = auto()
    OR_TO_AND = auto()
    NOT_REMOVAL = auto()

    # Boundary mutations
    CONSTANT_INCREASE = auto()
    CONSTANT_DECREASE = auto()
    ZERO_TO_ONE = auto()
    ONE_TO_ZERO = auto()

    # Scientific mutations
    MEAN_TO_MEDIAN = auto()
    STD_TO_VAR = auto()
    CORR_TO_COV = auto()

    # Statistical test mutations
    TTEST_TO_WILCOXON = auto()
    PVALUE_THRESHOLD = auto()


@dataclass
class Mutant:
    """Represents a single mutant."""

    mutant_id: str
    mutation_type: MutationType
    original_code: str
    mutated_code: str
    line_number: int
    function_name: str
    module_path: str
    status: str = "pending"  # pending, killed, survived, timeout, error


@dataclass
class MutationResult:
    """Result of testing a single mutant."""

    mutant: Mutant
    tests_killed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = 0.0


class MutationOperator:
    """Base class for mutation operators."""

    def __init__(self, mutation_type: MutationType):
        self.mutation_type = mutation_type

    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        """Apply mutation to AST node. Return mutated node or None if not applicable."""
        raise NotImplementedError


class ArithmeticMutator(MutationOperator):
    """Mutates arithmetic operators."""

    MUTATIONS = {
        MutationType.ADD_TO_SUB: (ast.Add, ast.Sub),
        MutationType.SUB_TO_ADD: (ast.Sub, ast.Add),
        MutationType.MUL_TO_DIV: (ast.Mult, ast.Div),
        MutationType.DIV_TO_MUL: (ast.Div, ast.Mult),
    }

    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if not isinstance(node, ast.BinOp):
            return None

        for mut_type, (orig, new) in self.MUTATIONS.items():
            if isinstance(node.op, orig) and mut_type == self.mutation_type:
                new_node = deepcopy(node)
                new_node.op = new()
                return new_node

        return None


class ComparisonMutator(MutationOperator):
    """Mutates comparison operators."""

    MUTATIONS = {
        MutationType.GT_TO_GE: (ast.Gt, ast.GtE),
        MutationType.GE_TO_GT: (ast.GtE, ast.Gt),
        MutationType.LT_TO_LE: (ast.Lt, ast.LtE),
        MutationType.LE_TO_LT: (ast.LtE, ast.Lt),
        MutationType.EQ_TO_NE: (ast.Eq, ast.NotEq),
        MutationType.NE_TO_EQ: (ast.NotEq, ast.Eq),
    }

    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if not isinstance(node, ast.Compare):
            return None

        for mut_type, (orig, new) in self.MUTATIONS.items():
            if (
                any(isinstance(op, orig) for op in node.ops)
                and mut_type == self.mutation_type
            ):
                new_node = deepcopy(node)
                new_node.ops = [
                    new() if isinstance(op, orig) else op for op in node.ops
                ]
                return new_node

        return None


class ConstantMutator(MutationOperator):
    """Mutates numeric constants."""

    def __init__(self, mutation_type: MutationType, delta: float = 0.1):
        super().__init__(mutation_type)
        self.delta = delta

    def apply(self, node: ast.AST) -> Optional[ast.AST]:
        if not isinstance(node, ast.Constant) or not isinstance(
            node.value, (int, float)
        ):
            return None

        value = node.value

        if self.mutation_type == MutationType.CONSTANT_INCREASE:
            new_value = value * (1 + self.delta)
        elif self.mutation_type == MutationType.CONSTANT_DECREASE:
            new_value = value * (1 - self.delta)
        elif self.mutation_type == MutationType.ZERO_TO_ONE:
            new_value = 1 if value == 0 else None
        elif self.mutation_type == MutationType.ONE_TO_ZERO:
            new_value = 0 if value == 1 else None
        else:
            return None

        if new_value is None:
            return None

        new_node = deepcopy(node)
        new_node.value = new_value
        return new_node


class MutationGenerator(ast.NodeTransformer):
    """Generates all possible mutations for a given AST."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.mutants: List[Mutant] = []
        self.mutant_counter = 0

    def generate_mutants(self, tree: ast.AST) -> List[Mutant]:
        """Generate all mutants for the given AST."""
        self.visit(tree)
        return self.mutants

    def _create_mutant(
        self,
        node: ast.AST,
        mutation_type: MutationType,
        mutator: MutationOperator,
        function_name: str = "<module>",
    ) -> None:
        """Create a mutant if the mutation applies."""
        mutated = mutator.apply(node)
        if mutated is not None:
            self.mutant_counter += 1
            mutant_id = f"{self.source_file}:{function_name}:{self.mutant_counter}"

            # Get original code
            original_code = ast.unparse(node)
            mutated_code = ast.unparse(mutated)

            self.mutants.append(
                Mutant(
                    mutant_id=mutant_id,
                    mutation_type=mutation_type,
                    original_code=original_code,
                    mutated_code=mutated_code,
                    line_number=getattr(node, "lineno", 0),
                    function_name=function_name,
                    module_path=self.source_file,
                )
            )

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Visit binary operations for arithmetic mutations."""
        # Generate arithmetic mutants
        for mut_type in [
            MutationType.ADD_TO_SUB,
            MutationType.SUB_TO_ADD,
            MutationType.MUL_TO_DIV,
            MutationType.DIV_TO_MUL,
        ]:
            self._create_mutant(node, mut_type, ArithmeticMutator(mut_type))

        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """Visit comparisons for comparison mutations."""
        for mut_type in [
            MutationType.GT_TO_GE,
            MutationType.GE_TO_GT,
            MutationType.LT_TO_LE,
            MutationType.LE_TO_LT,
            MutationType.EQ_TO_NE,
            MutationType.NE_TO_EQ,
        ]:
            self._create_mutant(node, mut_type, ComparisonMutator(mut_type))

        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Visit constants for boundary mutations."""
        if isinstance(node.value, (int, float)):
            for mut_type in [
                MutationType.CONSTANT_INCREASE,
                MutationType.CONSTANT_DECREASE,
                MutationType.ZERO_TO_ONE,
                MutationType.ONE_TO_ZERO,
            ]:
                self._create_mutant(node, mut_type, ConstantMutator(mut_type))

        return self.generic_visit(node)


class MutationTester:
    """
    Main mutation testing orchestrator.

    Runs mutation testing on specified modules and generates reports.
    """

    def __init__(
        self,
        target_modules: List[str],
        test_modules: List[str],
        timeout_seconds: float = 5.0,
        parallel_workers: int = 4,
    ):
        self.target_modules = target_modules
        self.test_modules = test_modules
        self.timeout_seconds = timeout_seconds
        self.parallel_workers = parallel_workers
        self.results: List[MutationResult] = []

    def run(self) -> Dict[str, Any]:
        """
        Run full mutation testing suite.

        Returns:
            Comprehensive mutation testing report
        """
        print("=" * 80)
        print("APGI MUTATION TESTING")
        print("=" * 80)

        # Generate mutants for all target modules
        all_mutants: List[Mutant] = []
        for module_path in self.target_modules:
            mutants = self._generate_mutants_for_module(module_path)
            all_mutants.extend(mutants)
            print(f"Generated {len(mutants)} mutants for {module_path}")

        print(f"\nTotal mutants to test: {len(all_mutants)}")

        # Test each mutant
        killed = 0
        survived = 0
        timeout_count = 0
        error_count = 0

        for i, mutant in enumerate(all_mutants, 1):
            print(f"Testing mutant {i}/{len(all_mutants)}: {mutant.mutant_id}", end=" ")

            result = self._test_mutant(mutant)
            self.results.append(result)

            if result.mutant.status == "killed":
                killed += 1
                print("✓ KILLED")
            elif result.mutant.status == "survived":
                survived += 1
                print("✗ SURVIVED")
            elif result.mutant.status == "timeout":
                timeout_count += 1
                print("⏱ TIMEOUT")
            else:
                error_count += 1
                print(f"⚠ ERROR: {result.error_message}")

        # Calculate mutation score
        summary = {}
        total_valid = killed + survived
        mutation_score = killed / total_valid * 100 if total_valid > 0 else 0
        summary["mutation_score"] = mutation_score

        report = {
            "total_mutants": len(all_mutants),
            "killed": killed,
            "survived": survived,
            "timeout": timeout_count,
            "error": error_count,
            "mutation_score": mutation_score,
            "survived_mutants": [
                {
                    "id": r.mutant.mutant_id,
                    "type": r.mutant.mutation_type.name,
                    "line": r.mutant.line_number,
                    "original": r.mutant.original_code,
                    "mutated": r.mutant.mutated_code,
                }
                for r in self.results
                if r.mutant.status == "survived"
            ],
            "weak_assertions": self._identify_weak_assertions(),
        }

        # Print summary
        print(f"\n{'=' * 80}")
        print("MUTATION TESTING SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total Mutants: {len(all_mutants)}")
        print(f"Killed: {killed} ✓")
        print(f"Survived: {survived} {'✓' if survived == 0 else '✗'}")
        print(f"Timeout: {timeout_count}")
        print(f"Error: {error_count}")
        print(f"Mutation Score: {mutation_score:.1f}%")

        if mutation_score < 50:
            print("\n⚠️  CRITICAL: Mutation score below 50%. Tests need strengthening.")
        elif mutation_score < 80:
            print(
                "\n⚠️  WARNING: Mutation score below 80%. Consider adding more assertions."
            )
        else:
            print("\n✅ Excellent mutation coverage!")

        return report

    def _generate_mutants_for_module(self, module_path: str) -> List[Mutant]:
        """Generate all mutants for a module."""
        path = Path(module_path)
        if not path.exists():
            return []

        with open(path, "r") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        generator = MutationGenerator(str(module_path))
        return generator.generate_mutants(tree)

    def _test_mutant(self, mutant: Mutant) -> MutationResult:
        """Test a single mutant by applying it and running tests."""
        import os
        import subprocess
        import tempfile

        start_time = time.time()
        result = MutationResult(mutant=mutant)

        try:
            # Create a temporary module with the mutation applied
            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy original module
                original_path = Path(mutant.module_path)
                temp_path = Path(tmpdir) / original_path.name

                with open(original_path, "r") as f:
                    source = f.read()

                # Apply mutation
                mutated_source = self._apply_mutation(source, mutant)

                with open(temp_path, "w") as f:
                    f.write(mutated_source)

                # Run tests against mutated code
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{tmpdir}:{env.get('PYTHONPATH', '')}"

                test_cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-x",  # Stop on first failure
                    "-v",
                    "--tb=no",
                    "--timeout=10",
                ] + self.test_modules

                test_result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=env,
                )

                result.execution_time = time.time() - start_time

                # If tests fail, mutant is killed
                if test_result.returncode != 0:
                    mutant.status = "killed"
                    # Extract failing test names
                    for line in test_result.stdout.split("\n"):
                        if "FAILED" in line:
                            result.tests_killed.append(line.strip())
                else:
                    mutant.status = "survived"

        except subprocess.TimeoutExpired:
            mutant.status = "timeout"
            result.error_message = f"Timeout after {self.timeout_seconds}s"
        except Exception as e:
            mutant.status = "error"
            result.error_message = str(e)

        return result

    def _apply_mutation(self, source: str, mutant: Mutant) -> str:
        """Apply a mutation to source code."""
        lines = source.split("\n")
        line_idx = mutant.line_number - 1

        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            # Replace original code with mutated code
            mutated_line = line.replace(mutant.original_code, mutant.mutated_code, 1)
            lines[line_idx] = mutated_line

        return "\n".join(lines)

    def _identify_weak_assertions(self) -> List[str]:
        """Identify potentially weak assertions in tests."""
        weak_assertions = []

        for result in self.results:
            if result.mutant.status == "survived":
                # Check if this is a comparison mutation that survived
                if result.mutant.mutation_type in [
                    MutationType.GT_TO_GE,
                    MutationType.LT_TO_LE,
                    MutationType.EQ_TO_NE,
                ]:
                    weak_assertions.append(
                        f"Weak comparison at {result.mutant.mutant_id}: "
                        f"{result.mutant.original_code} survived {result.mutant.mutation_type.name}"
                    )

        return weak_assertions

    def export_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Export mutation testing report."""
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate HTML report
        html_path = path.with_suffix(".html")
        self._generate_html_report(report, html_path)

        print(f"\nReports exported to {path.parent}")

    def _generate_html_report(self, report: Dict[str, Any], path: Path) -> None:
        """Generate HTML mutation report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>APGI Mutation Testing Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
        .header {{ background: #2c3e50; color: white; padding: 1.5rem; border-radius: 8px; }}
        .score {{ font-size: 3rem; font-weight: bold; }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .survived {{ background: #ffebee; }}
        tr:hover {{ background: #f5f5f5; }}
        pre {{ background: #f4f4f4; padding: 0.5rem; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>APGI Mutation Testing Report</h1>
        <div class="score {'good' if report['mutation_score'] >= 80 else 'warning' if report['mutation_score'] >= 50 else 'bad'}">
            {report['mutation_score']:.1f}%
        </div>
        <p>Mutation Score</p>
    </div>

    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Mutants</td><td>{report['total_mutants']}</td></tr>
        <tr><td>Killed</td><td class="good">{report['killed']}</td></tr>
        <tr><td>Survived</td><td class="{'good' if report['survived'] == 0 else 'bad'}">{report['survived']}</td></tr>
        <tr><td>Timeout</td><td>{report['timeout']}</td></tr>
        <tr><td>Error</td><td>{report['error']}</td></tr>
    </table>

    <h2>Survived Mutants (Require Attention)</h2>
    <table>
        <tr><th>ID</th><th>Type</th><th>Line</th><th>Original</th><th>Mutated</th></tr>
"""
        for mutant in report.get("survived_mutants", []):
            html += f"""
        <tr class="survived">
            <td>{mutant['id']}</td>
            <td>{mutant['type']}</td>
            <td>{mutant['line']}</td>
            <td><pre>{mutant['original']}</pre></td>
            <td><pre>{mutant['mutated']}</pre></td>
        </tr>
"""

        html += """
    </table>
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)


def run_mutation_testing():
    """Entry point for mutation testing."""
    tester = MutationTester(
        target_modules=[
            "utils/eeg_processing.py",
            "utils/statistical_tests.py",
            "utils/protocol_schema.py",
        ],
        test_modules=[
            "tests/test_eeg_processing.py",
            "tests/test_statistical_tests.py",
        ],
        timeout_seconds=10.0,
        parallel_workers=4,
    )

    report = tester.run()
    tester.export_report(report, "reports/mutation_report")

    return report


if __name__ == "__main__":
    run_mutation_testing()
