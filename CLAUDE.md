# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APGI Validation Framework - A CLI-based scientific computing framework for validating and falsifying active inference models through computational experiments.

## Development Commands

### Run the CLI application

```bash
python main.py --help                       # Show all available commands
python main.py formal-model --help          # Show formal model simulation options
python main.py formal-model                 # Run formal model simulation with defaults
python main.py formal-model --simulation-steps 500 --dt 0.01 --plot  # Custom steps, plot
python main.py formal-model --output-file results.csv --params config/custom_params.json
```

### Install dependencies

```bash
pip install -r requirements.txt              # Core dependencies
pip install -r requirements-dev.txt          # Development dependencies (includes test/lint tools)
```

### Testing

```bash
pytest                                              # All tests
pytest tests/test_cli_integration_comprehensive.py    # CLI integration tests
pytest tests/test_utility_modules_comprehensive.py   # Utility module tests
pytest tests/test_file_io_real.py                  # Real file I/O tests
pytest tests/test_data_pipeline_end_to_end.py      # Data pipeline tests
pytest tests/test_property_based_comprehensive.py   # Property-based tests (Hypothesis)
pytest tests/test_validation_falsification_protocols_individual.py  # Protocol tests
pytest tests/test_fixtures_utilization.py          # Fixture tests
pytest tests/test_performance_regression.py        # Performance tests
pytest tests/test_concurrent_config_access.py      # Concurrency tests
pytest tests/test_persistent_audit_logger.py       # Audit logger tests
pytest --hypothesis-profile=ci                      # Full Hypothesis runs (100 examples)
```

### Linting and formatting

```bash
black .                                            # Format code
isort .                                           # Sort imports
flake8 .                                          # Lint code (Config in .flake8)
```

## Architecture

### Main entry point

`main.py` defines the unified CLI interface using `click` with commands for validation, falsification, visualization, and benchmarking. It includes secure module loading (`secure_load_module`, `secure_load_module_from_path`) with path validation to prevent directory traversal attacks.

### Configuration

`config/` directory contains YAML configuration files:

- `default.yaml` - Default configuration settings
- `config_schema.json` - Configuration schema validation
- `profiles/` - Environment-specific profiles (adhd.yaml, anxiety-disorder.yaml, etc.)
- `versions/` - Version-specific configurations

### Key directories

- `Validation/` - Validation protocols (VP_3_ActiveInference_AgentSimulations_Protocol3.py, BayesianModelComparison_ParameterRecovery.py, etc.)
- `Falsification/` - Falsification protocols (FP_12_Falsification_Aggregator.py, CausalManipulations_TMS_Pharmacological_Priority2.py, etc.)
- `utils/` - Utility modules (dependency_scanner.py, security_audit_logger.py, backup_manager.py, batch_processor.py, etc.)
- `data_repository/` - Data directory for input/output files (organized: raw_data/, processed_data/, metadata/, codebooks/, dashboard_data/)
- `tests/` - Comprehensive test suite with property-based testing (Hypothesis)
- `docs/` - Documentation (APGI_Equations.md, APGI-Parameter-Specifications.md, etc.)

### Security features

- Secure module loading with path validation
- Audit logging via `SecurityAuditLogger` and persistent audit logging
- Environment variable validation for required keys (`PICKLE_SECRET_KEY`, `APGI_BACKUP_HMAC_KEY`)
- Dependency vulnerability scanning via `DependencyScanner` (pip-audit)
- Thread-safe configuration access with `_config_lock`

### Key Python modules

- `APGI_Equations.py` - Core APGI equations (entropy, KL divergence, free energy, etc.)
- `APGI_Entropy_Implementation.py` - Entropy implementation
- `APGI_Full_Dynamic_Model.py` - Full dynamic model
- `APGI_Liquid_Network_Implementation.py` - Liquid network implementation
- `APGI_Multimodal_Classifier.py` - Multimodal classifier
- `APGI_Multimodal_Integration.py` - Multimodal integration

### Test setup

Unit tests use pytest with Hypothesis for property-based testing. The `conftest.py` provides fixtures including `temp_dir`, `sample_config`, `sample_data`, `raises_fixture`, `oom_fixture`, and `flaky_operation`. Hypothesis profiles: `dev` (20 examples, default locally), `ci` (100 examples), `thorough` (1000 examples).

## Required Environment Variables

Minimum for development:

```bash
PICKLE_SECRET_KEY=<random 64-character hex string>
APGI_BACKUP_HMAC_KEY=<random 64-character hex string>
```

Generate keys: `python -c "import os; print(os.urandom(32).hex())"`

For production, additional security keys and settings may be required.

## CLI Commands

### Simulation commands

- `formal-model` - Run formal model simulations with configurable parameters
  - `--simulation-steps` - Number of simulation steps (default: from config)
  - `--dt` - Time step size (default: from config)
  - `--output-file` - Output file for results (.csv, .json, or .pkl)
  - `--params` - JSON file with custom model parameters
  - `--plot` - Generate visualization plots

### Testing commands

Run GUI test runners:

- `python Tests-GUI.py` - Launch the GUI test runner
- `python Utils-GUI.py` - Launch the utility scripts GUI

## Data formats

Input data formats:

- JSON files for structured data
- CSV files for time series and tabular data
- Pickle files for serialized Python objects

Output formats:

- JSON reports for validation/falsification results
- PNG/SVG plots for visualizations
- Pickle files for model checkpoints

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **apgi-validation** (11410 symbols, 29713 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/apgi-validation/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/apgi-validation/context` | Codebase overview, check index freshness |
| `gitnexus://repo/apgi-validation/clusters` | All functional areas |
| `gitnexus://repo/apgi-validation/processes` | All execution flows |
| `gitnexus://repo/apgi-validation/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
