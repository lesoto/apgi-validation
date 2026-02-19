# APGI Validation Framework Makefile
# ==================================

.PHONY: help clean test lint install dev-install docs

help:
	@echo "Available targets:"
	@echo "  clean        - Remove temporary files and cache directories"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run linting checks"
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install in development mode"
	@echo "  docs         - Generate documentation"

clean:
	@echo "Cleaning temporary files and cache directories..."
	python delete_pycache.py --yes

test:
	pytest

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

install:
	pip install .

dev-install:
	pip install -e .

docs:
	@echo "Documentation generation not yet implemented"
