# APGI Validation Framework Makefile
# ==================================

.PHONY: help clean test lint install dev-install venv docs

help:
	@echo "Available targets:"
	@echo "  venv         - Create virtual environment"
	@echo "  clean        - Remove temporary files and cache directories"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run linting checks"
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install in development mode"
	@echo "  docs         - Generate documentation"

venv:
	python3 -m venv .venv

clean:
	@echo "Cleaning temporary files and cache directories..."
	python3 delete_pycache.py --yes

test:
	python3 -m pytest

lint:
	python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=APGI-Validation-Pipeline.py
	python3 -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=APGI-Validation-Pipeline.py

install: venv
	.venv/bin/pip install .

dev-install: venv
	.venv/bin/pip install -e .

docs:
	@echo "Documentation generation not yet implemented"
