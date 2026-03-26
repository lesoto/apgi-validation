# APGI Validation Framework Makefile
# ==================================

.PHONY: help clean test lint threshold-lint install dev-install venv docs

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
	python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics 
	python3 -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	$(MAKE) threshold-lint

threshold-lint:
	python3 scripts/threshold_lint.py

install: venv
	.venv/bin/pip install .

dev-install: venv
	.venv/bin/pip install -e .

docs:
	@echo "Generating documentation..."
	@echo "Converting Markdown to HTML..."
	@command -v pandoc >/dev/null 2>&1 || { echo "pandoc not installed. Install with: brew install pandoc (macOS) or apt install pandoc (Ubuntu)"; exit 1; }
	@pandoc docs/APGI_Equations.md -o docs/APGI_Equations.html --metadata title="APGI Equations" --css=github-markdown.css
	@pandoc docs/APGI-Parameter-Specifications.md -o docs/APGI-Parameter-Specifications.html --metadata title="APGI Parameter Specifications" --css=github-markdown.css
	@pandoc docs/APGI-Empirical-Credibility-Roadmap.md -o docs/APGI-Empirical-Credibility-Roadmap.html --metadata title="APGI Empirical Credibility Roadmap" --css=github-markdown.css
	@pandoc docs/APGI-Falsification-Criteria.md -o docs/APGI-Falsification-Criteria.html --metadata title="APGI Falsification Criteria" --css=github-markdown.css
	@pandoc docs/Tutorial.md -o docs/Tutorial.html --metadata title="APGI Tutorial" --css=github-markdown.css
	@pandoc docs/Innovations.md -o docs/Innovations.html --metadata title="APGI Innovations" --css=github-markdown.css
	@pandoc docs/Innovations-Software-Implementation.md -o docs/Innovations-Software-Implementation.html --metadata title="APGI Software Implementation" --css=github-markdown.css
	@pandoc docs/Multimodal-Integration.md -o docs/Multimodal-Integration.html --metadata title="APGI Multimodal Integration" --css=github-markdown.css
	@echo "Documentation generated in docs/ directory"
	@echo "Open HTML files in a web browser to view"
