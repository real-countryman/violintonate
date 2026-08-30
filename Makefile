.SILENT:
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
ARGS ?=

SPHINX_BUILD := $(VENV)/bin/sphinx-build
SPHINX_APIDOC := $(VENV)/bin/sphinx-apidoc

DOCS_DIR := docs/source
DOCS_API_DIR := $(DOCS_DIR)/api
DOCS_BUILD_DIR := docs/_build/html

.DEFAULT_GOAL := help

.PHONY: help setup run clean test audacity docs

help:
	@echo "Usage:"
	@echo "  make setup"
	@echo "      Create the virtual environment and install dependencies."
	@echo ""
	@echo "  make run"
	@echo "      Set up the virtual environment, then run main.py."
	@echo ""
	@echo "  make run ARGS=\"foo bar --debug\""
	@echo "      Pass arguments to main.py."
	@echo ""
	@echo "  make audacity ARGS=\"times_json_path violin_audio_path output_aup3_path\""
	@echo ""
	@echo "  make clean"
	@echo "      Remove the virtual environment."
	@echo ""
	@echo "  make docs"
	@echo "      Generates the documentation"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run ARGS=\"--help\""
	@echo "  make run ARGS=\"input.txt --debug\""

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -qq --upgrade pip
	@if [ -f requirements.txt ]; then $(PIP) install -qq -r requirements.txt; fi

run: setup
	$(PYTHON_VENV) -m src.main $(ARGS)

audacity: setup
	audacity & \
	echo "Waiting for 10 seconds for Audacity to start"; \
	sleep 10; \
	$(PYTHON_VENV) -m src.audacity_pipeline \
		$(word 2,$(ARGS)) \
		$(word 3,$(ARGS)) \
		< $(word 1,$(ARGS))

test: setup
	$(PYTHON_VENV) -m pytest tests/

SPHINX_APIDOC := $(VENV)/bin/sphinx-apidoc

docs: setup
	$(SPHINX_APIDOC) -f -e -o $(DOCS_API_DIR) src
	$(SPHINX_BUILD) -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)
	@echo "Documentation generated at $(DOCS_BUILD_DIR)/index.html"

clean:
	rm -rf $(VENV)
	rm -rf $(DOCS_DIR)/_build
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf $(DOCS_API_DIR)