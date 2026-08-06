PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help setup run clean test

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
	@echo "  make clean"
	@echo "      Remove the virtual environment."
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run ARGS=\"--help\""
	@echo "  make run ARGS=\"input.txt --debug\""

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi

run:
	$(PYTHON_VENV) -m src.main $(ARGS)

test: setup
	$(PYTHON_VENV) -m tests.tests

clean:
	rm -rf $(VENV)