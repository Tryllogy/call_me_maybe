VENV := .venv
PYTHON := $(VENV)/bin/python
FUNCTIONS_DEFINITION := data/input/functions_definition.json
FUNCTION_CALLING_TESTS := data/input/function_calling_tests.json
OUTPUT := data/output/function_calls.json
SRC_DIR := src

install:
	uv sync

run: install
	uv run python -m $(SRC_DIR) --functions_definition $(FUNCTIONS_DEFINITION) --input $(FUNCTION_CALLING_TESTS) --output $(OUTPUT)
	
help:
	@echo "Available targets:"
	@echo "  make venv       			- Create virtualenv using uv"
	@echo "  make install    			- Install dependencies from pyproject.toml using uv"
	@echo "  make run        			- Run the project with specified input and output"
	@echo "  make clean      			- Remove $(VENV) and .mypy_cache"
	@echo "  make lint         			- Run flake8 and mypy with standard checks"
	@echo "  make lint-strict  			- Run flake8 and mypy with strict mode"

venv:
	@if [ -d "$(VENV)" ]; then \
		echo "Virtualenv exists at $(VENV)"; \
	else \
		uv venv $(VENV); \
	fi

debug:
	uv run python -m pdb src/__main__.py --functions_definition $(FUNCTIONS_DEFINITION) --input $(FUNCTION_CALLING_TESTS) --output $(OUTPUT)

lint:
	uv run flake8 $(SRC_DIR)/*.py
	uv run mypy $(SRC_DIR)/*.py --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(SRC_DIR)/*.py
	uv run mypy $(SRC_DIR)/*.py --strict

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache __pycache__ data/output

.PHONY: help venv install run clean lint lint-strict