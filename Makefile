VENV := .venv
PYTHON := $(VENV)/bin/python
FUNCTIONS_DEFINITION := data/input/functions_definition.json
FUNCTION_CALLING_TESTS := data/input/function_calling_tests.json
OUTPUT := data/output/function_calls.json
SRC_DIR := src

run:
	uv run python -m $(SRC_DIR) --functions_definition $(FUNCTIONS_DEFINITION) --input $(FUNCTION_CALLING_TESTS) --output $(OUTPUT)
	
help:
	@echo "Available targets:"
	@echo "  make venv       			- Create virtualenv using uv"
	@echo "  make install    			- Install dependencies from requirements.txt using uv"
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

install: venv
	@if [ -f requirements.txt ]; then \
		uv pip install --python $(PYTHON) -r requirements.txt; \
	else \
		echo "No requirements.txt found; skipping."; \
	fi
	@echo "\nsource $(VENV)/bin/activate && make"

lint:
	flake8 . --exclude=.git,.venv,venv,env,build,dist,.mypy_cache,.pytest_cache,__pycache__,dependencies,*.egg-info,llm_sdk
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs --exclude='(build|dist|venv|env|dependencies|llm_sdk)'

lint-strict:
	flake8 . --exclude=.git,.venv,venv,env,build,dist,.mypy_cache,.pytest_cache,__pycache__,dependencies,*.egg-info,llm_sdk
	mypy . --strict --exclude='(build|dist|venv|env|dependencies|llm_sdk)'

clean:
	rm -rf $(VENV)
	rm -rf .mypy_cache

.PHONY: help venv install run clean lint lint-strict