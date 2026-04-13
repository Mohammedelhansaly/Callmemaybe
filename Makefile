PYTHON = python3


install:
	uv sync
run: install
	uv run -m src --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
debug: install
	uv run -m pdb src/__main__.py
lint: 
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
clean:
	rm -rf */__pycache__ .mypy_cache