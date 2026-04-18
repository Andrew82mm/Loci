PYTHON ?= python
PYTEST  = $(PYTHON) -m pytest
RUFF    = $(PYTHON) -m ruff
MYPY    = $(PYTHON) -m mypy

.PHONY: test lint typecheck run

test:
	$(PYTEST)

lint:
	$(RUFF) check .

typecheck:
	$(MYPY) --ignore-missing-imports .

run:
	$(PYTHON) main.py
