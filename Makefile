.PHONY: install test lint benchmark detection app notebooks scenarios

install:
	pip install -e ".[llm,app,dev]"

test:
	pytest

lint:
	ruff check src tests app && ruff format --check src tests app

scenarios:
	kpi-detective scenarios

benchmark:
	kpi-detective benchmark --split test
	kpi-detective benchmark --split dev

detection:
	python -m kpi_detective.evaluation.evaluate_detection --scenarios data/scenarios/test --out results/test
	python -m kpi_detective.evaluation.evaluate_detection --scenarios data/scenarios/dev --out results/dev

app:
	streamlit run app/streamlit_app.py

notebooks:
	python scripts/build_notebooks.py
