install:
	pip install -e .[dev]

data:
	python scripts/download_data.py

inspect:
	python scripts/inspect_data.py

test:
	pytest -q

baseline:
	python -m lucidbench.train_baseline --config configs/baseline.yaml
