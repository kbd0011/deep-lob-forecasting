.PHONY: setup test lint typecheck check train
# One-command workflows. Seeds are fixed in code; Hydra configs drive training (reproducible).

setup:          ## install dependencies
	pip install -r requirements.txt

test:           ## run the test suite (CPU-only, tiny tensors)
	pytest -q

lint:           ## ruff lint
	ruff check src tests

typecheck:      ## mypy type check
	mypy src --ignore-missing-imports

check: lint typecheck test   ## lint + typecheck + test (what CI runs)

train:          ## train DeepLOB/TLOB on FI-2010 (drop Train_*/Test_* files into data/ first)
	python -m src.train
