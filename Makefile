.PHONY: setup test lint typecheck check train fi2010
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

fi2010:         ## download real FI-2010 (needs ~/.kaggle/kaggle.json) + report macro-F1 by horizon
	mkdir -p data
	kaggle datasets download wzy841127938/fi2010 -f NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt -p data/ --unzip
	kaggle datasets download wzy841127938/fi2010 -f NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt -p data/ --unzip
	PYTHONPATH=. python scripts/run_fi2010.py
