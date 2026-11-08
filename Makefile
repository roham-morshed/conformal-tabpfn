PY ?= python

.PHONY: setup download test smoke run budget crossover imbalance consolidate artifacts reproduce lint all clean

setup:          ## install pinned dependencies + the package itself
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

download:       ## THE ONLY NETWORKED TARGET: cache datasets + TabPFN weights
	$(PY) scripts/download_assets.py

test:           ## correctness gates
	$(PY) -m pytest tests/ -q

smoke:          ## ~60s end-to-end check on one tiny dataset
	$(PY) scripts/smoke_test.py

run:            ## main experiment grid (resumable)
	$(PY) scripts/run_experiments.py

budget:         ## RQ3 calibration-budget sweep
	$(PY) scripts/run_budget_sweep.py

crossover:      ## RQ6 n_train crossover study
	$(PY) scripts/run_crossover.py

imbalance:      ## RQ5 class-imbalanced variants
	$(PY) scripts/run_experiments.py --imbalance

consolidate:    ## merge per-cell parquets into results/all_results.parquet
	$(PY) scripts/consolidate_results.py

artifacts:      ## LaTeX tables + figures + results_summary.md
	$(PY) scripts/make_paper_artifacts.py

lint:
	$(PY) -m ruff check src scripts tests
	$(PY) -m ruff format --check src scripts tests

reproduce:      ## the full study, every stage, resumable
	bash scripts/reproduce_all.sh

all: test reproduce

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
