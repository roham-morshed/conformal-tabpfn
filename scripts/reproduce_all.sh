#!/usr/bin/env bash
# Reproduce the whole study end to end, in the order it was run.
# Every stage is resumable: each cell writes its own parquet keyed by a hash of
# the cell, and finished cells are skipped, so an interrupted run can simply be
# restarted.  Run `make download` once first (the only networked step).
set -euo pipefail
cd "$(dirname "$0")/.."

stage () { echo; echo "== $1  ($(date +%H:%M:%S))"; shift; python -u "$@"; }

stage "main grid"         scripts/run_experiments.py
stage "RQ3 budget sweep"  scripts/run_budget_sweep.py
stage "RQ6 crossover"     scripts/run_crossover.py
stage "RQ5 imbalance"     scripts/run_experiments.py --imbalance
# RQ1 jackknife+ timing arm: K = n_train, so it is kept deliberately small.
stage "RQ1 jackknife+"    scripts/run_experiments.py --cp-methods jackknife_plus \
                            --datasets wdbc climate-model --seeds 3 --n-train 300 \
                            --jackknife-max-n 300
stage "consolidate"       scripts/consolidate_results.py
stage "paper artifacts"   scripts/make_paper_artifacts.py

echo
echo "cells:    $(ls results/raw/*.parquet 2>/dev/null | wc -l)"
echo "failures: $(ls results/failures/*.json 2>/dev/null | wc -l)"
