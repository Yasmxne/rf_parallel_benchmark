# Adaptive Parallel Benchmark for Random Forest

This project compares two ways of choosing the best parallel configuration for a Random Forest.

The parameter studied is `n_jobs`, which controls how many CPU workers are used during training. Since a Random Forest is made of many decision trees, several trees can be trained in parallel.

## Goal

The goal is to find a good value of `n_jobs` without testing every possible configuration.

We compare:

- a full grid search: tests all values of `n_jobs`
- an adaptive benchmark: starts with a few values, then tests only around the best one

## Project structure

```text
rf_parallel_benchmark/
├── requirements.txt
├── README.md
├── src/
│   ├── benchmark.py
│   ├── adaptive_search.py
│   └── run_experiment.py
├── results/
└── report/
    └── figures/
```

## Installation

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```bash
python src/run_experiment.py
```

## Outputs

The results are saved in:

```text
results/grid_search_results.csv
results/adaptive_search_results.csv
results/summary.csv
```

The figures are saved in:

```text
report/figures/
```

## Metrics

For each value of `n_jobs`, we measure:

- training time
- accuracy
- speedup
- parallel efficiency

Speedup is computed as:

```text
speedup = time with n_jobs=1 / time with n_jobs=k
```

## Main idea

Using more workers is not always faster. After some point, the overhead of parallelization can become larger than the gain.

The adaptive benchmark tries to find a good configuration with fewer tests than a full grid search.