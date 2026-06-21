# Group Equivariant Deep Learning

This repository contains our 10 ECTS DTU project on group equivariant deep
learning. The main experiment trains an E(3)-equivariant neural network on the
QM9 molecular dataset and compares models with different spherical harmonic
orders (`lmax`).

## Setup

The project uses `uv`. Dependencies are defined in `pyproject.toml` and locked
in `uv.lock`.

```bash
uv sync
```

## Run the Experiment

The main pipeline is run from `main.py`:

```bash
uv run python main.py
```

When `main.py` is run, it:

- loads/downloads QM9 through `src/Preprocessing.py`
- splits the data into train, validation, and test sets
- standardizes the target using only the training split
- builds the model from `src/E3Model.py`
- trains and evaluates through `src/Evaluation.py`
- saves results in `results/`

The main experiment settings are defined at the top of `main.py`:

```python
SEEDS = [0, 1, 2, 3, 4]
SUBSETS = [10000]
LMAXES = [0, 1]
DATA_SEED = 915
TARGET = 4
EPOCHS = 100

HPARAMS = {
    0: dict(batch_size=8, lr=0.0015977084099269045, weight_decay=1.0974137910112876e-06, dropout=0.010025864511046036, r=3.4531164194681523),
    1: dict(batch_size=8, lr=0.0021367384353078, weight_decay=0.009003150257857846, dropout=0.01901422634781214, r=3.007031871631741),
} # Here the optimized hyperparameters obtained from baysian_optimization.py
```

The run writes:

- `results/test.csv` with final test MAE for each run
- `results/l0_n10000.csv` with training/validation history for `lmax = 0`
- `results/l1_n10000.csv` with training/validation history for `lmax = 1`

## File Overview

- `main.py` - main experiment script for training and testing.
- `src/Preprocessing.py` - loads QM9, creates splits, standardizes the target,
  and returns PyTorch Geometric dataloaders.
- `src/E3Model.py` - defines the E(3)-equivariant model using e3nn tensor
  products, spherical harmonics, gating, and message passing.
- `src/Evaluation.py` - contains the training loop, validation/test evaluation,
  optimizer, scheduler.
- `baysian_optimization.py` - runs Bayesian hyperparameter optimization for one.
  model setting.
- `invariance_equivariance.ipynb` - notebook for exploring invariance and
  equivariance of models.
- `data_work.ipynb` - notebook for results and dataset analysis.
- `results/` - saved CSV results and plots.
- `project_files/` - project documents and reference material.

## Short Version

```bash
uv sync
uv run python main.py
```

No separate preprocessing script has to be run manually. The preprocessing is
called directly from `main.py`.
