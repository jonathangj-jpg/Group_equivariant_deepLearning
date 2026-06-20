# Group Equivariant Deep Learning

This is the git repo for our 10 ECTS project at DTU about group equivariant deep learning.

## Setup

The project uses `uv`, and the dependencies are listed in `pyproject.toml`.

```bash
uv sync
```

If you do not use `uv`, the scripts can also be run with regular `python`, as long as the dependencies from `pyproject.toml` are installed.

## Run Order

### 1. Download QM9 data and check the dataset

```bash
uv run python qm9data.py
```

`qm9data.py` downloads QM9 to `./qm9_data` the first time the script is run and prints basic information about the first molecule.

This is not a required training step, but it is a useful sanity check to verify that QM9 can be downloaded and read correctly.

### 2. Preprocessing

There is no separate preprocessing script that has to be run manually.

Preprocessing is implemented in `src/Preprocessing.py` as the function `load_qm9(...)`. It is called directly from `main.py` and does the following:

- loads QM9 from `./qm9_data`
- splits the dataset into train, validation, and test sets
- optionally selects a subset of the training data
- standardizes the target value using only the training split
- returns `DataLoader`s for training, validation, and testing

If `./qm9_data` does not exist yet, `QM9(...)` automatically downloads the dataset when `load_qm9(...)` is called.

### 3. Train the model and save results

```bash
uv run python main.py
```

`main.py` is the main script for the experiments. It:

- calls `load_qm9(...)` from `src/Preprocessing.py`
- creates the model from `src/E3Model.py`
- trains and evaluates through `src/Evaluation.py`
- runs over the defined seeds, subset sizes, and `lmax` values
- saves CSV results in `results/`

The most important settings are defined at the top of `main.py`:

```python
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
SUBSETS = [5000, 25000]
LMAXES = [0, 1]
TARGET = 4
BATCH_SIZE = 32
EPOCHS = 100
```

The results are written to:

- `results/test.csv`
- `results/l0_n5000.csv`
- `results/l1_n5000.csv`
- `results/l0_n25000.csv`
- `results/l1_n25000.csv`

## File Overview

- `qm9data.py` - imports/checks the data
- `main.py` - runs training and testing
- `src/Preprocessing.py` - handles preprocessing
- `src/E3Model.py` - contains the model architecture
- `src/Evaluation.py` - handles evaluation
- `invariance_equivariance.ipynb` - shows invariance and equivariance of the convolution

## Short Version

If you just want to run the QM9 pipeline:

```bash
uv sync
uv run python qm9data.py
uv run python main.py
```

`qm9data.py` is only for download/checking the dataset. The actual preprocessing and training happen automatically when `main.py` is run.
