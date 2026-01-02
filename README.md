# ASI project. Group name: 11

## Setup

```bash
cd passStren
uv venv --python python3.10
uv sync
uv run dvc pull
```

## Usage

### PyCaret AutoML

```bash
uv run passstren pycaret --sample 0.01
```

### Optuna Hyperparameter Tuning

```bash
uv run passstren optuna --sample 0.05 --trials 20
```

### Predict Password Strength

```bash
uv run passstren predict "MyP@ssw0rd!"
```
