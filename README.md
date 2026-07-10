# jsonify-optuna

Convert an [Optuna](https://github.com/optuna/optuna) Study into a plain, JSON-serializable Python dict.

> [!IMPORTANT]
> If you would like to compress the resulting JSON file the most, use the `gzip` compression.

## Installation

```bash
pip install jsonify-optuna
```

## Usage

Convert an in-memory study to a JSON-friendly dict:

```python
import optuna
from jsonify_optuna import jsonify
from optuna.trial import TrialState

def objective(trial):
    x = trial.suggest_float("x", -10.0, 10.0)
    return x ** 2

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
result = jsonify(study, states=[TrialState.COMPLETE])
# This includes all the trials.
result = jsonify(study)
```

Reconstruct an `optuna.Study` from a JSON dict:

```python
from jsonify_optuna import json_to_optuna_study

study = json_to_optuna_study(result)
```

If you have a study stored in a backend, you can load it as follows:

```python
from jsonify_optuna import load_study

# From an RDB (SQLite, PostgreSQL, etc.)
study = load_study(rdb_url="sqlite:///example.db", study_name="my_study")

# From a journal file
study = load_study(journal_path="journal.log", study_name="my_study")

# From a storage object directly
study = load_study(storage=my_storage, study_name="my_study")
```

> [!NOTE]
> When the storage contains exactly one study, `study_name` can be omitted.

## API Reference

### `jsonify(study, *, states=None, deepcopy=True)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `study` | `optuna.Study` | The study to convert. |
| `states` | `list[TrialState] \| None` | Filter trials by state. `None` includes all states. |
| `deepcopy` | `bool` | Whether to deep-copy trials from storage. Default `True`. |

Returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `directions` | `list[str]` | `"minimize"` or `"maximize"` per objective. |
| `user_attrs` | `dict` | Study-level user attributes. |
| `metric_names` | `list[str] \| None` | Metric names if set, else `None`. |
| `trials` | `list[dict]` | Trial dicts (see below). |
| `best_trial_indices` | `list[int]` | Indices into `trials` for Pareto-front trials. |

Each trial dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `state` | `str` | `"complete"`, `"pruned"`, `"fail"`, `"running"`, or `"waiting"`. |
| `values` | `list[float] \| None` | Objective values. `None` if not complete. |
| `params` | `dict` | Parameter name-value pairs. |
| `user_attrs` | `dict` | Trial-level user attributes. |
| `intermediate_values` | `dict[int, float]` | Step-value pairs from `trial.report()`. |
| `distributions` | `dict` | Parameter distributions (see below). |

Distribution formats:

- **Int/Float**: `{"low": ..., "high": ..., "step": ..., "log": ...}`
- **Categorical**: `{"choices": [...]}`

### `json_to_optuna_study(study_json)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `study_json` | `StudyType` | A dict produced by `jsonify()`. |

Returns an in-memory `optuna.Study` with directions, user attributes, metric names, and all trials reconstructed from the JSON dict.

### `load_study(*, study_name=None, storage=None, journal_path=None, rdb_url=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `study_name` | `str \| None` | Study name. Optional when storage has exactly one study. |
| `storage` | `BaseStorage \| None` | An Optuna storage object. |
| `journal_path` | `str \| None` | Path to a journal log file. |
| `rdb_url` | `str \| None` | SQLAlchemy-style database URL. |

Exactly one of `storage`, `journal_path`, or `rdb_url` must be provided.
