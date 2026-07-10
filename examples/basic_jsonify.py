from __future__ import annotations

import json

import optuna

from jsonify_optuna import jsonify


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_int("y", 1, 5)
    return (x - 2) ** 2 + y


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    result = jsonify(study)
    print(json.dumps(result, indent=2))
