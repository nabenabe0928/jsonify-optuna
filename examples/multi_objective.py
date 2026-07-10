from __future__ import annotations

import json

import optuna

from jsonify_optuna import jsonify


def objective(trial: optuna.Trial) -> tuple[float, float]:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
    loss = lr * n_layers + (0.1 if activation == "relu" else 0.5)
    accuracy = 1.0 - loss
    return loss, accuracy


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.set_metric_names(["loss", "accuracy"])
    study.optimize(objective, n_trials=30)

    result = jsonify(study)
    print(f"Directions: {result['directions']}")
    print(f"Metric names: {result['metric_names']}")
    print(f"Total trials: {len(result['trials'])}")
    print(f"Pareto front size: {len(result['best_trial_indices'])}")
    print(json.dumps(result, indent=2))
