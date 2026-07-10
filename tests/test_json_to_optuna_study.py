from __future__ import annotations

import optuna
from optuna.trial import TrialState
import pytest

from jsonify_optuna import json_to_optuna_study
from jsonify_optuna import jsonify


def _single_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    return x**2


def _roundtrip(study: optuna.Study, **jsonify_kwargs: object) -> dict:
    json_data = jsonify(study, **jsonify_kwargs)
    reconstructed = json_to_optuna_study(json_data)
    return jsonify(reconstructed, **jsonify_kwargs)


def test_single_objective_roundtrip() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(_single_objective, n_trials=3)
    original = jsonify(study)
    result = _roundtrip(study)

    assert result == original


def test_multi_objective_with_metric_names_roundtrip() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.set_metric_names(["loss", "accuracy"])

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x

    study.optimize(objective, n_trials=5)
    original = jsonify(study)
    result = _roundtrip(study)

    assert result == original


def test_distribution_types_roundtrip() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int("i", 1, 10)
        trial.suggest_float("f", 0.0, 1.0)
        trial.suggest_float("f_log", 1e-5, 1.0, log=True)
        trial.suggest_float("f_step", 0.0, 1.0, step=0.1)
        trial.suggest_categorical("c", ["a", "b", "c"])
        return 0.0

    study.optimize(objective, n_trials=1)
    original = jsonify(study)
    result = _roundtrip(study)

    assert result == original


def test_int_distribution_step_and_log() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int("i_step", 0, 10, step=2)
        trial.suggest_int("i_log", 1, 100, log=True)
        return 0.0

    study.optimize(objective, n_trials=1)
    original = jsonify(study)
    result = _roundtrip(study)

    assert result == original


def test_pruned_trial_intermediate_values_roundtrip() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        for step in range(3):
            trial.report(float(step) * 0.5, step)
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=1)
    original = jsonify(study, states=[TrialState.PRUNED])
    result = _roundtrip(study, states=[TrialState.PRUNED])

    assert result == original


def test_failed_trial_roundtrip() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise RuntimeError("intentional failure")

    study.optimize(objective, n_trials=1, catch=(RuntimeError,))
    original = jsonify(study, states=[TrialState.FAIL])
    result = _roundtrip(study, states=[TrialState.FAIL])

    assert result == original


def test_user_attrs_roundtrip() -> None:
    study = optuna.create_study()
    study.set_user_attr("study_key", "study_value")

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("trial_key", "trial_value")
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=1)
    original = jsonify(study)
    result = _roundtrip(study)

    assert result == original


def test_no_metric_names() -> None:
    study = optuna.create_study()
    study.optimize(_single_objective, n_trials=1)
    json_data = jsonify(study)
    assert json_data["metric_names"] is None

    reconstructed = json_to_optuna_study(json_data)
    assert reconstructed.metric_names is None


def test_empty_trials() -> None:
    study = optuna.create_study()
    original = jsonify(study, states=[TrialState.PRUNED])
    assert original["trials"] == []

    result = _roundtrip(study, states=[TrialState.PRUNED])
    assert result == original


def test_directions_preserved() -> None:
    for directions in [["minimize"], ["maximize"], ["minimize", "maximize"]]:
        study = optuna.create_study(directions=directions)
        json_data = jsonify(study)
        reconstructed = json_to_optuna_study(json_data)
        assert jsonify(reconstructed)["directions"] == directions


def test_mixed_trial_states_roundtrip() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 0:
            raise optuna.TrialPruned()
        return x

    study.optimize(objective, n_trials=3)
    original = jsonify(study)
    result = _roundtrip(study)

    assert result == original
    states = {t["state"] for t in result["trials"]}
    assert "complete" in states
    assert "pruned" in states
