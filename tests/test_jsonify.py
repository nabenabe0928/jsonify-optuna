from __future__ import annotations

import optuna
from optuna.trial import TrialState
import pytest

from jsonify_optuna import jsonify


def _single_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    return x**2


def test_single_objective_complete_trial() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(_single_objective, n_trials=3)
    result = jsonify(study)

    assert result["directions"] == ["minimize"]
    assert len(result["trials"]) == 3
    assert result["metric_names"] is None

    trial = result["trials"][0]
    assert set(trial.keys()) == {
        "state",
        "values",
        "params",
        "user_attrs",
        "intermediate_values",
        "distributions",
    }
    assert trial["state"] == "complete"
    assert isinstance(trial["values"], list) and len(trial["values"]) == 1

    assert len(result["best_trial_indices"]) == 1
    best_idx = result["best_trial_indices"][0]
    assert 0 <= best_idx < len(result["trials"])


def test_multi_objective_with_metric_names() -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.set_metric_names(["loss", "accuracy"])

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x

    study.optimize(objective, n_trials=5)
    result = jsonify(study)

    assert result["directions"] == ["minimize", "maximize"]
    assert result["metric_names"] == ["loss", "accuracy"]
    for t in result["trials"]:
        assert isinstance(t["values"], list) and len(t["values"]) == 2
    assert len(result["best_trial_indices"]) >= 1
    for idx in result["best_trial_indices"]:
        assert 0 <= idx < len(result["trials"])


def test_distribution_types() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int("i", 1, 10)
        trial.suggest_float("f", 0.0, 1.0)
        trial.suggest_float("f_log", 1e-5, 1.0, log=True)
        trial.suggest_float("f_step", 0.0, 1.0, step=0.1)
        trial.suggest_categorical("c", ["a", "b", "c"])
        return 0.0

    study.optimize(objective, n_trials=1)
    dists = jsonify(study)["trials"][0]["distributions"]

    for name in ("i", "f", "f_log", "f_step"):
        assert set(dists[name].keys()) == {"low", "high", "step", "log"}
    assert dists["i"]["low"] == 1
    assert dists["i"]["high"] == 10
    assert dists["f"]["log"] is False and dists["f"]["step"] is None
    assert dists["f_log"]["log"] is True
    assert dists["f_step"]["step"] == 0.1

    assert set(dists["c"].keys()) == {"choices"}
    assert list(dists["c"]["choices"]) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "filter_state, expected_state_name",
    [
        (TrialState.COMPLETE, "complete"),
        (TrialState.PRUNED, "pruned"),
    ],
)
def test_filter_by_states(filter_state: TrialState, expected_state_name: str) -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number % 2 == 0:
            raise optuna.TrialPruned()
        return x

    study.optimize(objective, n_trials=4)
    result = jsonify(study, states=[filter_state])

    assert len(result["trials"]) > 0
    for t in result["trials"]:
        assert t["state"] == expected_state_name


def test_pruned_trial_intermediate_values() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        for step in range(3):
            trial.report(float(step) * 0.5, step)
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=1)
    result = jsonify(study, states=[TrialState.PRUNED])

    iv = result["trials"][0]["intermediate_values"]
    assert isinstance(iv, dict)
    assert iv == {0: 0.0, 1: 0.5, 2: 1.0}


def test_failed_trial_has_none_values() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise RuntimeError("intentional failure")

    study.optimize(objective, n_trials=1, catch=(RuntimeError,))
    result = jsonify(study, states=[TrialState.FAIL])

    assert len(result["trials"]) == 1
    assert result["trials"][0]["state"] == "fail"
    assert result["trials"][0]["values"] is None


def test_user_attrs() -> None:
    study = optuna.create_study()
    study.set_user_attr("study_key", "study_value")

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("trial_key", "trial_value")
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=1)
    result = jsonify(study)

    assert result["user_attrs"] == {"study_key": "study_value"}
    assert result["trials"][0]["user_attrs"] == {"trial_key": "trial_value"}


def test_best_trial_indices_in_filtered_results() -> None:
    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 0:
            raise optuna.TrialPruned()
        return x

    study.optimize(objective, n_trials=3)
    result = jsonify(study, states=[TrialState.COMPLETE])

    assert all(t["state"] == "complete" for t in result["trials"])
    for idx in result["best_trial_indices"]:
        assert 0 <= idx < len(result["trials"])
        assert result["trials"][idx]["state"] == "complete"


def test_deepcopy_false() -> None:
    study = optuna.create_study()
    study.optimize(_single_objective, n_trials=2)
    result = jsonify(study, deepcopy=False)

    assert len(result["trials"]) == 2
    assert result["directions"] == ["minimize"]


def test_no_matching_trials() -> None:
    study = optuna.create_study()
    study.optimize(_single_objective, n_trials=2)
    result = jsonify(study, states=[TrialState.PRUNED])

    assert result["trials"] == []
    assert result["best_trial_indices"] == []


def test_default_states_includes_all() -> None:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 0:
            raise optuna.TrialPruned()
        return x

    study.optimize(objective, n_trials=3)
    result = jsonify(study)

    states = {t["state"] for t in result["trials"]}
    assert "complete" in states
    assert "pruned" in states
    assert len(result["trials"]) == 3
