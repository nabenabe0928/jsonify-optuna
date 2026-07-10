from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypedDict

import optuna
from optuna.trial import TrialState
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution


if TYPE_CHECKING:
    from typing import Any
    from typing import Literal

    from optuna.distributions import CategoricalChoiceType

    class NumericalDistributionType(TypedDict):
        low: int | float
        high: int | float
        step: int | float | None
        log: bool

    class CategoricalDistributionType(TypedDict):
        choices: list[CategoricalChoiceType]


class TrialType(TypedDict):
    state: Literal["running", "waiting", "completepruned", "fail"]
    values: list[float]
    params: dict[str, int | float | CategoricalChoiceType]
    user_attrs: dict[str, Any]
    intermediate_values: list[float]
    distributions: dict[str, NumericalDistributionType | CategoricalDistributionType]


class StudyType:
    trials: list[TrialType]
    best_trial_indices: list[int]
    directions: list[Literal["minimize", "maximize"]]
    user_attrs: dict[str, Any]
    metric_names: list[str]


def jsonify(
    study: optuna.Study, *, states: list[TrialState] | None = None, deepcopy: bool = True
) -> StudyType:
    states = states or [state for state in TrialState]
    study_json = {
        "directions": [dire.name.lower() for dire in study.directions],
        "user_attrs": study.user_attrs,
        "metric_names": study.metric_names,
    }
    best_trial_numbers = set(t.number for t in study.best_trials)
    best_trial_indices = []
    results = []
    for t in study.get_trials(deepcopy=deepcopy):
        if t.state not in states:
            continue
        if t.number in best_trial_numbers:
            best_trial_indices.append(len(results))
        dists = {}
        for p, d in t.distributions.items():
            if isinstance(d, CategoricalDistribution):
                dists[p] = {"choices": d.choices}
                continue
            assert isinstance(d, (IntDistribution, FloatDistribution))
            dists[p] = {"low": d.low, "high": d.high, "step": d.step, "log": d.log}
        row = {
            "state": t.state.name.lower(),
            "values": t.values,
            "params": t.params,
            "user_attrs": t.user_attrs,
            "intermediate_values": t.intermediate_values,
            "distributions": dists,
        }
        results.append(row)
    study_json |= {"trials": results, "best_trial_indices": best_trial_indices}
    return study_json


def json_to_optuna_study(study_json: StudyType) -> optuna.Study:
    study = optuna.create_study(directions=study_json["directions"])
    for key, value in study_json["user_attrs"].items():
        study.set_user_attr(key, value)
    if study_json["metric_names"] is not None:
        study.set_metric_names(study_json["metric_names"])
    state_map = {s.name.lower(): s for s in TrialState}
    for trial_json in study_json["trials"]:
        dists: dict[str, optuna.distributions.BaseDistribution] = {}
        for name, dist_json in trial_json["distributions"].items():
            if (choices := dist_json.get("choices")) is not None:
                dists[name] = CategoricalDistribution(choices)
                continue
            low = dist_json["low"]
            high = dist_json["high"]
            step = dist_json["step"]
            log = dist_json["log"]
            if all(isinstance(v, int) for v in [low, high, step]):
                dists[name] = IntDistribution(low=low, high=high, step=step, log=log)
            else:
                dists[name] = FloatDistribution(low=low, high=high, step=step, log=log)
        trial = optuna.trial.create_trial(
            state=state_map[trial_json["state"]],
            values=trial_json["values"],
            params=trial_json["params"],
            distributions=dists,
            user_attrs=trial_json["user_attrs"],
            intermediate_values=trial_json["intermediate_values"] or None,
        )
        study.add_trial(trial)
    return study


def load_study(
    *,
    study_name: str | None = None,
    storage: optuna.storages.BaseStorage | None = None,
    journal_path: str | None = None,
    rdb_url: str | None = None,
) -> optuna.Study:
    if storage is not None:
        if rdb_url is not None or journal_path is not None:
            raise ValueError(
                f"storage is provided but either {rdb_url=} or {journal_path=} is also specified."
            )
    elif rdb_url is not None:
        if journal_path is not None:
            raise ValueError(f"Specify only one of {rdb_url=} or {journal_path=}.")
        storage = optuna.storages.RDBStorage(rdb_url)
    elif journal_path:
        journal_file = optuna.storages.journal.JournalFileBackend(journal_path)
        storage = optuna.storages.JournalStorage(journal_file)
    else:
        raise ValueError("No storage was specified.")
    assert isinstance(storage, optuna.storages.BaseStorage), "MyPy Redefinition"
    study_names = [study.study_name for study in storage.get_all_studies()]
    if study_name is None and len(study_names) == 1:
        return optuna.load_study(storage=storage, study_name=study_names[0])
    if study_name is None:
        raise ValueError(f"Specify `study_name` from {study_names=}.")
    if study_name not in study_names:
        raise ValueError(f"Specify `study_name` from {study_names=} but got {study_name=}.")
    return optuna.load_study(storage=storage, study_name=study_name)
