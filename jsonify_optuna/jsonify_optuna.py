from __future__ import annotations

from typing import TYPE_CHECKING

import optuna
from optuna.trial import TrialState


if TYPE_CHECKING:
    from typing import Any
    from typing import Literal
    from typing import TypedDict

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
            if isinstance(d, optuna.distributions.CategoricalDistribution):
                dists[p] = {"choices": d.choices}
                continue
            assert isinstance(
                d, (optuna.distributions.IntDistribution, optuna.distributions.FloatDistribution)
            )
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
                f"{storage=} is provided but either {rdb_url=} or {journal_path=} is also specified."
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
