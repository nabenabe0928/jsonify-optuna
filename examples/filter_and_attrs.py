from __future__ import annotations

import json

import optuna
from optuna.trial import TrialState

from jsonify_optuna import jsonify


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0.0, 10.0)
    trial.set_user_attr("note", f"trial {trial.number}")
    for step in range(5):
        intermediate = x + step
        trial.report(intermediate, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return x**2


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.set_user_attr("experiment", "demo")
    study.optimize(objective, n_trials=20)

    complete_only = jsonify(study, states=[TrialState.COMPLETE])
    print(f"Study user_attrs: {complete_only['user_attrs']}")
    print(f"Complete trials: {len(complete_only['trials'])}")
    print(json.dumps(complete_only, indent=2))

    pruned_only = jsonify(study, states=[TrialState.PRUNED])
    print(f"\nPruned trials: {len(pruned_only['trials'])}")
