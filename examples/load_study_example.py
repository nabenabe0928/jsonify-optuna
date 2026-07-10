from __future__ import annotations

import json
from pathlib import Path
import tempfile

import optuna

from jsonify_optuna import jsonify
from jsonify_optuna import load_study


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    return x**2


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "example.db"
        rdb_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            storage=rdb_url, study_name="rdb_demo", direction="minimize"
        )
        study.optimize(objective, n_trials=10)

        loaded = load_study(rdb_url=rdb_url, study_name="rdb_demo")
        print("=== RDB storage ===")
        print(json.dumps(jsonify(loaded), indent=2))

        journal_path = str(Path(tmpdir) / "journal.log")
        backend = optuna.storages.journal.JournalFileBackend(journal_path)
        storage = optuna.storages.JournalStorage(backend)
        study = optuna.create_study(
            storage=storage, study_name="journal_demo", direction="minimize"
        )
        study.optimize(objective, n_trials=10)

        loaded = load_study(
            journal_path=journal_path, study_name="journal_demo"
        )
        print("\n=== Journal storage ===")
        print(json.dumps(jsonify(loaded), indent=2))
