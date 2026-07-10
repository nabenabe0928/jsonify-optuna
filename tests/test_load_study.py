from __future__ import annotations

from typing import TYPE_CHECKING

import optuna
import pytest

from jsonify_optuna import load_study


if TYPE_CHECKING:
    from pathlib import Path


def _create_study_in_rdb(tmp_path: Path, study_name: str) -> str:
    url = f"sqlite:///{tmp_path / 'test.db'}"
    study = optuna.create_study(storage=url, study_name=study_name)
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
    return url


def test_load_with_rdb_url(tmp_path: Path) -> None:
    url = _create_study_in_rdb(tmp_path, "rdb_study")
    study = load_study(rdb_url=url, study_name="rdb_study")
    assert study.study_name == "rdb_study"
    assert len(study.trials) == 1


def test_load_with_journal_path(tmp_path: Path) -> None:
    journal_path = str(tmp_path / "journal.log")
    backend = optuna.storages.journal.JournalFileBackend(journal_path)
    storage = optuna.storages.JournalStorage(backend)
    study = optuna.create_study(storage=storage, study_name="journal_study")
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    loaded = load_study(journal_path=journal_path, study_name="journal_study")
    assert loaded.study_name == "journal_study"
    assert len(loaded.trials) == 1


def test_load_with_storage(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'test.db'}"
    storage = optuna.storages.RDBStorage(url)
    study = optuna.create_study(storage=storage, study_name="direct_study")
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    loaded = load_study(storage=storage, study_name="direct_study")
    assert loaded.study_name == "direct_study"


@pytest.mark.parametrize(
    "kwargs_key",
    ["storage_and_rdb_url", "storage_and_journal_path", "rdb_url_and_journal_path", "no_storage"],
)
def test_conflicting_args(kwargs_key: str) -> None:
    dummy_storage = optuna.storages.RDBStorage("sqlite:///:memory:")
    cases = {
        "storage_and_rdb_url": dict(storage=dummy_storage, rdb_url="sqlite:///x.db"),
        "storage_and_journal_path": dict(storage=dummy_storage, journal_path="/tmp/j.log"),
        "rdb_url_and_journal_path": dict(rdb_url="sqlite:///x.db", journal_path="/tmp/j.log"),
        "no_storage": {},
    }
    with pytest.raises(ValueError):
        load_study(**cases[kwargs_key])


def test_study_name_not_found(tmp_path: Path) -> None:
    url = _create_study_in_rdb(tmp_path, "existing")
    with pytest.raises(ValueError):
        load_study(rdb_url=url, study_name="nonexistent")


def test_multiple_studies_no_name(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'test.db'}"
    optuna.create_study(storage=url, study_name="study_a")
    optuna.create_study(storage=url, study_name="study_b")
    with pytest.raises(ValueError):
        load_study(rdb_url=url)


def test_auto_select_single_study(tmp_path: Path) -> None:
    url = _create_study_in_rdb(tmp_path, "only_one")
    study = load_study(rdb_url=url)
    assert study.study_name == "only_one"
