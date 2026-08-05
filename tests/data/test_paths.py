from __future__ import annotations

from pathlib import Path

import pytest

from uztts_data.paths import DATA_ROOT_ENV, data_root, manifests_root, raw_root


def test_defaults_to_repo_local_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    assert data_root() == Path("data")
    assert raw_root() == Path("data/raw")
    assert manifests_root() == Path("data/manifests")


def test_env_overrides_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "/srv/uztts-data")
    assert raw_root() == Path("/srv/uztts-data/raw")


def test_tilde_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "~/uztts-data")
    assert data_root() == Path.home() / "uztts-data"


def test_empty_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "")
    assert data_root() == Path("data")
