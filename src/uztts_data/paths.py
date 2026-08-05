from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "UZTTS_DATA_ROOT"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV) or "data").expanduser()


def raw_root() -> Path:
    return data_root() / "raw"


def manifests_root() -> Path:
    return data_root() / "manifests"
