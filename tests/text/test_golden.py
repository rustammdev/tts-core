import json
from pathlib import Path

import pytest

from uztts_text import normalize

GOLDEN = Path(__file__).parent / "golden.jsonl"
CASES = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("case", CASES, ids=[case["in"][:40] for case in CASES])
def test_golden(case: dict[str, str]) -> None:
    assert normalize(case["in"]) == case["out"]
