.PHONY: setup fmt lint typecheck test check tg-pull report segment

setup:
	uv sync

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

tg-pull:
	set -a; . ./.env; set +a; uv run uztts-data tg pull

report:
	set -a; . ./.env; set +a; uv run uztts-data report

segment:
	set -a; . ./.env; set +a; uv run uztts-data scan-raw && uv run uztts-segment
