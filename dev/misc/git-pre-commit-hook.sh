#!/usr/bin/env sh
uv run ruff check --quiet
uv run ruff format --check --quiet
uv run dev/tests/test_repository.py
