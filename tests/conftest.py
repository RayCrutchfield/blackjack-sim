"""Pytest defaults: isolate LangGraph SQLite checkpoints per test process."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_langgraph_sqlite(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Fresh DB file so fixed ``thread_id`` values in tests never load stale checkpoints."""
    db = tmp_path_factory.mktemp("langgraph_checkpoints") / "test.sqlite"
    os.environ["BLACKJACK_CHECKPOINT_DB"] = str(db)
