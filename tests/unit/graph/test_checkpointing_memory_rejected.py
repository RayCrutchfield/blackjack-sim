"""Checkpoint DB path policy."""

import pytest

from app.graph import checkpointing


def test_memory_checkpoint_path_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLACKJACK_CHECKPOINT_DB", ":memory:")
    checkpointing.reset_checkpointer_for_tests()
    with pytest.raises(ValueError, match=":memory"):
        checkpointing.get_checkpointer()
    checkpointing.reset_checkpointer_for_tests()
