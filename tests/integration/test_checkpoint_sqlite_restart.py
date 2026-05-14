"""SQLite checkpointer: survive simulated crash (connection close + recompile + reopen DB)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.graph.checkpointing import get_checkpointer, reset_checkpointer_for_tests
from app.graph.parent_graph import compile_blackjack_graph
from app.testing.scripted_model import ScriptedBindableChatModel

pytestmark = pytest.mark.integration


def test_sqlite_checkpoint_survives_simulated_process_restart(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Mimic: process A runs the graph, hits interrupt_before dealer, then dies.

    Process B opens the same SQLite file, recompiles the graph, and must see the
    same pending node plus durable state, then can resume with ``Command``.
    """
    db = tmp_path / "restart.sqlite"
    monkeypatch.setenv("BLACKJACK_CHECKPOINT_DB", str(db))
    reset_checkpointer_for_tests()
    get_checkpointer()

    model = ScriptedBindableChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "end_turn_human", "args": {}, "id": "x", "type": "tool_call"},
                ],
            ),
            AIMessage(content="dealer done"),
            AIMessage(
                content="",
                tool_calls=[{"name": "end_turn", "args": {}, "id": "y", "type": "tool_call"}],
            ),
            AIMessage(content="ai done"),
        ]
    )
    cfg = {"configurable": {"thread_id": "sqlite-restart-demo"}}
    initial = {"player_order": ["human", "agent_a"], "deck_seed": 2}

    graph_before = compile_blackjack_graph(model, interrupt_before_dealer=True)
    graph_before.invoke(initial, cfg)
    snap_before = graph_before.get_state(cfg)
    assert snap_before.next and snap_before.next[0] == "dealer"
    assert snap_before.values.get("phase") == "playing"

    # Simulated crash: lose in-process connection and LangGraph objects.
    reset_checkpointer_for_tests()
    get_checkpointer()

    graph_after = compile_blackjack_graph(model, interrupt_before_dealer=True)
    snap_after = graph_after.get_state(cfg)
    assert snap_after.next and snap_after.next[0] == "dealer"
    assert snap_after.values.get("phase") == "playing"
    assert snap_after.values.get("player_order") == ["human", "agent_a"]

    graph_after.invoke(Command(update={"pending_user_text": "stand"}), cfg)
    final = graph_after.get_state(cfg)
    assert not final.next
    assert final.values.get("phase") == "game_over"

    reset_checkpointer_for_tests()
