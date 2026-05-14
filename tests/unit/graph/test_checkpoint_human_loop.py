"""Checkpoint + interrupt_before dealer + Command(update) resume."""

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.graph.parent_graph import compile_blackjack_graph
from app.testing.scripted_model import ScriptedBindableChatModel


def test_resume_human_turn_updates_pending_and_finishes() -> None:
    model = ScriptedBindableChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "end_turn_human", "args": {}, "id": "x", "type": "tool_call"}],
            ),
            AIMessage(content="done"),
            AIMessage(
                content="",
                tool_calls=[{"name": "end_turn", "args": {}, "id": "y", "type": "tool_call"}],
            ),
            AIMessage(content="ai done"),
        ]
    )
    graph = compile_blackjack_graph(model, interrupt_before_dealer=True)
    cfg = {"configurable": {"thread_id": "ckpt-human"}}
    graph.invoke({"player_order": ["human", "agent_a"], "deck_seed": 2}, cfg)
    snap = graph.get_state(cfg)
    assert snap.next[0] == "dealer"
    graph.invoke(Command(update={"pending_user_text": "stand"}), cfg)
    final = graph.get_state(cfg)
    assert not final.next
    assert final.values["phase"] == "game_over"
