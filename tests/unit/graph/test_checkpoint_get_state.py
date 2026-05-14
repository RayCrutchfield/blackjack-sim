"""Checkpoint visibility on the parent graph (SQLite checkpointer)."""

from langchain_core.messages import AIMessage

from app.graph.parent_graph import compile_blackjack_graph
from app.testing.scripted_model import ScriptedBindableChatModel


def test_get_state_after_invoke() -> None:
    model = ScriptedBindableChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "end_turn_human", "args": {}, "id": "h1", "type": "tool_call"},
                ],
            ),
            AIMessage(content="ok"),
            AIMessage(
                content="",
                tool_calls=[{"name": "end_turn", "args": {}, "id": "a1", "type": "tool_call"}],
            ),
            AIMessage(content="bye"),
        ]
    )
    graph = compile_blackjack_graph(model)
    cfg = {"configurable": {"thread_id": "cp-test"}}
    graph.invoke(
        {
            "pending_user_text": "stand",
            "player_order": ["human", "agent_a"],
            "deck_seed": 1,
        },
        config=cfg,
    )
    snap = graph.get_state(cfg)
    assert snap.values.get("phase") == "game_over"
