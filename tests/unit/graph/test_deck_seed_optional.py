"""When ``deck_seed`` is omitted from the first invoke, the graph picks one and stores it."""

from langchain_core.messages import AIMessage

from app.graph.parent_graph import compile_blackjack_graph
from app.testing.scripted_model import ScriptedBindableChatModel


def test_deck_seed_generated_when_not_in_initial_input() -> None:
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
    cfg = {"configurable": {"thread_id": "auto-deck-seed"}}
    out = graph.invoke({"player_order": ["human", "agent_a"], "pending_user_text": "stand"}, cfg)
    assert isinstance(out["deck_seed"], int)
    assert 0 <= out["deck_seed"] < (1 << 31)
