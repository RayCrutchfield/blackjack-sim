"""Parent LangGraph with scripted models (no network)."""

from langchain_core.messages import AIMessage

from app.graph.parent_graph import compile_blackjack_graph
from app.testing.scripted_model import ScriptedBindableChatModel


def test_full_round_two_players_scripted() -> None:
    model = ScriptedBindableChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "end_turn_human", "args": {}, "id": "h1", "type": "tool_call"},
                ],
            ),
            AIMessage(content="Ack."),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "draw_for_self", "args": {}, "id": "a1", "type": "tool_call"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "end_turn", "args": {}, "id": "a2", "type": "tool_call"},
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    graph = compile_blackjack_graph(model, summarization_model=None)
    result = graph.invoke(
        {
            "pending_user_text": "I will stand",
            "player_order": ["human", "agent_a"],
            "deck_seed": 3,
        },
        config={"configurable": {"thread_id": "two-player-script"}},
    )
    assert result["phase"] == "game_over"
    assert result["hands_by_player"]["human"] == [5, 10]
    assert result["hands_by_player"]["agent_a"] == [11, 4, 7]
    assert result["winners"] == ["human"]
