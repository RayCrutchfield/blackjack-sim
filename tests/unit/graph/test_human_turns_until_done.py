"""Human stays active until stand/bust/max cards (casino-style seat order)."""

from langchain_core.messages import AIMessage

from app.cli_session import run_interactive_session
from app.testing.scripted_model import ScriptedBindableChatModel


def _tc(name: str, tid: str) -> dict:
    return {"name": name, "args": {}, "id": tid, "type": "tool_call"}


def test_human_prompted_three_times_two_hits_then_stand_before_ai() -> None:
    """
    Each dealer subgraph run uses two model calls for a hit (draw + no-tool exit).
    Three human NL lines -> six dealer model calls, then two for AI stand-only.
    """
    script = [
        AIMessage("", tool_calls=[_tc("draw_for_human", "d1")]),
        AIMessage("Drew one."),
        AIMessage("", tool_calls=[_tc("draw_for_human", "d2")]),
        AIMessage("Drew two."),
        AIMessage("", tool_calls=[_tc("end_turn_human", "d3")]),
        AIMessage("Standing."),
        AIMessage("", tool_calls=[_tc("end_turn", "a1")]),
        AIMessage("AI out."),
    ]
    model = ScriptedBindableChatModel(script=script)
    lines = iter(["hit", "hit again", "stand"])

    def fake_input(_prompt: str) -> str:
        return next(lines)

    final = run_interactive_session(
        session_id="human-turn-until-done-test",
        deck_seed=1,
        seat_model=model,
        summarization_model=None,
        player_order=["human", "agent_a"],
        input_fn=fake_input,
        print_fn=lambda _: None,
    )
    assert final["phase"] == "game_over"
    assert len(final["hands_by_player"]["human"]) == 4
    assert "agent_a" in final["finished_players"]
