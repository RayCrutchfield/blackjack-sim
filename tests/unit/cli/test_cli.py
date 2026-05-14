"""CLI entry, board formatting, and scripted interactive session."""

from __future__ import annotations

import sys

import pytest
from langchain_core.messages import AIMessage

import app.cli as cli
from app.cli_board import format_table
from app.cli_session import run_interactive_session
from app.testing.scripted_model import ScriptedBindableChatModel


# --- app.cli ---


def test_cli_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["blackjack-sim", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0


# --- format_table ---


def test_format_table_shows_hands_and_hides_deck_stream() -> None:
    text = format_table(
        {
            "player_order": ["human", "agent_a"],
            "hands_by_player": {"human": [10, 7], "agent_a": [9, 8]},
            "active_seat": "human",
            "finished_players": [],
            "phase": "playing",
        }
    )
    assert "human" in text
    assert "17" in text or "BUST" not in text
    assert "deck_stream" not in text.lower()
    assert "→" in text


def test_format_table_shows_dealer_label_for_legacy_agent_c_seat() -> None:
    text = format_table(
        {
            "player_order": ["human", "agent_a", "agent_b", "agent_c"],
            "hands_by_player": {"human": [10], "agent_a": [5], "agent_b": [7], "agent_c": [9, 4]},
            "active_seat": "agent_c",
            "finished_players": [],
            "phase": "playing",
        }
    )
    assert "agent_c" not in text
    assert "dealer" in text
    assert "→" in text


def test_format_table_no_turn_arrow_when_round_over() -> None:
    text = format_table(
        {
            "player_order": ["human", "agent_a"],
            "hands_by_player": {"human": [10, 11], "agent_a": [9, 8]},
            "active_seat": "human",
            "finished_players": ["human", "agent_a"],
            "stood_players": ["human", "agent_a"],
            "phase": "game_over",
            "winners": ["agent_a"],
        }
    )
    assert "→" not in text


def test_format_table_max_draws_without_stand() -> None:
    text = format_table(
        {
            "player_order": ["human"],
            "hands_by_player": {"human": [2, 2, 2, 2, 2]},
            "active_seat": "human",
            "finished_players": ["human"],
            "stood_players": [],
            "phase": "playing",
        }
    )
    assert "(max draws)" in text
    assert "(stood)" not in text


def test_format_table_stood_suffix() -> None:
    text = format_table(
        {
            "player_order": ["human", "agent_a"],
            "hands_by_player": {"human": [10, 8], "agent_a": [9, 7]},
            "active_seat": "agent_a",
            "finished_players": ["human"],
            "stood_players": ["human"],
            "phase": "playing",
        }
    )
    assert "(stood)" in text


# --- run_interactive_session ---


def test_interactive_session_two_player_scripted() -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "end_turn_human", "args": {}, "id": "h1", "type": "tool_call"}],
        ),
        AIMessage(content="ok"),
        AIMessage(
            content="",
            tool_calls=[{"name": "end_turn", "args": {}, "id": "a1", "type": "tool_call"}],
        ),
        AIMessage(content="bye"),
    ]
    model = ScriptedBindableChatModel(script=script)
    lines: list[str] = []

    def fake_input(_prompt: str) -> str:
        return "I stand."

    final = run_interactive_session(
        session_id="cli-script-test",
        deck_seed=9,
        seat_model=model,
        summarization_model=None,
        player_order=["human", "agent_a"],
        input_fn=fake_input,
        print_fn=lines.append,
    )
    assert final["phase"] == "game_over"
    board_text = "\n".join(lines)
    assert "Start human" in board_text
    assert "human" in board_text
    assert "deck_stream" not in board_text.lower()
