"""Unit tests for deterministic turn helpers."""

from app.graph.turns import (
    advance_after_seat,
    all_players_finished,
    auto_finish_if_ineligible,
    next_unfinished_seat,
)


def test_all_players_finished() -> None:
    assert all_players_finished(player_order=["a", "b"], finished_players=["a", "b"]) is True
    assert all_players_finished(player_order=["a", "b"], finished_players=["a"]) is False


def test_auto_finish_bust() -> None:
    hands = {"p": [10, 10, 5]}
    out = auto_finish_if_ineligible(hands, "p", [])
    assert "p" in out


def test_next_unfinished_seat_wraps() -> None:
    seat = next_unfinished_seat(
        player_order=["a", "b", "c"],
        finished_players=["a"],
        after_seat="a",
    )
    assert seat == "b"


def test_advance_sets_winners_when_all_done() -> None:
    hands = {"human": [10, 9], "agent_a": [5, 5, 5]}
    update = advance_after_seat(
        hands_by_player=hands,
        player_order=["human", "agent_a"],
        finished_players=["human", "agent_a"],
        active_seat="agent_a",
    )
    assert update["phase"] == "game_over"
    assert update["winners"] == ["human"]
