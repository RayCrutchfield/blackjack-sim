"""Deterministic seat rotation and auto-finish rules (no LLM)."""

from __future__ import annotations

from app.domain.hand import Hand
from app.domain.scoring import winners_for_round
from app.graph.state import hands_to_domain
from app.settings import CARD_CONFIG, DEFAULT_PLAYER_ORDER


def all_players_finished(*, player_order: list[str], finished_players: list[str]) -> bool:
    finished_set = set(finished_players)
    return all(player_id in finished_set for player_id in player_order)


def auto_finish_if_ineligible(
    hands_by_player: dict[str, list[int]],
    seat: str,
    finished_players: list[str],
) -> list[str]:
    """Append ``seat`` to finished if bust or cannot draw another card."""
    domain = hands_to_domain(hands_by_player)
    hand = domain[seat]
    # Deduplicate while preserving order (dict keys are unique; Py3.7+ insertion order).
    merged = list(dict.fromkeys(finished_players))
    if hand.is_bust() or not hand.may_take_another_card(CARD_CONFIG.max_cards_in_hand()):
        if seat not in merged:
            merged.append(seat)
    return merged


def next_unfinished_seat(
    *,
    player_order: list[str],
    finished_players: list[str],
    after_seat: str,
) -> str | None:
    """Next player in cyclic order after ``after_seat`` who is not in ``finished_players``."""
    finished_set = set(finished_players)
    if after_seat not in player_order:
        return None
    start_index = player_order.index(after_seat)
    for step in range(1, len(player_order) + 1):
        candidate = player_order[(start_index + step) % len(player_order)]
        if candidate not in finished_set:
            return candidate
    return None


def advance_after_seat(
    *,
    hands_by_player: dict[str, list[int]],
    player_order: list[str],
    finished_players: list[str],
    active_seat: str,
) -> dict:
    """
    After a seat subgraph ends, apply bust/max-card auto-finish.

    If ``active_seat`` is still not finished (e.g. human did not stand and is not
    bust), keep ``active_seat`` unchanged so the same seat runs again (new human
    input on the next parent step). Otherwise move to the next unfinished seat or
    end the round.
    """
    finished_after = auto_finish_if_ineligible(hands_by_player, active_seat, finished_players)
    outgoing: dict = {"finished_players": finished_after, "seat_subgraph_steps": 0}

    if active_seat not in finished_after:
        outgoing["active_seat"] = active_seat
        return outgoing

    if all_players_finished(player_order=player_order, finished_players=finished_after):
        winners = winners_for_round(hands_to_domain(hands_by_player))
        outgoing["winners"] = winners
        outgoing["phase"] = "game_over"
        outgoing["pending_user_text"] = ""
        return outgoing

    next_seat = next_unfinished_seat(
        player_order=player_order,
        finished_players=finished_after,
        after_seat=active_seat,
    )
    outgoing["active_seat"] = next_seat if next_seat else player_order[0]
    outgoing["pending_user_text"] = ""
    return outgoing


def initial_player_order(explicit: list[str] | None) -> list[str]:
    return list(explicit) if explicit else list(DEFAULT_PLAYER_ORDER)
