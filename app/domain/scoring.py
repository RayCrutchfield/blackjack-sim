"""Winner resolution and empty table helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.domain.hand import Hand


def empty_hands_for_players(player_ids: Iterable[str]) -> dict[str, Hand]:
    """Every player starts with an empty hand."""
    return {player_id: Hand() for player_id in player_ids}


def winners_for_round(hands_by_player: Mapping[str, Hand]) -> list[str]:
    """
    Non-busted players with the highest total win together.

    Returns player ids sorted ascending for stable, deterministic ordering.
    If every player is busted, returns an empty list.
    """
    eligible_scores: list[tuple[str, int]] = [
        (player_id, hand.total)
        for player_id, hand in hands_by_player.items()
        if not hand.is_bust()
    ]
    if not eligible_scores:
        return []
    best_total = max(score for _, score in eligible_scores)
    winning_ids = [player_id for player_id, score in eligible_scores if score == best_total]
    return sorted(winning_ids)
