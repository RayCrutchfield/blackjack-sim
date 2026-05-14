"""Format table state for the terminal (no deck stream / RNG internals)."""

from __future__ import annotations

from app.settings import CARD_CONFIG

# Last AI seat was renamed from ``agent_c`` to ``dealer``; checkpoints may still
# store the old id. The board shows the current name while state keys stay as-is.
_LEGACY_LAST_SEAT = "agent_c"
_DISPLAY_LAST_SEAT = "dealer"


def seat_display_label(player_id: str) -> str:
    return _DISPLAY_LAST_SEAT if player_id == _LEGACY_LAST_SEAT else player_id


def _finished_suffix(
    player_id: str,
    *,
    values: dict,
    cards: list[int],
    bust: bool,
) -> str:
    """Explain why a seat no longer acts (stand vs max draws vs bust)."""
    finished = set(values.get("finished_players") or [])
    if player_id not in finished:
        return ""
    if bust:
        return " (bust)"
    stood = set(values.get("stood_players") or [])
    cap = CARD_CONFIG.max_cards_in_hand()
    if player_id in stood:
        return " (stood)"
    if len(cards) >= cap:
        return " (max draws)"
    return " (out)"


def format_table(values: dict) -> str:
    """
    Human-readable board: each player's visible hand and total.

    The simplified assignment still shows card values as dealt by agents; the user
    never sees the underlying ``deck_stream`` or ``next_draw_index``.
    """
    order = values.get("player_order") or []
    hands = values.get("hands_by_player") or {}
    active = values.get("active_seat")
    phase = values.get("phase") or "playing"
    show_turn_arrow = phase != "game_over"
    lines: list[str] = []
    for player_id in order:
        cards = hands.get(player_id, [])
        total = sum(cards)
        bust = total > 21
        total_label = "BUST" if bust else str(total)
        marker = "→" if show_turn_arrow and player_id == active else " "
        suffix = _finished_suffix(player_id, values=values, cards=cards, bust=bust)
        cards_display = ", ".join(str(c) for c in cards) if cards else "—"
        label = seat_display_label(player_id)
        lines.append(f"{marker} {label:8}  cards: [{cards_display}]  total: {total_label}{suffix}")
    winners = values.get("winners")
    if winners is not None and len(winners) > 0:
        labels = ", ".join(seat_display_label(w) for w in winners)
        if len(winners) == 1:
            lines.append(f"Winner: {labels}")
        else:
            lines.append(f"Winners: {labels}")
    elif values.get("phase") == "game_over":
        lines.append("No winner — all bust")
    return "\n".join(lines)
