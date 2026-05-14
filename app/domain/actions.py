"""Pure transitions on hand collections."""

from __future__ import annotations

from collections.abc import Mapping

from app.domain.errors import IllegalDrawError
from app.domain.hand import Hand
from app.settings import CARD_CONFIG


def apply_draw(
    hands_by_player: Mapping[str, Hand],
    player_id: str,
    card_value: int,
    *,
    max_cards_total: int | None = None,
) -> Mapping[str, Hand]:
    """
    Return a new hands map with one card appended for ``player_id``.

    Does not mutate ``hands_by_player``. Raises ``IllegalDrawError`` if the
    draw is not allowed (unknown player, invalid card value, hand already at the
    maximum size, or already bust).

    Maximum hand size defaults to :meth:`app.settings.CARD_CONFIG.max_cards_in_hand`
    (``initial_cards_per_player`` + ``max_draws``). Pass ``max_cards_total`` only
    for unit tests.

    The argument type is :class:`~collections.abc.Mapping` (read-only contract);
    the return value is a fresh ``dict`` that also satisfies ``Mapping`` so
    callers can treat the result as an immutable snapshot or copy it again.
    """
    cap = CARD_CONFIG.max_cards_in_hand() if max_cards_total is None else max_cards_total
    allowed = CARD_CONFIG.allowed_point_values()
    if card_value not in allowed:
        raise IllegalDrawError(
            f"card_value must be one of {sorted(allowed)}, got {card_value!r}."
        )
    if player_id not in hands_by_player:
        raise IllegalDrawError(f"Unknown player_id: {player_id!r}")
    current_hand = hands_by_player[player_id]
    if not current_hand.may_take_another_card(max_cards_total=cap):
        raise IllegalDrawError(
            f"Player {player_id!r} cannot draw (cards={current_hand.cards!r}, "
            f"bust={current_hand.is_bust()})."
        )
    updated = dict(hands_by_player)
    updated[player_id] = current_hand.with_card(card_value)
    return updated
