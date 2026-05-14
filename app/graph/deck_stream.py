"""Deterministic deck stream for checkpoint-friendly state."""

from __future__ import annotations

import random

from app.settings import CARD_CONFIG

# One round needs at most ``players × max_cards_in_hand()`` draws (plus margin).
# We precompute a fixed slice so ``deck_stream`` is serializable in checkpoints;
# 128 is a loose upper bound, not a blackjack rule.
DEFAULT_DECK_STREAM_LENGTH = 128


def materialize_deck_stream(*, seed: int, length: int = DEFAULT_DECK_STREAM_LENGTH) -> list[int]:
    """Return ``length`` card point values using :data:`app.settings.CARD_CONFIG` bounds.

    ``length`` defaults to :data:`DEFAULT_DECK_STREAM_LENGTH` — large enough that
    ``next_draw_index`` never reaches the end during a normal single round for this
    simulator (see module comment).
    """
    generator = random.Random(seed)
    lo, hi = CARD_CONFIG.min_card_point_value, CARD_CONFIG.max_card_point_value
    return [generator.randint(lo, hi) for _ in range(length)]
