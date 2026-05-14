"""Random card draw in the simplified range 2–11 (inclusive)."""

from __future__ import annotations

import random

from app.settings import CARD_CONFIG


class CardDeck:
    """Draws one card at a time using an injectable random source (tests use a fixed seed)."""

    def __init__(self, random_generator: random.Random | None = None) -> None:
        self._random = random_generator if random_generator is not None else random.Random()

    def draw(self) -> int:
        # randint is inclusive on both ends, matching ``CardConfig`` point range.
        return self._random.randint(
            CARD_CONFIG.min_card_point_value, CARD_CONFIG.max_card_point_value
        )
