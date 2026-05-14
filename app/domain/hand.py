"""Player hand: immutable card list and simple queries."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.settings import CARD_CONFIG


@dataclass(frozen=True)
class Hand:
    """Ordered cards for one player. Values are point totals (2–11) per assignment."""

    cards: tuple[int, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return sum(self.cards, start=0)

    @property
    def card_count(self) -> int:
        return len(self.cards)

    def is_bust(self) -> bool:
        return self.total > 21

    def with_card(self, card_value: int) -> Hand:
        return Hand(self.cards + (card_value,))

    def may_take_another_card(self, max_cards_total: int | None = None) -> bool:
        cap = CARD_CONFIG.max_cards_in_hand() if max_cards_total is None else max_cards_total
        if self.card_count >= cap:
            return False
        if self.is_bust():
            return False
        return True
