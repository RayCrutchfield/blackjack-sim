"""Simulator-wide settings: card rules, seat-agent limits, and default table layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CardConfig:
    """Point-range and per-hand limits for the training table.

    ``max_draws`` limits how many **additional** cards a seat may receive after the
    opening deal (``initial_cards_per_player``). There is no separate “max cards”
    setting: the implied ceiling is ``initial_cards_per_player + max_draws``.
    """

    min_card_point_value: int = 2
    max_card_point_value: int = 11
    initial_cards_per_player: int = 2
    #: Maximum hit cards after the opening deal (simulator safety bound, not a casino rule).
    max_draws: int = 3

    def max_cards_in_hand(self) -> int:
        """Largest number of cards one seat may hold this round (opening + draws)."""
        return self.initial_cards_per_player + self.max_draws

    def allowed_point_values(self) -> frozenset[int]:
        return frozenset(
            range(self.min_card_point_value, self.max_card_point_value + 1)
        )


@dataclass(frozen=True, slots=True)
class SeatAgentLoopConfig:
    """Caps how many times the inner seat graph can go **model → tools → model**.

    Each increment of ``seat_subgraph_steps`` is one model call; this is only a
    failsafe against a stuck tool-calling loop. It is **not** a blackjack rule.
    """

    max_cycles: int = 24


CARD_CONFIG: CardConfig = CardConfig()
SEAT_AGENT_LOOP: SeatAgentLoopConfig = SeatAgentLoopConfig()

# --- Table layout (when ``player_order`` / ``human_player_id`` omitted on first invoke) ---
# human first, then AI seats; last id ``dealer`` is the house bank (same AI subgraph as the others).

DEFAULT_PLAYER_ORDER: tuple[str, ...] = ("human", "agent_a", "agent_b", "dealer")
DEFAULT_HUMAN_PLAYER_ID: str = "human"
