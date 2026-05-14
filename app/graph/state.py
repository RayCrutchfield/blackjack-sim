"""Shared LangGraph state for the blackjack table (parent + seat subgraphs)."""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, cast

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class BlackjackState(TypedDict, total=False):
    """
    Table state threaded through the parent ``StateGraph`` and seat subgraphs.

    Uses ``messages`` with ``add_messages`` plus domain fields updated from tool nodes.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    hands_by_player: dict[str, list[int]]
    finished_players: list[str]
    player_order: list[str]
    human_player_id: str
    active_seat: str
    deck_stream: list[int]
    next_draw_index: int
    deck_seed: int
    winners: list[str]
    phase: Literal["playing", "game_over"]
    # Human NL for the dealer subgraph (set by CLI ``Command`` before ``dealer`` runs).
    pending_user_text: str
    # Incremented each seat agent step; caps inner tool loops (see ``SEAT_AGENT_LOOP``).
    seat_subgraph_steps: int
    # Seats that ended via stand tools (vs bust / max cards).
    stood_players: list[str]
    # LLM plain-text lines for optional CLI display; each item has ``seat`` and ``text``.
    chatter_events: list[dict[str, str]]
    # Deterministic narration from tool effects (draws / stands / busts).
    round_log: list[str]


class SeatToolLoopState(TypedDict):
    """Keys present whenever a seat subgraph runs (after ``init_round``)."""

    messages: Annotated[list[AnyMessage], add_messages]
    hands_by_player: dict[str, list[int]]
    finished_players: list[str]
    active_seat: str
    next_draw_index: int
    deck_stream: list[int]
    human_player_id: str
    seat_subgraph_steps: NotRequired[int]


_SEAT_TOOL_KEYS: frozenset[str] = frozenset({
    "messages",
    "hands_by_player",
    "finished_players",
    "active_seat",
    "next_draw_index",
    "deck_stream",
    "human_player_id",
})


def require_seat_tool_loop_state(state: BlackjackState) -> SeatToolLoopState:
    """Narrow ``BlackjackState`` for seat subgraph code (satisfies type checkers)."""
    missing = [key for key in _SEAT_TOOL_KEYS if key not in state]
    if missing:
        raise KeyError(
            f"Seat subgraph expected keys {sorted(_SEAT_TOOL_KEYS)}; missing {missing}"
        )
    return cast(SeatToolLoopState, state)


def hands_to_domain(hands: dict[str, list[int]]) -> dict:
    from app.domain.hand import Hand

    return {player_id: Hand(tuple(cards)) for player_id, cards in hands.items()}


def hands_from_domain(hands_domain: dict) -> dict[str, list[int]]:
    return {player_id: list(hand.cards) for player_id, hand in hands_domain.items()}
