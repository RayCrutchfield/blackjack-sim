"""Shared logic for dealer/AI seat tool nodes (draw from stream, stand, unknown tool)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage

from app.cli_board import seat_display_label
from app.domain.actions import apply_draw
from app.domain.hand import Hand
from app.graph.state import hands_from_domain


def _article_for_card_value(value: int) -> str:
    return "an" if value in (8, 11) else "a"


def _total_suffix(hand: Hand) -> str:
    return f" (total: {hand.total})"


def _append_draw_narration(
    round_log: list[str],
    *,
    target_player_id: str,
    human_player_id: str,
    card_value: int,
    hand: Hand,
) -> None:
    if target_player_id == human_player_id:
        art = _article_for_card_value(card_value)
        round_log.append(f"You drew {art} {card_value}.{_total_suffix(hand)}")
        if hand.is_bust():
            round_log.append(f"Bust.{_total_suffix(hand)}")
        else:
            round_log.append("What's your next move?")
    else:
        label = seat_display_label(target_player_id)
        round_log.append(f"{label} was dealt {card_value}.{_total_suffix(hand)}")
        if hand.is_bust():
            round_log.append(f"{label} busts.{_total_suffix(hand)}")


def _append_stand_narration(
    round_log: list[str],
    target_player_id: str,
    human_player_id: str,
) -> None:
    if target_player_id == human_player_id:
        round_log.append("You chose to stand.")
    else:
        round_log.append(f"{seat_display_label(target_player_id)} chose to stand.")


@dataclass(frozen=True, slots=True)
class SeatToolBindings:
    """Which OpenAI tool names map to draw vs stand for this seat."""

    draw_name: str
    stand_name: str
    unknown_actor: str


@dataclass(slots=True)
class SeatToolRunState:
    """Mutable accumulator while processing one batched ``tool_calls`` list."""

    domain: dict[str, Hand]
    hands_map: dict[str, list[int]]
    draw_index: int
    finished: list[str]
    #: Seats that ended their turn via a stand tool (vs auto-finish at max cards / bust).
    stood: list[str]
    #: Lines appended on successful draw / stand (deterministic table narration).
    round_log: list[str]
    tool_messages: list[ToolMessage] = field(default_factory=list)


def accumulate_seat_tool_effects(
    tool_calls: Sequence[dict[str, Any]],
    *,
    bindings: SeatToolBindings,
    target_player_id: str,
    human_player_id: str,
    stream: list[int],
    state: SeatToolRunState,
    max_cards_total: int,
    draw_ack: Callable[[int], str],
    stand_ack: str,
) -> None:
    """
    Append :class:`ToolMessage` responses and update ``state`` for each call.

    ``state.domain`` / ``hands_map`` / ``draw_index`` / ``finished`` are updated
    in place so the parent tool node can return a single merged update dict.
    """
    for call in tool_calls:
        name = call["name"]
        call_id = call["id"]
        if name == bindings.draw_name:
            if state.draw_index >= len(stream):
                state.tool_messages.append(
                    ToolMessage(
                        content="Deck exhausted (internal error).",
                        tool_call_id=call_id,
                    )
                )
                continue
            value = stream[state.draw_index]
            state.draw_index += 1
            try:
                state.domain = dict(
                    apply_draw(
                        state.domain,
                        target_player_id,
                        value,
                        max_cards_total=max_cards_total,
                    )
                )
                state.hands_map = hands_from_domain(state.domain)
                state.tool_messages.append(
                    ToolMessage(content=draw_ack(value), tool_call_id=call_id)
                )
                _append_draw_narration(
                    state.round_log,
                    target_player_id=target_player_id,
                    human_player_id=human_player_id,
                    card_value=value,
                    hand=state.domain[target_player_id],
                )
            except Exception as exc:  # noqa: BLE001 — surface to model
                state.tool_messages.append(
                    ToolMessage(content=f"Cannot draw: {exc}", tool_call_id=call_id)
                )
        elif name == bindings.stand_name:
            hand = state.domain.get(target_player_id)
            bust = hand is not None and hand.is_bust()
            if target_player_id not in state.finished:
                state.finished.append(target_player_id)
            if not bust:
                if target_player_id not in state.stood:
                    state.stood.append(target_player_id)
                _append_stand_narration(state.round_log, target_player_id, human_player_id)
            state.tool_messages.append(ToolMessage(content=stand_ack, tool_call_id=call_id))
        else:
            state.tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool {name!r} for {bindings.unknown_actor}.",
                    tool_call_id=call_id,
                )
            )
