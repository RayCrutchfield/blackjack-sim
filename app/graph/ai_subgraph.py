"""AI seat subgraph: tool-calling loop for ``draw_for_self`` / ``end_turn``."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START

from app.graph.seat_tool_dispatch import (
    SeatToolBindings,
    SeatToolRunState,
    accumulate_seat_tool_effects,
)
from app.graph.state import (
    BlackjackState,
    hands_to_domain,
    require_seat_tool_loop_state,
)
from app.settings import CARD_CONFIG, SEAT_AGENT_LOOP


@tool
def draw_for_self() -> str:
    """Draw one card for yourself (the active AI seat)."""
    return "ack"


@tool
def end_turn() -> str:
    """Stand — you take no more cards this turn."""
    return "ack"


def ai_seat_system_content(state: BlackjackState) -> str:
    s = require_seat_tool_loop_state(state)
    seat = s["active_seat"]
    cards = s["hands_by_player"].get(seat, [])
    total = sum(cards)
    total_cap = CARD_CONFIG.max_cards_in_hand()
    max_d = CARD_CONFIG.max_draws
    opener = CARD_CONFIG.initial_cards_per_player
    opener_note = (
        f"The table already dealt {opener} opening card(s) per player from the shoe. "
        if opener > 0
        else ""
    )
    return (
        f"You are blackjack player {seat!r}. "
        f"Your cards are {cards} (total {total}). "
        f"{opener_note}"
        f"You may take at most {max_d} hit(s) after the opening (at most {total_cap} cards total) "
        "and bust if total > 21. "
        "You are playing to win the round: non-busted hands with the highest total win (ties share). "
        "Try to finish with the strongest total you reasonably can. "
        "Call draw_for_self to hit, or end_turn when you want to stand."
    )


def _should_continue(state: BlackjackState) -> Literal["tools", "done"]:
    s = require_seat_tool_loop_state(state)
    if s["active_seat"] in s["finished_players"]:
        return "done"
    if int(s.get("seat_subgraph_steps", 0)) > SEAT_AGENT_LOOP.max_cycles:
        return "done"
    last = s["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "done"


def _ai_agent(model: BaseChatModel):
    bound = model.bind_tools([draw_for_self, end_turn])

    def _run(state: BlackjackState) -> dict:
        s = require_seat_tool_loop_state(state)
        step = int(s.get("seat_subgraph_steps", 0)) + 1
        response = bound.invoke(s["messages"])
        out: dict = {"messages": [response], "seat_subgraph_steps": step}
        seat = s["active_seat"]
        if isinstance(response, AIMessage):
            raw = response.content
            text = (raw if isinstance(raw, str) else str(raw or "")).strip()
            if text:
                prev = list(s.get("chatter_events") or [])
                out["chatter_events"] = prev + [{"seat": seat, "text": text[:400]}]
        return out

    return _run


def _ai_tools(state: BlackjackState) -> dict:
    s = require_seat_tool_loop_state(state)
    last = s["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}
    seat = s["active_seat"]
    hands_map = {player_id: list(cards) for player_id, cards in s["hands_by_player"].items()}
    acc = SeatToolRunState(
        domain=hands_to_domain(hands_map),
        hands_map=hands_map,
        draw_index=int(s["next_draw_index"]),
        finished=list(dict.fromkeys(s["finished_players"])),
        stood=list(dict.fromkeys(s.get("stood_players") or [])),
        round_log=list(s.get("round_log") or []),
    )
    accumulate_seat_tool_effects(
        last.tool_calls,
        bindings=SeatToolBindings(
            draw_name="draw_for_self",
            stand_name="end_turn",
            unknown_actor="AI seat",
        ),
        target_player_id=seat,
        human_player_id=s["human_player_id"],
        stream=s["deck_stream"],
        state=acc,
        max_cards_total=CARD_CONFIG.max_cards_in_hand(),
        draw_ack=lambda v: f"Drew {v}.",
        stand_ack=f"{seat} stands.",
    )
    return {
        "messages": acc.tool_messages,
        "hands_by_player": acc.hands_map,
        "next_draw_index": acc.draw_index,
        "finished_players": acc.finished,
        "stood_players": acc.stood,
        "round_log": acc.round_log,
    }


def build_ai_subgraph(model: BaseChatModel) -> StateGraph:
    graph = StateGraph(BlackjackState)
    graph.add_node("ai_agent", _ai_agent(model))
    graph.add_node("ai_tools", _ai_tools)
    graph.add_edge(START, "ai_agent")
    graph.add_conditional_edges(
        "ai_agent",
        _should_continue,
        {"tools": "ai_tools", "done": END},
    )
    graph.add_edge("ai_tools", "ai_agent")
    return graph
