"""Dealer subgraph: natural language to ``draw_for_human`` / ``end_turn_human`` tools."""

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

DEALER_SYSTEM = """You are the blackjack table dealer for the HUMAN player only.
The human cannot touch the deck. They already have any opening cards shown on the table; you only apply further hits or stand via tools.
Table play is round-robin: after you resolve their message, other seats act until it is the human's turn again (they may hit again on a later pass if they have not stood or busted).
They speak in natural language.
You must choose exactly one tool per user message when intent is clear:
- If they want another card (hit, deal me one, another, etc.): call draw_for_human.
- If they want to stop (stand, I'm good, no more, done, etc.): call end_turn_human.
If their intent is ambiguous, respond with a short plain-text question only (no tools)."""


@tool
def draw_for_human() -> str:
    """Deal one hit card to the human player."""
    return "ack"


@tool
def end_turn_human() -> str:
    """The human is done drawing cards for this turn (stand)."""
    return "ack"


def _should_continue(state: BlackjackState) -> Literal["tools", "done"]:
    s = require_seat_tool_loop_state(state)
    human_id = s["human_player_id"]
    if human_id in s["finished_players"]:
        return "done"
    if int(s.get("seat_subgraph_steps", 0)) > SEAT_AGENT_LOOP.max_cycles:
        return "done"
    last = s["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "done"


def _dealer_agent(model: BaseChatModel):
    bound = model.bind_tools([draw_for_human, end_turn_human])

    def _run(state: BlackjackState) -> dict:
        s = require_seat_tool_loop_state(state)
        step = int(s.get("seat_subgraph_steps", 0)) + 1
        response = bound.invoke(s["messages"])
        out: dict = {"messages": [response], "seat_subgraph_steps": step}
        human_id = s["human_player_id"]
        if isinstance(response, AIMessage):
            raw = response.content
            text = (raw if isinstance(raw, str) else str(raw or "")).strip()
            if text:
                prev = list(s.get("chatter_events") or [])
                out["chatter_events"] = prev + [{"seat": human_id, "text": text[:400]}]
        return out

    return _run


def _dealer_tools(state: BlackjackState) -> dict:
    s = require_seat_tool_loop_state(state)
    last = s["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}
    hands_map = {player_id: list(cards) for player_id, cards in s["hands_by_player"].items()}
    human_id = s["human_player_id"]
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
            draw_name="draw_for_human",
            stand_name="end_turn_human",
            unknown_actor="dealer",
        ),
        target_player_id=human_id,
        human_player_id=human_id,
        stream=s["deck_stream"],
        state=acc,
        max_cards_total=CARD_CONFIG.max_cards_in_hand(),
        draw_ack=lambda v: f"Drew {v} for human.",
        stand_ack="Human turn ended (stand).",
    )
    return {
        "messages": acc.tool_messages,
        "hands_by_player": acc.hands_map,
        "next_draw_index": acc.draw_index,
        "finished_players": acc.finished,
        "stood_players": acc.stood,
        "round_log": acc.round_log,
    }


def build_dealer_subgraph(model: BaseChatModel) -> StateGraph:
    graph = StateGraph(BlackjackState)
    graph.add_node("dealer_agent", _dealer_agent(model))
    graph.add_node("dealer_tools", _dealer_tools)
    graph.add_edge(START, "dealer_agent")
    graph.add_conditional_edges(
        "dealer_agent",
        _should_continue,
        {"tools": "dealer_tools", "done": END},
    )
    graph.add_edge("dealer_tools", "dealer_agent")
    return graph
