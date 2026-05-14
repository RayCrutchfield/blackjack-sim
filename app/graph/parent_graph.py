"""Parent ``StateGraph``: turn order, seat routing, dealer/AI subgraphs, SQLite checkpoint."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START

from app.domain.actions import apply_draw
from app.graph.ai_subgraph import ai_seat_system_content, build_ai_subgraph
from app.graph.checkpointing import get_checkpointer
from app.graph.dealer_subgraph import DEALER_SYSTEM, build_dealer_subgraph
from app.graph.deck_stream import materialize_deck_stream
from app.graph.state import BlackjackState, hands_from_domain, hands_to_domain
from app.graph.structured_summary import summarize_table_structured
from app.graph.turns import advance_after_seat, initial_player_order
from app.settings import CARD_CONFIG, DEFAULT_HUMAN_PLAYER_ID


def _resolve_deck_seed(state: BlackjackState) -> int:
    """Use caller-supplied ``deck_seed`` when present; otherwise a fresh random stream."""
    if "deck_seed" not in state:
        return secrets.randbelow(1 << 31)
    return int(state["deck_seed"])


def _deal_opening_from_stream(
    order: list[str],
    stream: list[int],
    *,
    cards_per_player: int,
) -> tuple[dict[str, list[int]], int]:
    """
    Deal ``cards_per_player`` cards to each seat in ``order``, round-robin, from ``stream``.

    Returns ``(hands_by_player, next_draw_index)``.
    """
    if cards_per_player <= 0:
        return {player_id: [] for player_id in order}, 0
    domain = hands_to_domain({player_id: [] for player_id in order})
    draw_index = 0
    for _ in range(cards_per_player):
        for player_id in order:
            if draw_index >= len(stream):
                return hands_from_domain(domain), draw_index
            value = stream[draw_index]
            draw_index += 1
            domain = dict(
                apply_draw(
                    domain,
                    player_id,
                    value,
                    max_cards_total=CARD_CONFIG.max_cards_in_hand(),
                )
            )
    return hands_from_domain(domain), draw_index


def _init_round(state: BlackjackState) -> dict:
    order = initial_player_order(state.get("player_order"))
    seed = _resolve_deck_seed(state)
    stream = materialize_deck_stream(seed=seed)
    n_open = CARD_CONFIG.initial_cards_per_player
    hands, next_idx = _deal_opening_from_stream(order, stream, cards_per_player=n_open)
    human_id = state.get("human_player_id", DEFAULT_HUMAN_PLAYER_ID)
    return {
        "player_order": order,
        "human_player_id": human_id,
        "hands_by_player": hands,
        "finished_players": [],
        "stood_players": [],
        "chatter_events": [],
        "round_log": [],
        "deck_stream": stream,
        "deck_seed": seed,
        "next_draw_index": next_idx,
        "active_seat": order[0],
        "phase": "playing",
        "winners": [],
        "messages": [SystemMessage(content="Blackjack round initialized.")],
    }


def _route_seat(state: BlackjackState) -> Literal["human", "ai", "finish"]:
    if state.get("phase") == "game_over":
        return "finish"
    if state["active_seat"] == state["human_player_id"]:
        return "human"
    return "ai"


def _advance(state: BlackjackState) -> dict:
    return advance_after_seat(
        hands_by_player=state["hands_by_player"],
        player_order=state["player_order"],
        finished_players=state["finished_players"],
        active_seat=state["active_seat"],
    )


def _summarize_factory(summarization_model: BaseChatModel | None):
    def _summarize(state: BlackjackState) -> dict:
        if summarization_model is None:
            return {}
        narration = summarize_table_structured(
            summarization_model,
            hands_by_player=state["hands_by_player"],
            winners=list(state.get("winners", [])),
        )
        text = f"{narration.headline} ({narration.detail})"
        return {"messages": [AIMessage(content=text)]}

    return _summarize


def _echo_round_log_line(hook: Callable[..., None] | None, line: str) -> None:
    """Call ``hook(line)``; use ``flush=True`` when the hook accepts it (e.g. ``print``)."""
    if hook is None:
        return
    try:
        hook(line, flush=True)  # type: ignore[misc]
    except TypeError:
        hook(line)


def _stream_seat_subgraph(
    app: Any,
    inner: dict,
    *,
    round_log_echo: Callable[..., None] | None,
) -> dict:
    """
    Run a compiled seat subgraph with ``stream_mode='values'`` so ``round_log`` lines
    can be printed as each inner step completes (tool draws / stands), not only after
    ``invoke`` returns.
    """
    prev_len = len(inner.get("round_log") or [])
    final: dict | None = None
    for chunk in app.stream(inner, stream_mode="values"):
        if not isinstance(chunk, dict):
            continue
        final = chunk
        log = chunk.get("round_log") or []
        while prev_len < len(log):
            _echo_round_log_line(round_log_echo, str(log[prev_len]))
            prev_len += 1
    if final is None:
        return dict(app.invoke(inner))
    return final


def compile_blackjack_graph(
    seat_model: BaseChatModel,
    *,
    summarization_model: BaseChatModel | None = None,
    interrupt_before_dealer: bool = False,
    round_log_echo: Callable[..., None] | None = None,
):
    """
    Build and compile the full table orchestrator with the SQLite checkpointer from
    :func:`app.graph.checkpointing.get_checkpointer`.

    Uses durable checkpoints, an explicit ``StateGraph``, and nested compiled seat graphs
    (dealer / AI) invoked behind a thin wrapper so each seat starts a **fresh**
    OpenAI-valid ``[system, human]`` thread (avoids ``ToolMessage`` orphans in
    the chat API).

    When ``round_log_echo`` is set (e.g. ``print``), seat subgraphs are **streamed** so
    new ``round_log`` lines are emitted as tool effects run instead of only after the
    subgraph finishes.
    """
    dealer_app = build_dealer_subgraph(seat_model).compile()
    ai_app = build_ai_subgraph(seat_model).compile()

    def dealer_turn(state: BlackjackState) -> dict:
        inner: dict = dict(state)
        inner["messages"] = [
            SystemMessage(content=DEALER_SYSTEM),
            HumanMessage(content=state.get("pending_user_text") or ""),
        ]
        inner["seat_subgraph_steps"] = 0
        if round_log_echo is None:
            return dealer_app.invoke(inner)
        return _stream_seat_subgraph(dealer_app, inner, round_log_echo=round_log_echo)

    def ai_turn(state: BlackjackState) -> dict:
        inner: dict = dict(state)
        inner["messages"] = [
            SystemMessage(content=ai_seat_system_content(state)),
            HumanMessage(content="Use tools only; play to win: draw_for_self or end_turn."),
        ]
        inner["seat_subgraph_steps"] = 0
        if round_log_echo is None:
            return ai_app.invoke(inner)
        return _stream_seat_subgraph(ai_app, inner, round_log_echo=round_log_echo)

    graph = StateGraph(BlackjackState)
    graph.add_node("init_round", _init_round)
    graph.add_node("dealer", dealer_turn)
    graph.add_node("ai_seat", ai_turn)
    graph.add_node("advance", _advance)
    graph.add_node("summarize_structured", _summarize_factory(summarization_model))

    graph.add_edge(START, "init_round")
    graph.add_conditional_edges(
        "init_round",
        _route_seat,
        {"human": "dealer", "ai": "ai_seat", "finish": "summarize_structured"},
    )
    graph.add_edge("dealer", "advance")
    graph.add_edge("ai_seat", "advance")
    graph.add_conditional_edges(
        "advance",
        _route_seat,
        {"human": "dealer", "ai": "ai_seat", "finish": "summarize_structured"},
    )
    graph.add_edge("summarize_structured", END)

    memory = get_checkpointer()
    compile_kw: dict = {"checkpointer": memory}
    if interrupt_before_dealer:
        compile_kw["interrupt_before"] = ["dealer"]
    return graph.compile(**compile_kw)
