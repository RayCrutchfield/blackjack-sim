"""Interactive session: checkpointed graph + human input at dealer interrupts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from app.cli_board import format_table, seat_display_label
from app.graph.parent_graph import compile_blackjack_graph


def run_interactive_session(
    *,
    session_id: str,
    deck_seed: int,
    seat_model: BaseChatModel | None = None,
    summarization_model: BaseChatModel | None = None,
    player_order: list[str] | None = None,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Run one full round with ``interrupt_before=['dealer']`` so the human types
    each time the graph pauses before the dealer node.

    The human keeps the same ``active_seat`` until they stand, bust, or hit the
    per-hand card cap; only then does play move to the next seat (casino order).

    ``input_fn`` defaults to ``input``; tests inject a scripted callable.
    """
    model = seat_model or ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    read_line = input_fn or input
    echo = print_fn or print

    graph = compile_blackjack_graph(
        model,
        summarization_model=summarization_model,
        interrupt_before_dealer=True,
        round_log_echo=echo,
    )

    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
    initial: dict[str, Any] = {"deck_seed": deck_seed}
    if player_order is not None:
        initial["player_order"] = player_order

    graph.invoke(initial, config)
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        node = snapshot.next[0]
        echo("")
        echo(format_table(snapshot.values))
        if node == "dealer":
            hid = snapshot.values["human_player_id"]
            echo(f"Start {seat_display_label(hid)}'s turn.")
            line = read_line(
                "Your play (natural language; e.g. hit me, one more; stand, I'm good): "
            ).strip()
            if line.lower() in {"quit", "exit", "q"}:
                echo("Leaving the table early.")
                return dict(snapshot.values)
            graph.invoke(Command(update={"pending_user_text": line}), config)
        else:
            seat = snapshot.values["active_seat"]
            echo(f"Start {seat_display_label(seat)}'s turn.")
            graph.invoke(None, config)

    final = graph.get_state(config).values
    echo("")
    phase = final.get("phase")
    echo("— Round complete —" if phase == "game_over" else "— Session ended —")
    echo(format_table(final))
    return dict(final)

