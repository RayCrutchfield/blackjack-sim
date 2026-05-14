"""Pydantic structured output for a one-line table summary via LangChain ``with_structured_output``."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class TableNarration(BaseModel):
    """Short structured description of the table after a round (for logs / CLI)."""

    headline: str = Field(description="One sentence: who leads or that the round ended.")
    detail: str = Field(
        description="Comma-separated each player id with their current total or BUST."
    )


def summarize_table_structured(
    model: BaseChatModel,
    *,
    hands_by_player: dict[str, list[int]],
    winners: list[str],
) -> TableNarration:
    """Call ``model.with_structured_output(TableNarration)`` on a short factual table string."""
    lines = []
    for player_id, cards in sorted(hands_by_player.items()):
        total = sum(cards)
        label = "BUST" if total > 21 else str(total)
        lines.append(f"{player_id}={label}")
    payload = "Hands: " + "; ".join(lines) + f". Official winner(s): {winners}."
    structured = model.with_structured_output(TableNarration)
    return structured.invoke(
        [
            SystemMessage(
                content=(
                    "You echo table facts in structured fields only. "
                    "Do not invent totals; use the numbers implied by the user message."
                )
            ),
            HumanMessage(content=payload),
        ]
    )
