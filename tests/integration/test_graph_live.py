"""Live OpenAI: parent graph with two seats (costs a few tokens)."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.graph.parent_graph import compile_blackjack_graph
from app.graph.structured_summary import summarize_table_structured

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _load_env() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")


@pytest.fixture
def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY missing")
    return key


def test_live_mini_graph_human_then_ai(openai_key: str) -> None:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_key)
    graph = compile_blackjack_graph(model, summarization_model=None)
    result = graph.invoke(
        {
            "pending_user_text": (
                "Call the tool end_turn_human once (no arguments). Do not use plain text."
            ),
            "player_order": ["human", "agent_a"],
            "deck_seed": 42,
        },
        config={
            "configurable": {"thread_id": "live-mini-graph"},
            "recursion_limit": 120,
        },
    )
    assert result["phase"] == "game_over"
    assert "human" in result["finished_players"]


def test_structured_table_narration(openai_key: str) -> None:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_key)
    narration = summarize_table_structured(
        model,
        hands_by_player={"human": [10, 8], "agent_a": [9, 9]},
        winners=["agent_a"],
    )
    assert narration.headline.strip()
    assert "human" in narration.detail or "18" in narration.detail
