"""Integration tests: live OpenAI HTTP (no mocked LLM)."""

from __future__ import annotations

import pytest
from openai import OpenAI


@pytest.mark.integration
def test_openai_chat_minimal_completion(openai_api_key: str) -> None:
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly the single word: ok",
            }
        ],
        max_tokens=8,
        temperature=0,
    )
    text = (response.choices[0].message.content or "").strip().lower()
    assert "ok" in text
