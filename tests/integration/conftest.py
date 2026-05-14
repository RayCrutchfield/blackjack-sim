"""Load API keys for integration tests (real OpenAI calls; no LLM mocks)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pytest_configure(config: pytest.Config) -> None:
    root = _repository_root()
    local_env = root / ".env"
    if local_env.is_file():
        load_dotenv(local_env, override=True)


@pytest.fixture
def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip(
            "OPENAI_API_KEY missing: create blackjack-sim/.env (see .env.example) "
            "or export the variable in your environment."
        )
    return key
