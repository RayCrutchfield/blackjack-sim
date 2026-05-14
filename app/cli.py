"""Command-line entry: one blackjack round with checkpointed LangGraph."""

from __future__ import annotations

import argparse
import os
import secrets
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.cli_session import run_interactive_session


def _load_repository_dotenv() -> None:
    root = Path(__file__).resolve().parent.parent
    local = root / ".env"
    if local.is_file():
        load_dotenv(local, override=True)


def main() -> None:
    _load_repository_dotenv()
    parser = argparse.ArgumentParser(description="Simplified blackjack — AI crew deals; you never touch the deck.")
    parser.add_argument(
        "--session-id",
        default=os.environ.get("BLACKJACK_SESSION_ID") or str(uuid.uuid4()),
        help="LangGraph thread id (checkpoint key). Default: env BLACKJACK_SESSION_ID or random UUID.",
    )
    parser.add_argument(
        "--deck-seed",
        type=int,
        default=None,
        help="Integer seed for the deck stream (reproducible). "
        "Default: BLACKJACK_DECK_SEED from the environment if set, otherwise random.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BLACKJACK_MODEL", "gpt-4o-mini"),
        help="OpenAI chat model name (default gpt-4o-mini).",
    )
    parser.add_argument(
        "--with-summary",
        action="store_true",
        help="After the round, call structured-output narration (extra API tokens).",
    )
    args = parser.parse_args()

    raw_env_seed = (os.environ.get("BLACKJACK_DECK_SEED") or "").strip()
    if args.deck_seed is not None:
        deck_seed = args.deck_seed
    elif raw_env_seed:
        deck_seed = int(raw_env_seed)
    else:
        deck_seed = secrets.randbelow(1 << 31)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to blackjack-sim/.env "
            "(see .env.example) or export it in your shell."
        )

    seat_model = ChatOpenAI(model=args.model, temperature=0.0)
    summary_model = ChatOpenAI(model=args.model, temperature=0.0) if args.with_summary else None

    final = run_interactive_session(
        session_id=args.session_id,
        deck_seed=deck_seed,
        seat_model=seat_model,
        summarization_model=summary_model,
    )


if __name__ == "__main__":
    main()
