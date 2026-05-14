# blackjack-sim

Simplified **CLI blackjack** where a LangGraph **crew** (you, three AI seats, and the last seat labeled **`dealer`** as the house bank) applies draws and turns; you play in **natural language** and never see the raw deck or RNG stream. Each seat gets **two opening cards** from the shoe before anyone acts (see ``initial_cards_per_player`` in ``app/settings.py``). **Turn order matches a typical table:** you stay up until you stand, bust, or exhaust the simulator’s **max_draws** limit after the opening deal; only then does play move to the next seat.

## Requirements

- **Python 3.12+**
- An **OpenAI API key** (`OPENAI_API_KEY`)

## Run locally

### 1. Environment

```bash
cd blackjack-sim
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Create **`blackjack-sim/.env`** (see `.env.example`):

```bash
OPENAI_API_KEY=sk-...
```

`app/cli.py` and integration tests load **only** this repo’s **`.env`** (not parent directories).

### 2. Start the game

```bash
blackjack-sim
```

Equivalent:

```bash
python -m app.cli
```

When it is your turn, type a line (e.g. “hit” or “stand”). The table is printed between steps; **deck indices and `deck_stream` are not shown** (see `app/cli_board.py`).

### Useful flags and environment variables

| Flag / env | Meaning |
|------------|--------|
| `--session-id` / `BLACKJACK_SESSION_ID` | LangGraph checkpoint **thread id** (same id resumes across restarts when using the default SQLite file). |
| `BLACKJACK_CHECKPOINT_DB` | SQLite **file** path for checkpoints (default `data/checkpoints.sqlite` under the repo). In-memory URLs are not supported. |
| `--deck-seed` / `BLACKJACK_DECK_SEED` | Integer seed for the deck stream (reproducible). If neither is set, the CLI picks a **random** seed each run. The graph uses a random seed only when ``deck_seed`` is omitted from the initial invoke (e.g. programmatic use). |
| `--model` / `BLACKJACK_MODEL` | OpenAI chat model (default `gpt-4o-mini`). |
| `--with-summary` | After the round, one extra structured call for a short table narration. |

Example:

```bash
BLACKJACK_DECK_SEED=7 blackjack-sim --model gpt-4o-mini
```

### Non-interactive smoke (piped stdin)

If you only want to confirm the binary and API without typing, pipe several lines (one per prompt if the model asks again):

```bash
( for _ in 1 2 3 4 5 6 7 8 9 10; do echo "Stand."; done ) | python -m app.cli --deck-seed 42
```

In a normal session, use **Ctrl+C** to exit.

## Checkpoints (SQLite)

The graph uses LangGraph’s **`SqliteSaver`** via `app/graph/checkpointing.py`. State is keyed by **`configurable["thread_id"]`** (CLI `--session-id`), so you can **resume after closing the terminal** as long as the same SQLite file and thread id are used.

- **Default file:** `data/checkpoints.sqlite` (created automatically; `data/*.sqlite` is gitignored).
- **Override path:** set `BLACKJACK_CHECKPOINT_DB` to an absolute path or a path relative to the repo root (always a real file on disk).
- **Start a brand-new game** with the default file: use a new `--session-id`, or delete the SQLite file.

Details and ops notes are in the module docstring of `app/graph/checkpointing.py`.

## Tests

`tests/conftest.py` points `BLACKJACK_CHECKPOINT_DB` at a **session-scoped temp SQLite file** so tests never reuse your local `data/checkpoints.sqlite`.

`tests/integration/test_checkpoint_sqlite_restart.py` (marked `integration`) simulates a **process crash**: it closes the global SQLite connection, opens a new one on the same file, recompiles the graph, asserts `get_state` still shows the interrupted **dealer** step, then resumes with `Command` until `game_over`.

```bash
pytest tests/ -q
pytest tests/integration -m integration -q   # live OpenAI; needs .env
```
