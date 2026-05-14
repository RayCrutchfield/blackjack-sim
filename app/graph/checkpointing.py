"""LangGraph checkpointing: durable **SQLite** store (default) keyed by ``thread_id``.

Checkpoints include graph fields (hands, ``finished_players``, ``phase``, …) and
``messages``. Set ``configurable["thread_id"]`` (CLI: ``--session-id``) to resume
the same logical session.

**Configuration**

- ``BLACKJACK_CHECKPOINT_DB`` — SQLite **file** path (always on disk under this design).

  - **Unset:** ``<repo>/data/checkpoints.sqlite`` (parent directory of ``app/``).
  - **Relative path:** resolved against the repo root (where ``app/`` lives).

  In-memory SQLite (``:memory:``) is **not** supported: checkpoints must survive
  process restarts for local resume and for integration tests that simulate a crash.

**Operations**

- **Fresh game, same default file:** pass a new ``--session-id`` *or* delete the
  SQLite file (or run ``rm data/checkpoints.sqlite`` from the repo root).
- **Backup / inspect:** any SQLite client can open the file; schema is owned by LangGraph.

For high-throughput or multi-writer production, prefer Postgres-backed savers from
the LangGraph docs; this SQLite saver is synchronous and intended for local runs
and demos.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

_REPO_ROOT = Path(__file__).resolve().parent.parent

# One open connection per process for the active DB path (SqliteSaver is not free to stack).
_conn: sqlite3.Connection | None = None
_saver: SqliteSaver | None = None
_bound_path: str | None = None


def checkpoint_database_uri() -> str:
    """Return the SQLite database **file** path (absolute after resolution)."""
    raw = (os.environ.get("BLACKJACK_CHECKPOINT_DB") or "").strip()
    if not raw:
        return str(_REPO_ROOT / "data" / "checkpoints.sqlite")
    if raw == ":memory:":
        raise ValueError(
            'BLACKJACK_CHECKPOINT_DB=":memory:" is not supported — use a file path '
            "so checkpoints survive process restarts (see module docstring)."
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return str(candidate)


def _open_connection(uri: str) -> sqlite3.Connection:
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(uri, check_same_thread=False)


def get_checkpointer() -> BaseCheckpointSaver[str]:
    """Return the process-wide SQLite checkpointer for the configured database path."""
    global _conn, _saver, _bound_path
    uri = checkpoint_database_uri()
    if _saver is not None and _bound_path == uri:
        return _saver
    if _conn is not None:
        _conn.close()
        _conn = None
        _saver = None
    _bound_path = uri
    _conn = _open_connection(uri)
    _saver = SqliteSaver(_conn)
    return _saver


def reset_checkpointer_for_tests() -> None:
    """Close the global SQLite connection (advanced tests or multi-path scenarios)."""
    global _conn, _saver, _bound_path
    if _conn is not None:
        _conn.close()
    _conn = None
    _saver = None
    _bound_path = None
