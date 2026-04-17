#!/usr/bin/env python3
"""
reset_db.py — Drop and recreate the public schema in the PostgreSQL database.

Usage:
    python scripts/reset_db.py

Requires the DATABASE_URL environment variable to be set (or a .env file in
the project root).  The script works with both async-driver prefixes used by
the app (postgresql+asyncpg://...) and plain postgresql:// URLs.

WARNING: This permanently deletes ALL tables and data.  The application will
recreate the schema automatically on its next startup via init_db().
"""

import os
import sys
import re

# ── Load .env from the project root (one level up from scripts/) ──────────────
try:
    from dotenv import load_dotenv

    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on the environment directly


def _normalise_url(url: str) -> str:
    """
    Convert an async-driver SQLAlchemy URL to a plain psycopg2-compatible URL.

    postgresql+asyncpg://user:pass@host/db  →  postgresql://user:pass@host/db
    postgres://...                          →  postgresql://...
    """
    # Strip async driver suffix (e.g. +asyncpg, +aiopg)
    url = re.sub(r"^postgresql\+\w+://", "postgresql://", url)
    # Heroku / Railway sometimes emit "postgres://" — SQLAlchemy requires "postgresql://"
    url = re.sub(r"^postgres://", "postgresql://", url)
    return url


def main() -> None:
    raw_url = os.getenv("DATABASE_URL")
    if not raw_url:
        print(
            "ERROR: DATABASE_URL environment variable is not set.\n"
            "       Export it or add it to a .env file in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Reject SQLite — this script is PostgreSQL-only
    if raw_url.startswith("sqlite"):
        print(
            "ERROR: DATABASE_URL points to SQLite, not PostgreSQL.\n"
            "       This script only supports PostgreSQL.",
            file=sys.stderr,
        )
        sys.exit(1)

    db_url = _normalise_url(raw_url)

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print(
            "ERROR: SQLAlchemy is not installed.\n"
            "       Run: pip install sqlalchemy psycopg2-binary",
            file=sys.stderr,
        )
        sys.exit(1)

    # psycopg2 is the sync driver; fall back to pg8000 if available
    for driver in ("psycopg2", "pg8000"):
        try:
            __import__(driver.replace("-", "_"))
            break
        except ImportError:
            continue
    else:
        print(
            "ERROR: No synchronous PostgreSQL driver found.\n"
            "       Run: pip install psycopg2-binary   (or pg8000)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Connecting to database …")

    try:
        engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            print("  → Dropping public schema (all tables, types, sequences) …")
            conn.execute(text("DROP SCHEMA public CASCADE"))

            print("  → Recreating public schema …")
            conn.execute(text("CREATE SCHEMA public"))

            # Restore default privileges so the app user can create objects
            conn.execute(text("GRANT ALL ON SCHEMA public TO PUBLIC"))

        engine.dispose()

    except Exception as exc:
        print(f"ERROR: Database reset failed.\n       {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "\n✓ Database reset complete.\n"
        "  All tables and data have been removed.\n"
        "  Start the application to reinitialise the schema automatically."
    )


if __name__ == "__main__":
    main()
