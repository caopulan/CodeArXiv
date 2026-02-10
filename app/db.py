import sqlite3
from pathlib import Path

import click
from flask import current_app, g
from flask.cli import with_appcontext


def _connect(db_path: str):
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except sqlite3.OperationalError:
        pass
    return conn


def get_db():
    """Return the main SQLite connection stored on the application context."""
    if "db" not in g:
        db_path = current_app.config["DATABASE"]
        g.db = _connect(db_path)
    return g.db


def close_db(e=None):
    """Close database connections for the current context."""
    for key in ("db",):
        db_conn = g.pop(key, None)
        if db_conn is not None:
            db_conn.close()


def _run_schema(db_conn, schema_filename: str):
    schema_path = Path(current_app.root_path) / schema_filename
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        db_conn.executescript(schema_file.read())
    db_conn.commit()


def init_main_db():
    """Create tables for the main application database if they do not exist."""
    _run_schema(get_db(), "schema.sql")


def init_db():
    """Create tables in the main database."""
    init_main_db()


def _ensure_columns(db_conn, table: str, columns: dict[str, str]) -> None:
    info_rows = db_conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not info_rows:
        return
    existing = {row["name"] for row in info_rows}
    for col, ddl in columns.items():
        if col not in existing:
            db_conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
    db_conn.commit()


def _table_exists(db_conn, table: str) -> bool:
    row = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def apply_light_migrations() -> None:
    """
    Best-effort migrations: add optional columns if missing.
    Does not drop or modify existing columns.
    """
    main_db = get_db()
    if not _table_exists(main_db, "Users"):
        return
    # Ensure the minimal Papers table exists (older deployments used FK constraints to it).
    if not _table_exists(main_db, "Papers"):
        main_db.execute(
            """
            CREATE TABLE IF NOT EXISTS Papers (
                id TEXT PRIMARY KEY
            );
            """
        )
        main_db.commit()
    else:
        # Ensure the table is usable even if created with a slightly different schema.
        pass

    # Backfill Papers from any existing references to avoid FK errors once foreign_keys=ON.
    for tbl in ("FavoritePapers", "BrowsingHistory"):
        if _table_exists(main_db, tbl):
            try:
                main_db.execute(
                    f"""
                    INSERT OR IGNORE INTO Papers (id)
                    SELECT DISTINCT paper_id FROM {tbl}
                    WHERE paper_id IS NOT NULL AND TRIM(paper_id) != ''
                    """
                )
            except Exception:
                continue
    main_db.commit()
    _ensure_columns(
        main_db,
        "Users",
        {
            "language_preference": "TEXT DEFAULT 'en'",
        },
    )
    if not _table_exists(main_db, "UserFilters"):
        main_db.execute(
            """
            CREATE TABLE IF NOT EXISTS UserFilters (
                user_id INTEGER PRIMARY KEY,
                categories TEXT,
                tags TEXT,
                sim_favorites TEXT,
                last_date TEXT,
                last_paper_id TEXT,
                last_position INTEGER,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES Users (id) ON DELETE CASCADE
            );
            """
        )
    else:
        _ensure_columns(
            main_db,
            "UserFilters",
            {
                "last_date": "TEXT",
                "last_paper_id": "TEXT",
                "last_position": "INTEGER",
            },
        )
    main_db.commit()

    if not _table_exists(main_db, "AuthAttempts"):
        main_db.execute(
            """
            CREATE TABLE IF NOT EXISTS AuthAttempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                ip TEXT,
                ts INTEGER NOT NULL,
                success INTEGER NOT NULL
            );
            """
        )
        main_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip_ts ON AuthAttempts (ip, ts);"
        )
        main_db.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_attempts_user_ip_ts ON AuthAttempts (username, ip, ts);"
        )
        main_db.commit()


@click.command("init-db")
@with_appcontext
def init_db_command():
    """CLI command: initialize the main and papers databases."""
    init_db()
    click.echo("Initialized the database.")


@click.command("db-info")
@with_appcontext
def db_info_command():
    """CLI command: print table names for the main database."""

    def _print_tables(conn, label: str):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        click.echo(label)
        if not rows:
            click.echo("  (no tables)")
            return
        for row in rows:
            click.echo(f"- {row['name']}")

    click.echo(f"Main DB: {current_app.config['DATABASE']}")
    _print_tables(get_db(), "Tables:")


def init_app(app):
    """Register database helpers with the Flask app."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(db_info_command)
