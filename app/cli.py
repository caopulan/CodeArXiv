import click
from flask.cli import with_appcontext

from .db import get_db
from .services import favorites as favorites_service


@click.command("recompute-favorites")
@click.option("--user-id", type=int, default=None, help="Only recompute for a specific user id.")
@with_appcontext
def recompute_favorites_cmd(user_id: int | None):
    """Recompute favorite embeddings from current paper data."""
    db_conn = get_db()
    if user_id is None:
        rows = db_conn.execute("SELECT id FROM Favorites").fetchall()
    else:
        rows = db_conn.execute(
            "SELECT id FROM Favorites WHERE user_id = ?", (user_id,)
        ).fetchall()
    count = 0
    for row in rows:
        favorites_service.recompute_favorite_embedding(row["id"])
        count += 1
    click.echo(f"Recomputed {count} favorite(s).")


def init_app(app):
    """Register CLI hooks."""
    app.cli.add_command(recompute_favorites_cmd)
    return None
