import click
from flask.cli import with_appcontext

from typing import Optional

from .db import get_db
from .services import favorites as favorites_service


@click.command("recompute-favorites")
@click.option("--user-id", type=int, default=None, help="Only recompute for a specific user id.")
@with_appcontext
def recompute_favorites_cmd(user_id: Optional[int]):
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

@click.command("delete-guest")
@with_appcontext
def delete_guest_cmd():
    """Delete the legacy 'guest' user and cascade-delete their data."""
    db_conn = get_db()
    row = db_conn.execute("SELECT id FROM Users WHERE username = ?", ("guest",)).fetchone()
    if row is None:
        click.echo("No guest user found.")
        return
    user_id = int(row["id"])
    db_conn.execute("DELETE FROM Users WHERE id = ?", (user_id,))
    db_conn.commit()
    # Best-effort verification
    fav_cnt = int(
        db_conn.execute("SELECT COUNT(1) AS c FROM Favorites WHERE user_id = ?", (user_id,)).fetchone()["c"]
        or 0
    )
    filt_cnt = int(
        db_conn.execute("SELECT COUNT(1) AS c FROM UserFilters WHERE user_id = ?", (user_id,)).fetchone()["c"]
        or 0
    )
    hist_cnt = int(
        db_conn.execute("SELECT COUNT(1) AS c FROM BrowsingHistory WHERE user_id = ?", (user_id,)).fetchone()["c"]
        or 0
    )
    click.echo(f"Deleted guest user id={user_id}. Remaining rows: Favorites={fav_cnt} UserFilters={filt_cnt} History={hist_cnt}")


def init_app(app):
    """Register CLI hooks."""
    app.cli.add_command(recompute_favorites_cmd)
    app.cli.add_command(delete_guest_cmd)
    return None
