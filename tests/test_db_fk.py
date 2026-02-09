from app.db import get_db


def test_sqlite_foreign_keys_enabled(app):
    with app.app_context():
        db = get_db()
        row = db.execute("PRAGMA foreign_keys;").fetchone()
        # sqlite3.Row supports index access; first col is the pragma value.
        assert int(row[0]) == 1

