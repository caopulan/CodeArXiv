import re

from werkzeug.security import generate_password_hash

from app.db import get_db


def extract_csrf(html: str) -> str:
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    assert m, "csrf_token input not found"
    return m.group(1)


def test_requires_login_redirects_to_login(client):
    r = client.get("/")
    assert r.status_code in (301, 302)
    assert "/auth/login" in (r.headers.get("Location") or "")


def test_signup_creates_user_and_filters_and_logs_in(client, app):
    r = client.get("/auth/signup")
    assert r.status_code == 200
    token = extract_csrf(r.data.decode("utf-8", "ignore"))

    r2 = client.post(
        "/auth/signup",
        data={"username": "alice", "password": "password123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r2.status_code in (301, 302)

    with client.session_transaction() as sess:
        user_id = sess.get("user_id")
    assert user_id is not None

    with app.app_context():
        db = get_db()
        row = db.execute("SELECT user_id FROM UserFilters WHERE user_id = ?", (user_id,)).fetchone()
        assert row is not None


def test_login_logout_roundtrip(client, app):
    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO Users (username, password, language_preference) VALUES (?, ?, ?)",
            ("bob", generate_password_hash("pw123456"), "en"),
        )
        db.commit()

    r = client.get("/auth/login")
    token = extract_csrf(r.data.decode("utf-8", "ignore"))
    r2 = client.post(
        "/auth/login",
        data={"username": "bob", "password": "pw123456", "csrf_token": token},
        follow_redirects=False,
    )
    assert r2.status_code in (301, 302)

    r3 = client.get("/")
    assert r3.status_code == 200

    r4 = client.get("/auth/logout", follow_redirects=False)
    assert r4.status_code in (301, 302)
    assert "/auth/login" in (r4.headers.get("Location") or "")

    r5 = client.get("/", follow_redirects=False)
    assert r5.status_code in (301, 302)
    assert "/auth/login" in (r5.headers.get("Location") or "")


def test_no_guest_auto_login(client, app):
    # Hitting / should not create any user implicitly.
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302)
    with app.app_context():
        db = get_db()
        cnt = db.execute("SELECT COUNT(1) AS c FROM Users").fetchone()["c"]
        assert int(cnt) == 0


def test_reserved_username_guest_cannot_signup(client):
    r = client.get("/auth/signup")
    token = extract_csrf(r.data.decode("utf-8", "ignore"))
    r2 = client.post(
        "/auth/signup",
        data={"username": "guest", "password": "password123", "csrf_token": token},
        follow_redirects=True,
    )
    body = r2.data.decode("utf-8", "ignore").lower()
    assert "reserved" in body or "danger" in body
