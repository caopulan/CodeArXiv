import functools
import time

from flask import (
    Blueprint,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from .db import get_db
from .security import safe_redirect_target

bp = Blueprint("auth", __name__, url_prefix="/auth")


def _normalize_user_row(row):
    if row is None:
        return None
    data = dict(row)
    if not data.get("language_preference"):
        data["language_preference"] = "en"
    return data


def _ensure_default_user():
    """Create and return the default user for no-auth mode."""
    db_conn = get_db()
    username = current_app.config.get("DEFAULT_USER_USERNAME", "default_user")
    password = current_app.config.get("DEFAULT_USER_PASSWORD", "")
    user = (
        db_conn.execute(
            "SELECT id, username, password, language_preference FROM Users WHERE username = ?",
            (username,),
        )
        .fetchone()
    )
    if user is None:
        password_hash = generate_password_hash(password)
        db_conn.execute(
            "INSERT OR IGNORE INTO Users (username, password, language_preference) VALUES (?, ?, ?)",
            (username, password_hash, "en"),
        )
        db_conn.commit()
        user = (
            db_conn.execute(
                "SELECT id, username, language_preference FROM Users WHERE username = ?",
                (username,),
            )
            .fetchone()
        )
        return _normalize_user_row(user)

    user_data = _normalize_user_row(user)
    stored_password = user_data.pop("password", "")
    if stored_password and "$" not in stored_password:
        db_conn.execute(
            "UPDATE Users SET password = ? WHERE id = ?",
            (generate_password_hash(stored_password), user_data["id"]),
        )
        db_conn.commit()
    return user_data


def _client_ip() -> str:
    if current_app.config.get("TRUST_PROXY_HEADERS"):
        xff = (request.headers.get("X-Forwarded-For") or "").strip()
        if xff:
            return xff.split(",")[0].strip()
    return (request.remote_addr or "").strip()


def _cleanup_auth_attempts(db_conn) -> None:
    cutoff = int(time.time()) - int(current_app.config.get("AUTH_CLEANUP_SECONDS", 60 * 60 * 24 * 7))
    try:
        db_conn.execute("DELETE FROM AuthAttempts WHERE ts < ?", (cutoff,))
        db_conn.commit()
    except Exception:
        # Best-effort cleanup; don't block login if table is missing/corrupt.
        return


def _record_auth_attempt(db_conn, *, username: str, ip: str, success: bool) -> None:
    ts = int(time.time())
    try:
        db_conn.execute(
            "INSERT INTO AuthAttempts (username, ip, ts, success) VALUES (?, ?, ?, ?)",
            (username[:256], ip[:128], ts, 1 if success else 0),
        )
        db_conn.commit()
    except Exception:
        return


def _is_rate_limited(db_conn, *, username: str, ip: str) -> bool:
    now = int(time.time())
    window = int(current_app.config.get("AUTH_WINDOW_SECONDS", 600))
    max_fails = int(current_app.config.get("AUTH_MAX_FAILS", 10))
    block = int(current_app.config.get("AUTH_BLOCK_SECONDS", 300))
    since = max(0, now - window)

    try:
        last_success_row = db_conn.execute(
            """
            SELECT MAX(ts) AS ts
            FROM AuthAttempts
            WHERE ip = ? AND username = ? AND success = 1 AND ts >= ?
            """,
            (ip, username, since),
        ).fetchone()
        last_success = int(last_success_row["ts"] or 0)
        effective_since = max(since, last_success)

        fail_row = db_conn.execute(
            """
            SELECT COUNT(1) AS cnt, MAX(ts) AS last_fail
            FROM AuthAttempts
            WHERE ip = ? AND username = ? AND success = 0 AND ts >= ?
            """,
            (ip, username, effective_since),
        ).fetchone()
        fail_cnt = int(fail_row["cnt"] or 0)
        last_fail = int(fail_row["last_fail"] or 0)
        if fail_cnt >= max_fails and last_fail and (now - last_fail) < block:
            return True

        # Backup IP-only throttling (helps if attacker rotates usernames).
        ip_fail_row = db_conn.execute(
            """
            SELECT COUNT(1) AS cnt, MAX(ts) AS last_fail
            FROM AuthAttempts
            WHERE ip = ? AND success = 0 AND ts >= ?
            """,
            (ip, since),
        ).fetchone()
        ip_fail_cnt = int(ip_fail_row["cnt"] or 0)
        ip_last_fail = int(ip_fail_row["last_fail"] or 0)
        if ip_fail_cnt >= (max_fails * 2) and ip_last_fail and (now - ip_last_fail) < block:
            return True
    except Exception:
        # If the attempts table is unavailable, don't block login.
        return False

    return False


@bp.before_app_request
def load_logged_in_user():
    if current_app.config.get("NO_AUTH_MODE"):
        # In no-auth mode we auto-login a default user, but we should not
        # overwrite an existing session identity (e.g. when a real user session
        # already exists or when multiple workers have different defaults).
        user_id = session.get("user_id")
        if user_id is not None:
            row = (
                get_db()
                .execute(
                    "SELECT id, username, language_preference FROM Users WHERE id = ?",
                    (user_id,),
                )
                .fetchone()
            )
            existing = _normalize_user_row(row)
            if existing is not None:
                g.user = existing
                session.permanent = True
                g.language_preference = existing.get("language_preference", "en")
                return

        user = _ensure_default_user()
        session["user_id"] = user["id"]
        session.permanent = True
        g.user = user
        g.language_preference = user.get("language_preference", "en")
        return
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        row = (
            get_db()
            .execute(
                "SELECT id, username, language_preference FROM Users WHERE id = ?",
                (user_id,),
            )
            .fetchone()
        )
        g.user = _normalize_user_row(row)
    g.language_preference = (g.user or {}).get("language_preference", "en") if isinstance(g.user, dict) else "en"


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if current_app.config.get("NO_AUTH_MODE"):
            # Keep any existing session identity; only fall back to the default.
            if g.get("user") is None and session.get("user_id") is None:
                user = _ensure_default_user()
                session["user_id"] = user["id"]
                g.user = user
            return view(**kwargs)
        if g.user is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(**kwargs)

    return wrapped_view


@bp.route("/signup", methods=("GET", "POST"))
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        error = None

        max_user_len = int(current_app.config.get("MAX_USERNAME_LENGTH", 64))
        min_pw_len = int(current_app.config.get("MIN_PASSWORD_LENGTH", 8))
        if not username or not password:
            error = "Username and password are required."
        elif len(username) > max_user_len:
            error = "Username is too long."
        elif len(password) < min_pw_len:
            error = f"Password must be at least {min_pw_len} characters."
        elif (
            get_db()
            .execute("SELECT id FROM Users WHERE username = ?", (username,))
            .fetchone()
            is not None
        ):
            error = "User already exists."

        if error is None:
            password_hash = generate_password_hash(password)
            db_conn = get_db()
            cursor = db_conn.execute(
                "INSERT INTO Users (username, password, language_preference) VALUES (?, ?, ?)",
                (username, password_hash, "en"),
            )
            db_conn.commit()
            session.clear()
            session["user_id"] = cursor.lastrowid
            session.permanent = True
            flash("Signup successful.", "success")
            return redirect(url_for("feed.index"))

        flash(error, "danger")
    return render_template("signup.html")


@bp.route("/login", methods=("GET", "POST"))
def login():
    if current_app.config.get("NO_AUTH_MODE"):
        return redirect(url_for("feed.index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        error = None

        db_conn = get_db()
        _cleanup_auth_attempts(db_conn)
        ip = _client_ip()
        max_user_len = int(current_app.config.get("MAX_USERNAME_LENGTH", 64))
        if len(username) > max_user_len:
            username = username[:max_user_len]
        if _is_rate_limited(db_conn, username=username, ip=ip):
            flash("Too many login attempts. Please try again later.", "danger")
            _record_auth_attempt(db_conn, username=username, ip=ip, success=False)
            return render_template("login.html"), 429
        user = (
            db_conn.execute(
                "SELECT id, username, password FROM Users WHERE username = ?",
                (username,),
            )
            .fetchone()
        )

        password_ok = False
        if user is not None:
            stored_password = user["password"] or ""
            password_ok = check_password_hash(stored_password, password)
            if not password_ok and "$" not in stored_password and stored_password == password:
                password_ok = True
                db_conn.execute(
                    "UPDATE Users SET password = ? WHERE id = ?",
                    (generate_password_hash(password), user["id"]),
                )
                db_conn.commit()

        if user is None or not password_ok:
            error = "Invalid username or password."

        if error is None:
            session.clear()
            session["user_id"] = user["id"]
            session.permanent = True
            flash("Welcome back!", "success")
            next_url = request.args.get("next")
            _record_auth_attempt(db_conn, username=username, ip=ip, success=True)
            return redirect(safe_redirect_target(next_url, url_for("feed.index")))

        _record_auth_attempt(db_conn, username=username, ip=ip, success=False)
        flash(error, "danger")

    return render_template("login.html")


@bp.route("/logout")
def logout():
    if current_app.config.get("NO_AUTH_MODE"):
        # In no-auth mode, keep user logged in to avoid blocking.
        return redirect(url_for("feed.index"))
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))
