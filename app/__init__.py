import os
import secrets
import datetime as dt
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, flash, redirect, request, session, url_for
from werkzeug.exceptions import HTTPException

from . import auth
from . import cli as cli_commands
from . import db
from . import feed
from . import security
from .services import paper_store


def create_app(test_config=None):
    """Application factory for the frontend-only paper viewer."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
    app = Flask(__name__, instance_relative_config=True)

    default_db_path = Path(app.instance_path) / "app.db"
    default_data_dir = (Path(app.root_path).parent / "CodeArXiv-data").resolve()
    raw_data_dir = (os.getenv("PAPERS_DATA_DIR") or "").strip()
    data_dir_env = Path(raw_data_dir).expanduser() if raw_data_dir else default_data_dir
    if not data_dir_env.is_absolute():
        data_dir_env = (Path(app.root_path).parent / data_dir_env).resolve()
    else:
        data_dir_env = data_dir_env.resolve()
    no_auth_env = os.getenv("NO_AUTH_MODE", "false").lower()
    no_auth_mode = no_auth_env in ("1", "true", "yes", "on")
    # Session stability:
    # - Prefer an explicit FLASK_SECRET_KEY.
    # - Otherwise persist a generated key under instance/ so reloaders/workers won't invalidate sessions.
    raw_secret = (os.getenv("FLASK_SECRET_KEY") or "").strip()
    if raw_secret and raw_secret.lower() != "change-me":
        secret_key = raw_secret
    else:
        key_path = Path(app.instance_path) / "secret_key"
        try:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            if key_path.exists():
                secret_key = key_path.read_text(encoding="utf-8").strip()
            else:
                secret_key = secrets.token_hex(32)
                key_path.write_text(secret_key, encoding="utf-8")
        except Exception:
            # Fall back to an in-memory key; worst case sessions may reset on reload.
            secret_key = secrets.token_hex(32)
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        DATABASE=os.getenv("DATABASE_PATH", str(default_db_path)),
        PAPERS_DATA_DIR=str(data_dir_env),
        NO_AUTH_MODE=no_auth_mode,
        DEFAULT_USER_USERNAME=os.getenv("DEFAULT_USER_USERNAME", "guest"),
        DEFAULT_USER_PASSWORD=os.getenv("DEFAULT_USER_PASSWORD", "guest"),
        # Session hardening (keep defaults explicit; do not force Secure cookies on localhost HTTP).
        SESSION_COOKIE_HTTPONLY=True,
        # Avoid cookie collisions if other local Flask apps share the same domain.
        SESSION_COOKIE_NAME=os.getenv("SESSION_COOKIE_NAME", "codearxiv_session"),
        SESSION_COOKIE_SAMESITE=os.getenv("SESSION_COOKIE_SAMESITE", "Lax"),
        SESSION_COOKIE_SECURE=(os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("1", "true", "yes", "on")),
        PERMANENT_SESSION_LIFETIME=dt.timedelta(
            days=int(os.getenv("SESSION_LIFETIME_DAYS", "7"))
        ),
        SESSION_REFRESH_EACH_REQUEST=True,
        # Auth security
        AUTH_MAX_FAILS=int(os.getenv("AUTH_MAX_FAILS", "10")),
        AUTH_WINDOW_SECONDS=int(os.getenv("AUTH_WINDOW_SECONDS", "600")),
        AUTH_BLOCK_SECONDS=int(os.getenv("AUTH_BLOCK_SECONDS", "300")),
        AUTH_CLEANUP_SECONDS=int(os.getenv("AUTH_CLEANUP_SECONDS", str(60 * 60 * 24 * 7))),
        TRUST_PROXY_HEADERS=(os.getenv("TRUST_PROXY_HEADERS", "false").lower() in ("1", "true", "yes", "on")),
        MIN_PASSWORD_LENGTH=int(os.getenv("MIN_PASSWORD_LENGTH", "8")),
        MAX_USERNAME_LENGTH=int(os.getenv("MAX_USERNAME_LENGTH", "64")),
    )

    if test_config:
        app.config.update(test_config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    cli_commands.init_app(app)
    app.register_blueprint(auth.bp)
    app.register_blueprint(feed.bp)

    # Apply lightweight migrations (adds optional columns if missing)
    with app.app_context():
        db.apply_light_migrations()

    @app.before_request
    def csrf_protect():
        try:
            security.verify_csrf()
        except HTTPException as exc:
            # If the session cookie is missing/rotated, CSRF validation will fail.
            # For login/signup, prefer a friendly retry over a hard 400.
            if (
                exc.code == 400
                and request.method == "POST"
                and request.endpoint in ("auth.login", "auth.signup")
            ):
                # Reset any stale session and ask the user to retry with a fresh token.
                session.clear()
                flash("Session expired. Please try again.", "warning")
                return redirect(url_for(request.endpoint))
            raise

    @app.context_processor
    def inject_csrf_token():
        return {"csrf_token": security.get_csrf_token()}

    @app.route("/health")
    def health():
        dates = paper_store.list_dates()
        payload = {"status": "ok"}
        if dates:
            payload["dates"] = {
                "count": len(dates),
                "min": dates[0].isoformat(),
                "max": dates[-1].isoformat(),
            }
        else:
            payload["dates"] = {"count": 0, "min": None, "max": None}
        return jsonify(payload), 200

    return app
