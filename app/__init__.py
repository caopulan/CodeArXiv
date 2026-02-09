import os
import secrets
import datetime as dt
import hashlib
import socket
import subprocess
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
    # Prefer the repo's .env over ambient environment variables to avoid
    # surprising behavior across reloaders/workers.
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
    app = Flask(__name__, instance_relative_config=True)

    default_db_path = Path(app.instance_path) / "app.db"
    default_data_dir = (Path(app.root_path).parent / "CodeArXiv-data").resolve()
    raw_data_dir = (os.getenv("PAPERS_DATA_DIR") or "").strip()
    data_dir_env = Path(raw_data_dir).expanduser() if raw_data_dir else default_data_dir
    if not data_dir_env.is_absolute():
        data_dir_env = (Path(app.root_path).parent / data_dir_env).resolve()
    else:
        data_dir_env = data_dir_env.resolve()
    # Session stability:
    # - Prefer an explicit FLASK_SECRET_KEY.
    # - Otherwise persist a generated key under instance/ so reloaders/workers won't invalidate sessions.
    raw_secret = (os.getenv("FLASK_SECRET_KEY") or "").strip()
    secret_source = "env"
    if raw_secret and raw_secret.lower() != "change-me":
        secret_key = raw_secret
    else:
        key_path = Path(app.instance_path) / "secret_key"
        try:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            if key_path.exists():
                secret_key = key_path.read_text(encoding="utf-8").strip()
                secret_source = "instance_file"
            else:
                secret_key = secrets.token_hex(32)
                key_path.write_text(secret_key, encoding="utf-8")
                secret_source = "instance_file"
        except Exception:
            # Fall back to an in-memory key; worst case sessions may reset on reload.
            secret_key = secrets.token_hex(32)
            secret_source = "random"
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        DATABASE=os.getenv("DATABASE_PATH", str(default_db_path)),
        PAPERS_DATA_DIR=str(data_dir_env),
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

    @app.after_request
    def no_cache_dynamic(response):
        """
        Prevent intermediate proxies from caching dynamic HTML/JSON, which can cause
        "stuck" dates or old UI to appear on specific networks/devices.
        """
        endpoint = request.endpoint or ""
        if endpoint == "static" or endpoint.startswith("static") or endpoint == "feed.data_image":
            return response

        # Force no-cache headers (setdefault isn't enough if upstream middleware sets something else).
        response.headers["Cache-Control"] = "private, no-store, no-cache, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        # Extra signals for CDNs that support surrogate directives.
        response.headers["Surrogate-Control"] = "no-store"
        response.headers["CDN-Cache-Control"] = "no-store"

        # If a shared cache is present, separate by session.
        # Note: some CDNs (e.g. Tencent EdgeOne) require enabling Vary support explicitly.
        existing_vary = response.headers.get("Vary", "")
        if "Cookie" not in existing_vary:
            response.headers["Vary"] = (existing_vary + ", Cookie").strip(", ").strip()
        return response

    @app.route("/health")
    def health():
        def _payload():
            dates = paper_store.list_dates()
            payload = {"status": "ok"}
            # Lightweight diagnostics for debugging "session flapping"/routing issues.
            payload["env"] = {
                "session_cookie_name": app.config.get("SESSION_COOKIE_NAME"),
                "papers_data_dir": app.config.get("PAPERS_DATA_DIR"),
            }
            payload["proc"] = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
            }
            try:
                fp = hashlib.sha256((app.config.get("SECRET_KEY") or "").encode("utf-8")).hexdigest()[:10]
            except Exception:
                fp = None
            payload["secret_key"] = {"source": secret_source, "fp": fp}
            try:
                repo_root = Path(app.root_path).parent
                if (repo_root / ".git").exists():
                    sha = (
                        subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"])
                        .decode("utf-8", "ignore")
                        .strip()
                    )
                else:
                    sha = None
            except Exception:
                sha = None
            payload["build"] = {"git_sha": sha}
            if dates:
                payload["dates"] = {
                    "count": len(dates),
                    "min": dates[0].isoformat(),
                    "max": dates[-1].isoformat(),
                }
            else:
                payload["dates"] = {"count": 0, "min": None, "max": None}
            return payload

        return jsonify(_payload()), 200

    @app.route("/healthz")
    def healthz():
        """Uncached health endpoint for debugging CDN/proxy caching."""
        now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        # Rebuild payload to avoid relying on cached /health behavior.
        dates = paper_store.list_dates()
        payload = {
            "status": "ok",
            "utc": now,
            "env": {
                "session_cookie_name": app.config.get("SESSION_COOKIE_NAME"),
                "papers_data_dir": app.config.get("PAPERS_DATA_DIR"),
            },
            "proc": {"hostname": socket.gethostname(), "pid": os.getpid()},
            "secret_key": {
                "source": secret_source,
                "fp": hashlib.sha256((app.config.get("SECRET_KEY") or "").encode("utf-8")).hexdigest()[:10]
                if (app.config.get("SECRET_KEY") or "")
                else None,
            },
            "build": {"git_sha": None},
            "dates": {
                "count": len(dates),
                "min": dates[0].isoformat() if dates else None,
                "max": dates[-1].isoformat() if dates else None,
            },
        }
        try:
            repo_root = Path(app.root_path).parent
            if (repo_root / ".git").exists():
                payload["build"]["git_sha"] = (
                    subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"])
                    .decode("utf-8", "ignore")
                    .strip()
                )
        except Exception:
            pass
        return jsonify(payload), 200

    return app
