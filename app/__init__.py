import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify

from . import auth
from . import cli as cli_commands
from . import db
from . import feed
from . import security


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
    # Avoid a hard-coded default secret key; encourage setting FLASK_SECRET_KEY in .env.
    secret_key = (os.getenv("FLASK_SECRET_KEY") or "").strip() or secrets.token_hex(32)
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        DATABASE=os.getenv("DATABASE_PATH", str(default_db_path)),
        PAPERS_DATA_DIR=str(data_dir_env),
        NO_AUTH_MODE=no_auth_mode,
        DEFAULT_USER_USERNAME=os.getenv("DEFAULT_USER_USERNAME", "guest"),
        DEFAULT_USER_PASSWORD=os.getenv("DEFAULT_USER_PASSWORD", "guest"),
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
        security.verify_csrf()

    @app.context_processor
    def inject_csrf_token():
        return {"csrf_token": security.get_csrf_token()}

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    return app
