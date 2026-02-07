from app import create_app

app = create_app()


if __name__ == "__main__":
    import os

    debug_env = (os.getenv("FLASK_DEBUG") or "").strip().lower()
    debug = debug_env in ("1", "true", "yes", "on")
    app.run(debug=debug)
