import sys
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app
from app import db as db_mod


@pytest.fixture()
def app(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "test.db"
    app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(db_path),
            "PAPERS_DATA_DIR": str(data_dir),
            "WTF_CSRF_ENABLED": False,
        }
    )
    with app.app_context():
        db_mod.init_db()
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def extract_csrf(html: str) -> str:
    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    assert m, "csrf_token input not found"
    return m.group(1)
