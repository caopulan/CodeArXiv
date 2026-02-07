from __future__ import annotations

import secrets
from urllib.parse import urljoin, urlparse

from flask import abort, request, session

SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}

CSRF_SESSION_KEY = "_csrf_token"
CSRF_FORM_KEY = "csrf_token"
CSRF_HEADER_KEY = "X-CSRFToken"


def get_csrf_token() -> str:
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return str(token)


def verify_csrf() -> None:
    if request.method in SAFE_METHODS:
        return

    expected = session.get(CSRF_SESSION_KEY)
    if not expected:
        abort(400)

    supplied = request.headers.get(CSRF_HEADER_KEY) or request.form.get(CSRF_FORM_KEY)
    if not supplied and request.is_json:
        payload = request.get_json(silent=True) or {}
        supplied = payload.get(CSRF_FORM_KEY)

    if supplied != expected:
        abort(400)


def is_safe_redirect_target(target: str | None) -> bool:
    if not target:
        return False

    host_url = request.host_url
    ref = urlparse(host_url)
    test = urlparse(urljoin(host_url, target))
    return test.scheme in {"http", "https"} and ref.netloc == test.netloc


def safe_redirect_target(target: str | None, fallback: str) -> str:
    return target if is_safe_redirect_target(target) else fallback

