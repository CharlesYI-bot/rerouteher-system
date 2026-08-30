"""ASGI message preservation and privacy; no HTTP server needed."""

import logging
from types import SimpleNamespace

import pytest

from app.core.logging import RequestLoggingMiddleware


@pytest.mark.asyncio
async def test_chunked_request_above_4000_bytes_is_unchanged_and_private(caplog):
    chunks = [
        {"type": "http.request", "body": b"private-resume-" * 400, "more_body": True},
        {"type": "http.request", "body": b"tail-secret", "more_body": False},
        {"type": "http.disconnect"},
    ]
    remaining = iter(chunks)
    received, sent = [], []

    async def receive():
        return next(remaining)

    async def send(message):
        sent.append(message)

    async def app(scope, recv, out):
        for _ in chunks:
            received.append(await recv())
        await out({"type": "http.response.start", "status": 200, "headers": []})
        await out({"type": "http.response.body", "body": b"response-secret", "more_body": True})
        await out({"type": "http.response.body", "body": b"end", "more_body": False})

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/private-user",
        "query_string": b"email=private@example.com",
        "route": SimpleNamespace(path="/api/cv/parse"),
    }
    with caplog.at_level(logging.INFO):
        await RequestLoggingMiddleware(app)(scope, receive, send)
    assert received == chunks
    assert sent[-2]["body"] == b"response-secret"
    assert sent[-2]["more_body"] is True
    assert "POST /api/cv/parse -> 200" in caplog.text
    for private in [
        "private-resume",
        "tail-secret",
        "response-secret",
        "private-user",
        "private@example.com",
    ]:
        assert private not in caplog.text


@pytest.mark.asyncio
async def test_exception_payload_not_logged_and_exception_propagates(caplog):
    async def app(scope, receive, send):
        raise ValueError("private-resume-secret")

    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        await RequestLoggingMiddleware(app)({"type": "http", "method": "POST"}, None, None)
    assert "private-resume-secret" not in caplog.text
    assert "500" in caplog.text


@pytest.mark.asyncio
async def test_non_http_scope_is_untouched():
    seen = []

    async def app(scope, receive, send):
        seen.append((scope, receive, send))

    scope = {"type": "lifespan"}
    await RequestLoggingMiddleware(app)(scope, None, None)
    assert seen == [(scope, None, None)]
