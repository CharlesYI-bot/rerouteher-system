"""Metadata-only access logs. Never buffer/replay CV requests or log their bodies."""

from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("rerouteher.request")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level)


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            # Original callable: preserve every chunk, backpressure and disconnect.
            await self.app(scope, receive, send_wrapper)
        finally:
            # Arbitrary paths, query strings, headers and exception text may contain
            # PII too. Only log the route template; do not log exception payloads.
            route = getattr(scope.get("route"), "path", "<unmatched>")
            logger.info(
                "%s %s -> %d (%.1f ms)",
                scope["method"],
                route,
                status,
                (time.perf_counter() - start) * 1000,
            )
