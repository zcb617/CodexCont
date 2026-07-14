"""Native WebSocket transport for the upstream Responses endpoint.

One ``UpstreamWebSocketSession`` belongs to one downstream WebSocket.  It
serializes ``response.create`` rounds on the same upstream connection and
adapts each round to the small response interface used by the fold state
machine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import connect as websockets_connect

from .sse import serialize_event

log = logging.getLogger("middleware.ws_upstream")

_DIAG_ATTR = "_codexcont_upstream_diagnostics"
_WEBSOCKET_BETA = "responses_websockets=2026-02-06"
_TERMINAL_EVENTS = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "error",
}


def to_websocket_url(url: str) -> str:
    """Convert an HTTP Responses URL to the corresponding WebSocket URL."""
    parsed = urlsplit(url)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(
        parsed.scheme.lower()
    )
    if scheme is None or not parsed.netloc:
        raise ValueError(f"Unsupported upstream WebSocket URL: {url!r}")
    # URL fragments are client-side only and aren't valid in a WebSocket URI.
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))


def response_create_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the wire event accepted by Responses WebSocket mode."""
    event = dict(payload)
    event["type"] = "response.create"
    event.pop("stream", None)
    event.pop("background", None)
    return event


def _pop_headers(headers: dict[str, str], name: str) -> list[str]:
    values: list[str] = []
    for key in list(headers):
        if key.lower() == name.lower():
            values.append(headers.pop(key))
    return values


def _websocket_headers(headers: dict[str, str]) -> tuple[dict[str, str], str | None]:
    out = dict(headers)
    user_agents = _pop_headers(out, "user-agent")
    user_agent = user_agents[-1] if user_agents else None
    beta_values = [
        item.strip()
        for value in _pop_headers(out, "openai-beta")
        for item in value.split(",")
        if item.strip()
    ]
    if _WEBSOCKET_BETA not in beta_values:
        beta_values.append(_WEBSOCKET_BETA)
    out["OpenAI-Beta"] = ", ".join(beta_values)
    return out, user_agent


def _error_status(event: dict[str, Any]) -> int:
    candidates = [event.get("status")]
    error = event.get("error")
    if isinstance(error, dict):
        candidates.append(error.get("status"))
    for value in candidates:
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    return 502


def _exception_status(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int) and 400 <= value <= 599:
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and 400 <= value <= 599:
        return value
    return None


class UpstreamWebSocketError(ConnectionError):
    """A handshake, send, receive, or protocol failure on the upstream WS."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def upstream_websocket_error_status(exc: BaseException) -> int:
    """Return an upstream HTTP handshake status when one exists, else 502."""
    return _exception_status(exc) or 502


class UpstreamWebSocketResponse:
    """One response.create round, exposed as both events and SSE bytes."""

    def __init__(
        self,
        session: "UpstreamWebSocketSession",
        first_event: dict[str, Any],
        *,
        diagnostics: dict[str, Any],
    ) -> None:
        self._session = session
        self._first_event: dict[str, Any] | None = first_event
        self._consumed = False
        self._closed = False
        self._terminal_received = first_event.get("type") in _TERMINAL_EVENTS
        self.status_code = (
            _error_status(first_event) if first_event.get("type") == "error" else 200
        )
        self.headers = session.response_headers
        setattr(self, _DIAG_ATTR, diagnostics)

    async def _events(self) -> AsyncIterator[dict[str, Any]]:
        if self._consumed:
            raise RuntimeError("Upstream WebSocket response was already consumed")
        self._consumed = True

        event = self._first_event
        self._first_event = None
        while event is not None:
            terminal = event.get("type") in _TERMINAL_EVENTS
            if terminal:
                self._terminal_received = True
                self._session._release(self)
            yield event
            if terminal:
                return
            event = await self._session._receive_event(self)

    async def aiter_events(self) -> AsyncIterator[dict[str, Any]]:
        async for event in self._events():
            yield event

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        async for event in self._events():
            yield serialize_event(event)

    async def aread(self) -> bytes:
        if self._first_event is not None and self._first_event.get("type") == "error":
            return json.dumps(
                self._first_event, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")

        events = [event async for event in self._events()]
        return json.dumps(events, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._terminal_received:
            self._session._release(self)
            return
        await self._session._abort_round(self)


class UpstreamWebSocketSession:
    """A persistent upstream connection supporting sequential response rounds."""

    def __init__(
        self,
        connection: Any,
        *,
        url: str,
        connection_id: str,
        payload_logger: Any | None = None,
    ) -> None:
        self._connection = connection
        self.url = url
        self.connection_id = connection_id
        self.payload_logger = payload_logger
        self._active: UpstreamWebSocketResponse | None = None
        self._closed = False
        response = getattr(connection, "response", None)
        self.response_headers = getattr(response, "headers", {}) or {}

    async def open_round(
        self,
        payload: dict[str, Any],
        *,
        phase: str,
        trace_id: str,
    ) -> UpstreamWebSocketResponse:
        if self._closed or self._connection is None:
            raise UpstreamWebSocketError("Upstream WebSocket connection is closed")
        if self._active is not None:
            raise RuntimeError("Only one upstream WebSocket response may be active")

        request_id = f"upws_{uuid.uuid4().hex[:16]}"
        started = time.perf_counter()
        event = response_create_payload(payload)
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        log.info(
            "event=upstream_request_start request_id=%s trace_id=%s transport=ws "
            "phase=%s method=response.create url=%s model=%s input_items=%d body_bytes=%d",
            request_id,
            trace_id or "-",
            phase or "-",
            self.url,
            event.get("model") or "-",
            len(event.get("input") or []),
            len(encoded.encode("utf-8")),
        )

        try:
            await self._connection.send(encoded)
            first_event = await self._receive_json()
        except asyncio.CancelledError:
            log.warning(
                "event=upstream_request_cancelled request_id=%s trace_id=%s phase=%s "
                "elapsed_ms=%.2f",
                request_id,
                trace_id or "-",
                phase or "-",
                (time.perf_counter() - started) * 1000,
            )
            raise
        except Exception as exc:
            status = _exception_status(exc)
            log.warning(
                "event=upstream_request_exception request_id=%s trace_id=%s phase=%s "
                "elapsed_ms=%.2f error_type=%s status=%s error=%r",
                request_id,
                trace_id or "-",
                phase or "-",
                (time.perf_counter() - started) * 1000,
                type(exc).__name__,
                status or "-",
                exc,
            )
            await self._drop_connection()
            if isinstance(exc, UpstreamWebSocketError):
                raise
            raise UpstreamWebSocketError(
                "Upstream WebSocket request failed", status_code=status
            ) from exc

        status_code = _error_status(first_event) if first_event.get("type") == "error" else 200
        diagnostics = {
            "request_id": request_id,
            "trace_id": trace_id or "-",
            "transport": "ws",
            "phase": phase or "-",
            "started": started,
        }
        response = UpstreamWebSocketResponse(
            self, first_event, diagnostics=diagnostics
        )
        self._active = response
        log.info(
            "event=upstream_response_first_event request_id=%s trace_id=%s phase=%s "
            "status=%s event_type=%s elapsed_ms=%.2f",
            request_id,
            trace_id or "-",
            phase or "-",
            status_code,
            first_event.get("type") or "-",
            (time.perf_counter() - started) * 1000,
        )
        if self.payload_logger is not None:
            self.payload_logger.record(
                transport="ws",
                phase=phase,
                url=self.url,
                payload=event,
                status_code=status_code,
            )
        return response

    async def _receive_json(self) -> dict[str, Any]:
        if self._connection is None:
            raise UpstreamWebSocketError("Upstream WebSocket connection is closed")
        raw = await self._connection.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise UpstreamWebSocketError(
                "Upstream WebSocket returned an invalid JSON event"
            ) from exc
        if not isinstance(event, dict):
            raise UpstreamWebSocketError(
                "Upstream WebSocket returned a non-object JSON event"
            )
        return event

    async def _receive_event(
        self, response: UpstreamWebSocketResponse
    ) -> dict[str, Any]:
        if self._active is not response:
            raise UpstreamWebSocketError("Upstream WebSocket response is no longer active")
        try:
            return await self._receive_json()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._drop_connection()
            if isinstance(exc, UpstreamWebSocketError):
                raise
            raise UpstreamWebSocketError("Upstream WebSocket receive failed") from exc

    def _release(self, response: UpstreamWebSocketResponse) -> None:
        if self._active is response:
            self._active = None

    async def _abort_round(self, response: UpstreamWebSocketResponse) -> None:
        self._release(response)
        await self._drop_connection()

    async def _drop_connection(self) -> None:
        connection = self._connection
        self._connection = None
        self._active = None
        if connection is not None:
            with suppress(Exception):
                await connection.close()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._drop_connection()


WebSocketConnect = Callable[..., Awaitable[Any]]


async def connect_upstream_websocket(
    *,
    url: str,
    headers: dict[str, str],
    payload_logger: Any | None,
    connection_id: str,
    connector: WebSocketConnect = websockets_connect,
) -> UpstreamWebSocketSession:
    """Open the native upstream WS corresponding to an HTTP Responses URL."""
    ws_url = to_websocket_url(url)
    additional_headers, user_agent = _websocket_headers(headers)
    started = time.perf_counter()
    log.info(
        "event=upstream_ws_connect_start connection_id=%s url=%s",
        connection_id,
        ws_url,
    )
    try:
        connection = await connector(
            ws_url,
            additional_headers=additional_headers,
            user_agent_header=user_agent,
            open_timeout=None,
            close_timeout=1,
            max_size=None,
        )
    except asyncio.CancelledError:
        log.warning(
            "event=upstream_ws_connect_cancelled connection_id=%s url=%s elapsed_ms=%.2f",
            connection_id,
            ws_url,
            (time.perf_counter() - started) * 1000,
        )
        raise
    except Exception as exc:
        status = _exception_status(exc)
        log.warning(
            "event=upstream_ws_connect_exception connection_id=%s url=%s "
            "elapsed_ms=%.2f error_type=%s status=%s error=%r",
            connection_id,
            ws_url,
            (time.perf_counter() - started) * 1000,
            type(exc).__name__,
            status or "-",
            exc,
        )
        raise UpstreamWebSocketError(
            "Upstream WebSocket handshake failed", status_code=status
        ) from exc

    handshake = getattr(getattr(connection, "response", None), "status_code", 101)
    log.info(
        "event=upstream_ws_connect_end connection_id=%s url=%s status=%s elapsed_ms=%.2f",
        connection_id,
        ws_url,
        handshake,
        (time.perf_counter() - started) * 1000,
    )
    return UpstreamWebSocketSession(
        connection,
        url=ws_url,
        connection_id=connection_id,
        payload_logger=payload_logger,
    )
