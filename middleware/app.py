"""Starlette app: route the agent's Responses request through the fold logic.

Only ACTS when continuation is enabled and the agent did not itself declare a
`continue_thinking` tool (collision rule). Otherwise it is a pure passthrough,
so it is safe in front of all traffic.
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Iterable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .codex import (
    build_round_payload,
    declares_continue_tool,
    reasoning_enabled,
    repair_followup_input,
)
from .config import Config
from .creds import build_upstream_headers, would_inject_authorization
from .proxy import fold_stream, open_passthrough, open_round
from .sse import DONE, incremental_sse
from .store import IdStore

log = logging.getLogger("middleware.app")


def _header_base(request: Request) -> str | None:
    """The non-blank Responses-API-Base header value, or None (case-insensitive)."""
    v = request.headers.get("responses-api-base")
    v = v.strip() if v else ""
    return v or None


def _join_responses(base: str) -> str:
    """Build the Responses endpoint from a base URL (OpenAI base_url convention:
    `<base>/responses`). Lenient: if the value already ends in `/responses`
    (a full endpoint was passed), use it as-is."""
    base = base.rstrip("/")
    return base if base.endswith("/responses") else base + "/responses"


def _resolve_upstream_url(cfg: Config, request: Request) -> str | None:
    """Target URL for this request.

    - "fixed": always the configured URL (header ignored).
    - "header": the Responses-API-Base header (case-insensitive) is treated as a
      base URL and `/responses` is appended; overrides the configured URL when
      present, else the configured URL.
    - "header_required": the header MUST be present; returns None when it is
      absent/blank so the caller can reject the request (400).

    The header is stripped before forwarding upstream (build_upstream_headers).
    """
    if cfg.upstream.mode in ("header", "header_required"):
        base = _header_base(request)
        if base:
            return _join_responses(base)
        if cfg.upstream.mode == "header_required":
            return None
    return cfg.upstream.url


def _url_is_from_header(cfg: Config, request: Request) -> bool:
    return cfg.upstream.mode in ("header", "header_required") and _header_base(request) is not None


def _collision(cfg: Config, body: dict[str, Any]) -> bool:
    return (
        cfg.cont.method == "tool_pair"
        and declares_continue_tool(body, cfg.cont.continue_tool_name)
    )


def _should_fold(cfg: Config, body: dict[str, Any]) -> bool:
    return (
        cfg.cont.enabled
        and bool(body.get("stream"))
        and reasoning_enabled(body)
        and not _collision(cfg, body)
    )


def _passthrough_reason(cfg: Config, body: dict[str, Any]) -> str:
    if not cfg.cont.enabled:
        return "disabled"
    if not body.get("stream"):
        return "non-stream"
    if not reasoning_enabled(body):
        return "non-reasoning"
    return "declares-continue_thinking"


def _header_override_error(
    cfg: Config,
    request: Request | WebSocket,
    *,
    model: Any,
) -> str | None:
    if _url_is_from_header(cfg, request) and would_inject_authorization(
        cfg, agent_has_authorization=request.headers.get("authorization") is not None
    ):
        log.warning("blocked: Responses-API-Base override without own auth (model=%s)", model)
        return (
            "When overriding the upstream base (Responses-API-Base), the request must provide "
            "its own Authorization; the proxy will not send its configured credentials to an "
            "externally supplied URL."
        )
    return None


async def _iter_sse_events(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    async for ev in incremental_sse(byte_iter):
        if ev is DONE or not isinstance(ev, dict):
            continue
        yield ev


def _json_body_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _ensure_header(headers: dict[str, str], name: str, value: str) -> None:
    if not any(k.lower() == name.lower() for k in headers):
        headers[name] = value


def _ws_upstream_headers(headers: Iterable[tuple[str, str]], cfg: Config) -> dict[str, str]:
    out = build_upstream_headers(headers, cfg)
    _ensure_header(out, "Content-Type", "application/json")
    return out


def _ws_error_event(
    message: str,
    *,
    status: int = 400,
    code: str = "invalid_request_error",
    param: str | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"type": "invalid_request_error", "code": code, "message": message}
    if param is not None:
        err["param"] = param
    return {"type": "error", "status": status, "error": err}


def _upstream_error_event(raw: bytes, status: int) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")[:1000] or "Upstream error"
        return _ws_error_event(text, status=status, code="upstream_error")

    if isinstance(parsed, dict):
        if parsed.get("type") == "error":
            return parsed
        return {"type": "error", "status": status, **parsed}

    return _ws_error_event(str(parsed), status=status, code="upstream_error")


def _is_ws_warmup(body: dict[str, Any]) -> bool:
    return body.get("generate") is False


async def _open_json_request(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    req = client.build_request(
        "POST",
        url,
        content=_json_body_bytes(body),
        headers=headers,
        timeout=None,
    )
    return await client.send(req, stream=False)


def _terminal_type_for_response(resp: dict[str, Any]) -> str:
    status = resp.get("status")
    if status == "failed":
        return "response.failed"
    if status == "incomplete":
        return "response.incomplete"
    return "response.completed"


def _warmup_events_from_response(resp: dict[str, Any]) -> list[dict[str, Any]]:
    response = dict(resp)
    response.setdefault("status", "completed")
    return [
        {"type": "response.created", "response": response, "sequence_number": 0},
        {"type": _terminal_type_for_response(response), "response": response, "sequence_number": 1},
    ]


def _local_warmup_response(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"resp_warmup_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model"),
        "output": [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _merge_warmup_body(warmup: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(warmup)
    warmup_input = list(warmup.get("input") or [])
    current_input = list(current.get("input") or [])

    merged.pop("generate", None)
    merged.pop("input", None)
    merged.pop("previous_response_id", None)

    for k, v in current.items():
        if k not in ("input", "previous_response_id", "generate"):
            merged[k] = v

    if warmup_input or current_input:
        merged["input"] = warmup_input + current_input
    return merged


async def _passthrough(
    client: httpx.AsyncClient, cfg: Config, request: Request, raw: bytes, url: str
):
    """Pure proxy: forward the raw request and stream the raw response back."""
    headers = build_upstream_headers(request.headers.items(), cfg)
    resp = await open_passthrough(client, url, raw, headers)

    async def body_iter():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


async def handle_responses(request: Request) -> Response:
    cfg: Config = request.app.state.cfg
    client: httpx.AsyncClient = request.app.state.client

    raw = await request.body()
    try:
        body: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    url = _resolve_upstream_url(cfg, request)
    if url is None:
        return JSONResponse(
            {"error": "Responses-API-Base header is required (upstream mode=header_required)"},
            status_code=400,
        )

    if err := _header_override_error(cfg, request, model=body.get("model")):
        return JSONResponse(
            {"error": err},
            status_code=400,
        )

    should_fold = _should_fold(cfg, body)
    if not should_fold:
        why = _passthrough_reason(cfg, body)
        log.info("passthrough (%s): model=%s path=%s url=%s",
                 why, body.get("model"), request.url.path, url)
        return await _passthrough(client, cfg, request, raw, url)

    log.info("fold start: model=%s path=%s url=%s input_items=%d",
             body.get("model"), request.url.path, url, len(body.get("input") or []))

    # repair_followup="stateful": re-insert tool_pair continue pairs after recorded
    # ids (tool_pair only — commentary preserves cross-turn structure via forward_marker).
    if cfg.cont.repair_followup == "stateful" and cfg.cont.method == "tool_pair":
        body = {
            **body,
            "input": repair_followup_input(
                list(body.get("input") or []),
                request.app.state.id_store,
                tool_name=cfg.cont.continue_tool_name,
                output_text=cfg.cont.continue_output_text,
            ),
        }

    headers = build_upstream_headers(request.headers.items(), cfg)
    payload = build_round_payload(
        body,
        input_items=list(body.get("input") or []),
        force_include_encrypted=cfg.stream.force_include_encrypted,
        drop_previous_response_id=False,  # round 1 passes it through
    )

    # Open round 1 here so a non-2xx (e.g. bad auth) is mirrored with its real
    # status code rather than buried inside a 200 SSE stream.
    resp = await open_round(client, url, payload, headers)
    if resp.status_code >= 400:
        err = await resp.aread()
        await resp.aclose()
        return Response(
            err, status_code=resp.status_code, media_type=resp.headers.get("content-type")
        )

    return StreamingResponse(
        fold_stream(client, cfg, body, headers, resp, request.app.state.id_store, url=url),
        media_type="text/event-stream",
    )


async def handle_responses_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    cfg: Config = websocket.app.state.cfg
    client: httpx.AsyncClient = websocket.app.state.client
    warmups: dict[str, dict[str, Any]] = {}

    while True:
        try:
            msg = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except Exception:
            await websocket.send_json(
                _ws_error_event("Expected a JSON response.create event.", param="type")
            )
            continue

        if not isinstance(msg, dict):
            await websocket.send_json(
                _ws_error_event("Expected a JSON object.", param="type")
            )
            continue
        if msg.get("type") != "response.create":
            await websocket.send_json(
                _ws_error_event(
                    "Unsupported event type. Send response.create.",
                    param="type",
                )
            )
            continue

        body = {k: v for k, v in msg.items() if k != "type"}
        body.pop("background", None)
        warmup = _is_ws_warmup(body)
        if not warmup:
            body["stream"] = True
            prev_id = body.get("previous_response_id")
            if isinstance(prev_id, str) and prev_id in warmups:
                body = _merge_warmup_body(warmups.pop(prev_id), body)

        url = _resolve_upstream_url(cfg, websocket)
        if url is None:
            await websocket.send_json(
                _ws_error_event(
                    "Responses-API-Base header is required (upstream mode=header_required)"
                )
            )
            continue
        if err := _header_override_error(cfg, websocket, model=body.get("model")):
            await websocket.send_json(_ws_error_event(err))
            continue

        headers = _ws_upstream_headers(websocket.headers.items(), cfg)
        should_fold = (not warmup) and _should_fold(cfg, body)

        if warmup:
            log.info("warmup(ws): model=%s path=%s url=%s input_items=%d local_state=1",
                     body.get("model"), websocket.url.path, url, len(body.get("input") or []))
            resp_obj = _local_warmup_response(body)
            warmups[resp_obj["id"]] = dict(body)
            for ev in _warmup_events_from_response(resp_obj):
                await websocket.send_json(ev)
            continue
        elif should_fold:
            log.info("fold start(ws): model=%s path=%s url=%s input_items=%d",
                     body.get("model"), websocket.url.path, url, len(body.get("input") or []))
            payload = build_round_payload(
                body,
                input_items=list(body.get("input") or []),
                force_include_encrypted=cfg.stream.force_include_encrypted,
                drop_previous_response_id=False,
            )
            resp = await open_round(client, url, payload, headers)
        else:
            why = _passthrough_reason(cfg, body)
            log.info("passthrough(ws) (%s): model=%s path=%s url=%s",
                     why, body.get("model"), websocket.url.path, url)
            resp = await open_passthrough(client, url, _json_body_bytes(body), headers)

        if resp.status_code >= 400:
            err = await resp.aread()
            await resp.aclose()
            await websocket.send_json(_upstream_error_event(err, resp.status_code))
            continue

        try:
            if should_fold:
                event_iter = _iter_sse_events(
                    fold_stream(
                        client,
                        cfg,
                        body,
                        headers,
                        resp,
                        websocket.app.state.id_store,
                        url=url,
                    )
                )
                async for ev in event_iter:
                    await websocket.send_json(ev)
            else:
                try:
                    async for ev in _iter_sse_events(resp.aiter_bytes()):
                        await websocket.send_json(ev)
                finally:
                    await resp.aclose()
        except WebSocketDisconnect:
            return


def _make_client() -> httpx.AsyncClient:
    """A client that does NOT invent a User-Agent or Accept of its own; those
    are forwarded from the agent or omitted. httpx still manages Host /
    Content-Length / Accept-Encoding / Connection (plan-allowed)."""
    client = httpx.AsyncClient(timeout=None)
    for h in ("user-agent", "accept"):
        if h in client.headers:
            del client.headers[h]
    return client


def create_app(cfg: Config) -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.cfg = cfg
        app.state.client = _make_client()
        app.state.id_store = IdStore()
        try:
            yield
        finally:
            await app.state.client.aclose()

    routes = [
        Route(path, handle_responses, methods=["POST"]) for path in cfg.server.listen_paths
    ]
    routes.extend(
        WebSocketRoute(path, handle_responses_ws) for path in cfg.server.listen_paths
    )
    return Starlette(routes=routes, lifespan=lifespan)
