"""Starlette app: route the agent's Responses request through the fold logic.

Only ACTS when continuation is enabled and the agent did not itself declare a
`continue_thinking` tool (collision rule). Otherwise it is a pure passthrough,
so it is safe in front of all traffic.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Iterable
from urllib.parse import urlparse

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
from .payload_log import SQLitePayloadLogger
from .proxy import (
    fold_stream,
    observed_response_bytes,
    open_passthrough,
    open_round,
    read_upstream_error,
    upstream_request_id,
)
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


def _is_chatgpt_codex_responses_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    return host == "chatgpt.com" and path.endswith("/backend-api/codex/responses")


def _body_for_upstream(body: dict[str, Any], url: str) -> dict[str, Any]:
    if "previous_response_id" not in body or not _is_chatgpt_codex_responses_url(url):
        return body
    out = dict(body)
    out.pop("previous_response_id", None)
    return out


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


def _merge_previous_response_body(
    contexts: dict[str, list[Any]],
    current: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    prev_id = current.get("previous_response_id")
    if not isinstance(prev_id, str) or prev_id not in contexts:
        return current, None

    merged = dict(current)
    merged["input"] = contexts[prev_id] + list(current.get("input") or [])
    merged.pop("previous_response_id", None)
    return merged, prev_id


def _remember_response_context(
    contexts: dict[str, list[Any]],
    response: Any,
    request_input: list[Any],
    consumed_prev_id: str | None,
) -> None:
    if not isinstance(response, dict):
        return
    response_id = response.get("id")
    output = response.get("output")
    if not isinstance(response_id, str) or not isinstance(output, list):
        return

    contexts[response_id] = request_input + output
    if consumed_prev_id is not None and consumed_prev_id != response_id:
        contexts.pop(consumed_prev_id, None)

    while len(contexts) > 16:
        contexts.pop(next(iter(contexts)))


async def _passthrough(
    client: httpx.AsyncClient,
    cfg: Config,
    request: Request,
    raw: bytes,
    url: str,
    payload_logger: Any | None = None,
    trace_id: str = "",
):
    """Pure proxy: forward the raw request and stream the raw response back."""
    headers = build_upstream_headers(request.headers.items(), cfg)
    resp = await open_passthrough(
        client,
        url,
        raw,
        headers,
        payload_logger=payload_logger,
        transport="http",
        phase="passthrough",
        trace_id=trace_id,
    )

    async def body_iter():
        try:
            async for chunk in observed_response_bytes(resp):
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
    payload_logger = getattr(request.app.state, "payload_logger", None)
    trace_id = f"http_{uuid.uuid4().hex[:16]}"
    started = time.perf_counter()

    raw = await request.body()
    try:
        body: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    log.info(
        "event=downstream_request_start trace_id=%s transport=http path=%s "
        "model=%s input_items=%d body_bytes=%d",
        trace_id, request.url.path, body.get("model") or "-",
        len(body.get("input") or []), len(raw),
    )

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
        upstream_body = _body_for_upstream(body, url)
        upstream_raw = _json_body_bytes(upstream_body) if upstream_body is not body else raw
        return await _passthrough(
            client, cfg, request, upstream_raw, url, payload_logger, trace_id
        )

    log.info("fold start: trace_id=%s model=%s path=%s url=%s input_items=%d",
             trace_id, body.get("model"), request.url.path, url,
             len(body.get("input") or []))

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
    upstream_body = _body_for_upstream(body, url)
    payload = build_round_payload(
        upstream_body,
        input_items=list(upstream_body.get("input") or []),
        force_include_encrypted=cfg.stream.force_include_encrypted,
        drop_previous_response_id=False,  # round 1 uses the already sanitized body
    )

    # Open round 1 here so a non-2xx (e.g. bad auth) is mirrored with its real
    # status code rather than buried inside a 200 SSE stream.
    resp = await open_round(
        client,
        url,
        payload,
        headers,
        payload_logger=payload_logger,
        transport="http",
        phase="fold_round_1",
        trace_id=trace_id,
    )
    if resp.status_code >= 400:
        err = await read_upstream_error(resp)
        await resp.aclose()
        log.info(
            "event=downstream_request_end trace_id=%s transport=http outcome=upstream_error "
            "upstream_request_id=%s status=%s elapsed_ms=%.2f",
            trace_id, upstream_request_id(resp), resp.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return Response(
            err, status_code=resp.status_code, media_type=resp.headers.get("content-type")
        )

    return StreamingResponse(
        fold_stream(
            client,
            cfg,
            upstream_body,
            headers,
            resp,
            request.app.state.id_store,
            url=url,
            payload_logger=payload_logger,
            transport="http",
            trace_id=trace_id,
        ),
        media_type="text/event-stream",
    )


async def _receive_ws_messages(
    websocket: WebSocket,
    inbox: asyncio.Queue[dict[str, Any]],
) -> dict[str, Any]:
    """Receive downstream frames until the peer disconnects.

    This runs independently from upstream processing so a disconnect remains
    observable while ``httpx`` is still waiting for response headers.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return message
        await inbox.put(message)


async def _receive_ws_json(inbox: asyncio.Queue[dict[str, Any]]) -> Any:
    message = await inbox.get()
    if message.get("type") != "websocket.receive":
        raise RuntimeError(f"Unexpected ASGI WebSocket message: {message.get('type')}")

    text = message.get("text")
    if text is None:
        data = message.get("bytes")
        if data is None:
            raise RuntimeError("WebSocket receive message contained neither text nor bytes")
        text = data.decode("utf-8")
    return json.loads(text)


async def _close_active_ws_response(activity: dict[str, Any]) -> None:
    response = activity.pop("response", None)
    if response is None:
        return
    try:
        await response.aclose()
    except Exception as exc:
        log.warning(
            "event=upstream_response_close_exception trace_id=%s error_type=%s error=%r",
            activity.get("trace_id") or "-", type(exc).__name__, exc,
        )


async def handle_responses_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    connection_id = f"ws_{uuid.uuid4().hex[:16]}"
    activity: dict[str, Any] = {
        "request_no": 0,
        "trace_id": None,
        "request_started": None,
        "response": None,
    }
    inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    log.info(
        "event=downstream_ws_open connection_id=%s path=%s client=%s",
        connection_id, websocket.url.path, websocket.client,
    )

    receiver = asyncio.create_task(
        _receive_ws_messages(websocket, inbox),
        name=f"{connection_id}-receiver",
    )
    processor = asyncio.create_task(
        _process_responses_ws(websocket, inbox, connection_id, activity),
        name=f"{connection_id}-processor",
    )

    try:
        done, _ = await asyncio.wait(
            {receiver, processor}, return_when=asyncio.FIRST_COMPLETED
        )

        if receiver in done:
            disconnect = receiver.result()
            trace_id = activity.get("trace_id")
            request_started = activity.get("request_started")
            processor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await processor
            await _close_active_ws_response(activity)

            if trace_id:
                elapsed_ms = (
                    (time.perf_counter() - request_started) * 1000
                    if isinstance(request_started, (int, float)) else -1.0
                )
                log.warning(
                    "event=downstream_ws_disconnect_cancel trace_id=%s connection_id=%s "
                    "code=%s elapsed_ms=%.2f",
                    trace_id, connection_id, disconnect.get("code"), elapsed_ms,
                )
            else:
                log.info(
                    "event=downstream_ws_disconnect connection_id=%s requests=%d code=%s",
                    connection_id, activity.get("request_no", 0), disconnect.get("code"),
                )
            return

        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
        try:
            await processor
        except WebSocketDisconnect as exc:
            log.info(
                "event=downstream_ws_disconnect connection_id=%s requests=%d code=%s",
                connection_id, activity.get("request_no", 0), exc.code,
            )
    finally:
        for task in (receiver, processor):
            if not task.done():
                task.cancel()
        await asyncio.gather(receiver, processor, return_exceptions=True)
        await _close_active_ws_response(activity)


async def _process_responses_ws(
    websocket: WebSocket,
    inbox: asyncio.Queue[dict[str, Any]],
    connection_id: str,
    activity: dict[str, Any],
) -> None:
    cfg: Config = websocket.app.state.cfg
    client: httpx.AsyncClient = websocket.app.state.client
    payload_logger = getattr(websocket.app.state, "payload_logger", None)
    warmups: dict[str, dict[str, Any]] = {}
    response_contexts: dict[str, list[Any]] = {}
    request_no = 0

    while True:
        try:
            msg = await _receive_ws_json(inbox)
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

        request_no += 1
        activity["request_no"] = request_no
        trace_id = f"{connection_id}.{request_no}"
        request_started = time.perf_counter()

        body = {k: v for k, v in msg.items() if k != "type"}
        body.pop("background", None)
        warmup = _is_ws_warmup(body)
        consumed_prev_id: str | None = None
        if not warmup:
            body["stream"] = True
            prev_id = body.get("previous_response_id")
            if isinstance(prev_id, str) and prev_id in warmups:
                body = _merge_warmup_body(warmups.pop(prev_id), body)
            else:
                body, consumed_prev_id = _merge_previous_response_body(
                    response_contexts, body
                )

        log.info(
            "event=downstream_request_start trace_id=%s transport=ws connection_id=%s "
            "request_no=%d model=%s input_items=%d previous_response_id=%s",
            trace_id, connection_id, request_no, body.get("model") or "-",
            len(body.get("input") or []), "present" if body.get("previous_response_id") else "absent",
        )

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
        upstream_body = _body_for_upstream(body, url)
        request_input = list(upstream_body.get("input") or [])
        should_fold = (not warmup) and _should_fold(cfg, upstream_body)

        if warmup:
            log.info("warmup(ws): model=%s path=%s url=%s input_items=%d local_state=1",
                     body.get("model"), websocket.url.path, url, len(body.get("input") or []))
            resp_obj = _local_warmup_response(body)
            warmups[resp_obj["id"]] = dict(body)
            for ev in _warmup_events_from_response(resp_obj):
                await websocket.send_json(ev)
            continue
        elif should_fold:
            activity["trace_id"] = trace_id
            activity["request_started"] = request_started
            log.info("fold start(ws): trace_id=%s model=%s path=%s url=%s input_items=%d",
                     trace_id, upstream_body.get("model"), websocket.url.path, url,
                     len(upstream_body.get("input") or []))
            payload = build_round_payload(
                upstream_body,
                input_items=list(upstream_body.get("input") or []),
                force_include_encrypted=cfg.stream.force_include_encrypted,
                drop_previous_response_id=False,
            )
            resp = await open_round(
                client,
                url,
                payload,
                headers,
                payload_logger=payload_logger,
                transport="ws",
                phase="fold_round_1",
                trace_id=trace_id,
            )
        else:
            activity["trace_id"] = trace_id
            activity["request_started"] = request_started
            why = _passthrough_reason(cfg, body)
            log.info("passthrough(ws) (%s): model=%s path=%s url=%s",
                     why, body.get("model"), websocket.url.path, url)
            resp = await open_passthrough(
                client,
                url,
                _json_body_bytes(upstream_body),
                headers,
                payload_logger=payload_logger,
                transport="ws",
                phase="passthrough",
                trace_id=trace_id,
            )

        activity["response"] = resp

        if resp.status_code >= 400:
            err = await read_upstream_error(resp)
            await resp.aclose()
            activity["response"] = None
            log.warning(
                "event=downstream_error_send trace_id=%s connection_id=%s "
                "upstream_request_id=%s status=%s elapsed_ms=%.2f",
                trace_id, connection_id, upstream_request_id(resp), resp.status_code,
                (time.perf_counter() - request_started) * 1000,
            )
            await websocket.send_json(_upstream_error_event(err, resp.status_code))
            activity["trace_id"] = None
            activity["request_started"] = None
            continue

        try:
            if should_fold:
                event_iter = _iter_sse_events(
                    fold_stream(
                        client,
                        cfg,
                        upstream_body,
                        headers,
                        resp,
                        websocket.app.state.id_store,
                        url=url,
                        payload_logger=payload_logger,
                        transport="ws",
                        trace_id=trace_id,
                    )
                )
                async for ev in event_iter:
                    if ev.get("type") in (
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                    ):
                        _remember_response_context(
                            response_contexts,
                            ev.get("response"),
                            request_input,
                            consumed_prev_id,
                        )
                    await websocket.send_json(ev)
            else:
                try:
                    async for ev in _iter_sse_events(observed_response_bytes(resp)):
                        if ev.get("type") in (
                            "response.completed",
                            "response.failed",
                            "response.incomplete",
                        ):
                            _remember_response_context(
                                response_contexts,
                                ev.get("response"),
                                request_input,
                                consumed_prev_id,
                            )
                        await websocket.send_json(ev)
                finally:
                    await resp.aclose()
                    activity["response"] = None
        except WebSocketDisconnect:
            log.warning(
                "event=downstream_ws_disconnect_active trace_id=%s connection_id=%s "
                "elapsed_ms=%.2f",
                trace_id, connection_id,
                (time.perf_counter() - request_started) * 1000,
            )
            return
        else:
            log.info(
                "event=downstream_request_end trace_id=%s transport=ws outcome=stream_end "
                "elapsed_ms=%.2f",
                trace_id, (time.perf_counter() - request_started) * 1000,
            )
        activity["response"] = None
        activity["trace_id"] = None
        activity["request_started"] = None


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
        app.state.payload_logger = SQLitePayloadLogger.from_config(cfg)
        if app.state.payload_logger is not None:
            app.state.payload_logger.initialize()
        else:
            log.info("outbound payload sqlite logging disabled")
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
