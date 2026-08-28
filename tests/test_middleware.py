#!/usr/bin/env python3
"""Offline tests for the continue_thinking middleware.

Run: .venv/Scripts/python.exe tests/test_middleware.py
No pytest dependency — a tiny runner prints PASS/FAIL per check.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from starlette.datastructures import Headers
from starlette.testclient import TestClient

from middleware.app import (
    create_app,
    handle_responses_ws,
    _make_client,
    _resolve_upstream_url,
    _resolve_models_upstream_url,
    _url_is_from_header,
)
from middleware.codex import (
    continue_call_id,
    is_truncation_pattern,
    reasoning_enabled,
    repair_followup_input,
    should_continue,
    tier_n,
)
from middleware.config import load_config
from middleware.creds import build_upstream_headers, would_inject_authorization
from middleware import proxy as proxy_module
from middleware.proxy import fold_stream, open_round, read_upstream_error
from middleware.sse import DONE, incremental_sse
from middleware.store import IdStore
from middleware.ws_upstream import (
    UpstreamWebSocketError,
    connect_upstream_websocket,
    response_create_payload,
    to_websocket_url,
)


# --- helpers ----------------------------------------------------------------


def make_sse(events: list[dict]) -> bytes:
    out = b""
    for ev in events:
        out += f"event: {ev['type']}\r\n".encode()
        out += b"data: " + json.dumps(ev).encode() + b"\r\n\r\n"
    return out


async def _aiter_once(data: bytes):
    yield data


async def parse_events(data: bytes) -> list:
    evs = []
    async for e in incremental_sse(_aiter_once(data)):
        evs.append(e)
    return evs


class FakeResp:
    def __init__(
        self,
        data: bytes,
        status: int = 200,
        chunk: int = 4096,
        headers: dict[str, str] | None = None,
    ):
        # Raw upstream bytes returned by the fake response.
        self._data = data
        # HTTP status code exposed to the middleware.
        self.status_code = status
        # Headers supplied by the test case to exercise transparent forwarding.
        self.headers: dict[str, str] = dict(headers or {})
        # Maximum chunk size used by the fake streaming iterator.
        self._chunk = chunk

    async def aiter_bytes(self):
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i : i + self._chunk]

    async def aiter_events(self):
        async for event in incremental_sse(self.aiter_bytes()):
            if event is not DONE and isinstance(event, dict):
                yield event

    async def aread(self) -> bytes:
        return self._data

    async def aclose(self) -> None:
        pass


class FakeClient:
    """Returns the queued responses on successive send() calls; records the JSON
    body of each build_request (the per-continuation-round upstream payload)."""

    def __init__(self, responses: list[FakeResp]):
        # Queue of responses consumed by successive upstream sends.
        self._responses = list(responses)
        # Index of the next queued response.
        self._i = 0
        # JSON request payloads used by existing Responses continuation tests.
        self.payloads: list[dict] = []
        # Complete request metadata used by Models proxy assertions.
        self.requests: list[dict] = []

    def build_request(self, *a, **k):
        self.requests.append({
            "method": a[0] if a else None,
            "url": str(a[1]) if len(a) > 1 else None,
            "headers": dict(k.get("headers") or {}),
            "timeout": k.get("timeout"),
            "content": k.get("content"),
        })
        content = k.get("content")
        if content is not None:
            try:
                self.payloads.append(json.loads(content))
            except (json.JSONDecodeError, TypeError):
                pass
        return ("req", a, k)

    async def send(self, req, stream=True):
        r = self._responses[self._i]
        self._i += 1
        return r

    async def aclose(self) -> None:
        pass


class FakeWsSession:
    def __init__(self, owner, *, url, payload_logger):
        self.owner = owner
        self.url = to_websocket_url(url)
        self.payload_logger = payload_logger
        self.closed = False

    async def open_round(self, payload, *, phase, trace_id):
        wire_payload = response_create_payload(payload)
        self.owner.payloads.append(wire_payload)
        self.owner.phases.append(phase)
        response = self.owner.responses[self.owner.response_index]
        self.owner.response_index += 1
        if self.payload_logger is not None:
            self.payload_logger.record(
                transport="ws",
                phase=phase,
                url=self.url,
                payload=wire_payload,
                status_code=response.status_code,
            )
        return response

    async def aclose(self):
        self.closed = True


class FakeWsConnector:
    """Creates one fake upstream session per downstream WS connection."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.response_index = 0
        self.payloads: list[dict] = []
        self.phases: list[str] = []
        self.calls: list[dict] = []
        self.sessions: list[FakeWsSession] = []

    async def __call__(
        self, *, url, headers, payload_logger, connection_id, timeout_cfg=None
    ):
        self.calls.append({
            "url": url,
            "headers": dict(headers),
            "connection_id": connection_id,
            "timeout_cfg": timeout_cfg,
        })
        session = FakeWsSession(self, url=url, payload_logger=payload_logger)
        self.sessions.append(session)
        return session


class BlockingWsSession:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()

    async def open_round(self, payload, *, phase, trace_id):
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def aclose(self):
        self.closed.set()


class BlockingWsConnector:
    def __init__(self):
        self.session = BlockingWsSession()

    async def __call__(self, **kwargs):
        return self.session


class FakeDownstreamWebSocket:
    def __init__(self, app):
        self.app = app
        self.headers = Headers({"authorization": "Bearer test"})
        self.url = SimpleNamespace(path="/v1/responses")
        self.client = ("test-client", 1234)
        self.incoming: asyncio.Queue[dict] = asyncio.Queue()
        self.sent: list[dict] = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, data):
        self.sent.append(data)

    async def send_response_create(self):
        await self.incoming.put({
            "type": "websocket.receive",
            "text": json.dumps({
                "type": "response.create",
                "model": "gpt-5.5",
                "input": [{"role": "user", "content": "wait forever"}],
            }),
        })

    async def disconnect(self):
        await self.incoming.put({"type": "websocket.disconnect", "code": 1000})


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []
        # 捕获的日志记录对象，用于断言业务日志级别。
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """同时保留格式化消息与原始记录，供日志行为断言使用。"""
        self.records.append(record)
        self.messages.append(record.getMessage())


async def run_fold(cfg, base_body, first_resp, later_resps) -> list:
    client = FakeClient(later_resps)
    out = b""
    async for chunk in fold_stream(client, cfg, base_body, {}, first_resp):
        out += chunk
    return await parse_events(out)


async def run_fold_capture(cfg, base_body, first_resp, client) -> list:
    """Like run_fold but uses a caller-supplied client (to inspect client.payloads)."""
    out = b""
    async for chunk in fold_stream(client, cfg, base_body, {}, first_resp):
        out += chunk
    return await parse_events(out)


# --- test registry ----------------------------------------------------------

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


# --- 1. truncation math -----------------------------------------------------


def test_truncation_math():
    for n, tok in enumerate([516, 1034, 1552, 2070, 2588], start=1):
        check(f"is_truncation({tok})", is_truncation_pattern(tok))
        check(f"tier_n({tok})=={n}", tier_n(tok) == n, str(tier_n(tok)))
    for bad in (515, 517, 0, None):
        check(f"not is_truncation({bad})", not is_truncation_pattern(bad))
    # window
    check("should_continue 516 default", should_continue(516, min_n=1, max_n=0))
    check("should_continue 2588 max_n=3 blocked", not should_continue(2588, min_n=1, max_n=3))
    check("should_continue 516 min_n=2 blocked", not should_continue(516, min_n=2, max_n=0))
    check("should_continue None", not should_continue(None, min_n=1, max_n=0))


# --- 2. SSE framing robustness ---------------------------------------------


async def test_sse_framing():
    data = (FIXTURES / "codex_poc_r1.sse.txt").read_bytes()
    whole = await parse_events(data)

    # odd-sized chunks must produce identical events
    async def chunked(src: bytes, size: int):
        for i in range(0, len(src), size):
            yield src[i : i + size]

    pieces = []
    async for e in incremental_sse(chunked(data, 7)):
        pieces.append(e)

    check("sse whole-vs-chunked count", len(whole) == len(pieces), f"{len(whole)} vs {len(pieces)}")
    types_w = [e.get("type") for e in whole if isinstance(e, dict)]
    types_c = [e.get("type") for e in pieces if isinstance(e, dict)]
    check("sse whole-vs-chunked types", types_w == types_c)
    check("sse has completed", "response.completed" in types_w)
    check("sse no spurious DONE", DONE not in whole)  # Codex sends no [DONE]


# --- 3. fold rewrite on real r1 + r2 captures -------------------------------


async def test_fold_real_captures():
    cfg = load_config(ROOT / "config.toml")
    cfg = replace(cfg, cont=replace(cfg.cont, max_continue=1))  # r1 -> continue -> r2 -> stop

    r1 = FakeResp((FIXTURES / "codex_poc_r1.sse.txt").read_bytes())
    r2 = FakeResp((FIXTURES / "codex_poc_r2.sse.txt").read_bytes())
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}

    evs = await run_fold(cfg, base_body, r1, [r2])
    dict_evs = [e for e in evs if isinstance(e, dict)]
    types = [e.get("type") for e in dict_evs]

    check("fold one created", types.count("response.created") == 1)
    check("fold one in_progress", types.count("response.in_progress") == 1)
    check("fold one terminal", sum(types.count(t) for t in
          ("response.completed", "response.failed", "response.incomplete")) == 1)

    seqs = [e["sequence_number"] for e in dict_evs]
    check("fold seq monotonic 0..n", seqs == list(range(len(dict_evs))), str(seqs[:5]))

    # reasoning items forwarded at ds_oi 0 then 1
    rdone = [e for e in dict_evs if e.get("type") == "response.output_item.done"
             and (e.get("item") or {}).get("type") == "reasoning"]
    check("fold 2 reasoning items", len(rdone) == 2, str(len(rdone)))
    check("fold reasoning oi 0,1", [e["output_index"] for e in rdone] == [0, 1],
          str([e.get("output_index") for e in rdone]))

    # message flushed (r2) at ds_oi 2; r1 message discarded
    deltas = "".join(e.get("delta", "") for e in dict_evs
                     if e.get("type") == "response.output_text.delta")
    check("fold r2 answer present", "答案是" in deltas or "21" in deltas, deltas[:40])
    check("fold r1 message discarded", "最少需要取出" not in deltas)

    created = next(e for e in dict_evs if e.get("type") == "response.created")
    completed = dict_evs[-1]
    created_id = (created.get("response") or {}).get("id")
    completed_id = (completed.get("response") or {}).get("id")
    check("fold created/completed share id", created_id == completed_id,
          f"{created_id} vs {completed_id}")
    out_items = (completed.get("response") or {}).get("output") or []
    check("fold reconstructed output non-empty (3 items)", len(out_items) == 3, str(len(out_items)))
    # Agent-facing usage = single-response equivalent (NOT summed input).
    usage = (completed.get("response") or {}).get("usage") or {}
    check("fold input = round1 (4582, not summed)", usage.get("input_tokens") == 4582,
          str(usage.get("input_tokens")))
    check("fold cached = round1 (3840)",
          (usage.get("input_tokens_details") or {}).get("cached_tokens") == 3840)
    rt = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    check("fold reasoning summed 3104", rt == 516 + 2588, str(rt))
    # output = summed reasoning + final round's non-reasoning (2947-2588=359)
    check("fold output = reasoning + final msg",
          usage.get("output_tokens") == 3104 + (2947 - 2588), str(usage.get("output_tokens")))
    check("fold total = input + output",
          usage.get("total_tokens") == 4582 + 3104 + (2947 - 2588), str(usage.get("total_tokens")))

    md = (completed.get("response") or {}).get("metadata") or {}
    check("fold proxy_rounds has 2 entries", len(md.get("proxy_rounds") or []) == 2,
          str(md.get("proxy_rounds")))
    check("fold stopped_reason max_continue", md.get("proxy_stopped_reason") == "max_continue",
          str(md.get("proxy_stopped_reason")))
    billed = md.get("proxy_billed_usage") or {}
    check("fold billed input summed 9722", billed.get("input_tokens") == 4582 + 5140,
          str(billed.get("input_tokens")))


# --- 3b. truncated tool call is discarded; clean tool call flushes ----------


def _round(rs_id, enc, reasoning_tokens_val, *, extra_items=None, msg=None):
    evs = [
        {"type": "response.created", "response": {"id": "resp_x", "status": "in_progress",
         "model": "gpt-5.5", "metadata": {}}},
        {"type": "response.in_progress", "response": {"id": "resp_x"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"id": rs_id, "type": "reasoning"}},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {"id": rs_id, "type": "reasoning", "encrypted_content": enc}},
    ]
    oi = 1
    for it in (extra_items or []):
        evs.append({"type": "response.output_item.added", "output_index": oi, "item": it})
        if it["type"] == "function_call":
            evs.append({"type": "response.function_call_arguments.delta", "output_index": oi,
                        "item_id": it["id"], "delta": it.get("arguments", "{}")})
        evs.append({"type": "response.output_item.done", "output_index": oi, "item": it})
        oi += 1
    if msg is not None:
        evs += [
            {"type": "response.output_item.added", "output_index": oi,
             "item": {"id": "msg_x", "type": "message"}},
            {"type": "response.content_part.added", "output_index": oi, "item_id": "msg_x",
             "content_index": 0, "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "output_index": oi, "item_id": "msg_x",
             "content_index": 0, "delta": msg},
            {"type": "response.output_text.done", "output_index": oi, "item_id": "msg_x",
             "content_index": 0, "text": msg},
            {"type": "response.content_part.done", "output_index": oi, "item_id": "msg_x",
             "content_index": 0, "part": {"type": "output_text", "text": msg}},
            {"type": "response.output_item.done", "output_index": oi,
             "item": {"id": "msg_x", "type": "message",
                      "content": [{"type": "output_text", "text": msg}]}},
        ]
    evs.append({"type": "response.completed", "response": {"id": "resp_x", "status": "completed",
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                          "output_tokens_details": {"reasoning_tokens": reasoning_tokens_val}}}})
    return make_sse(evs)


async def test_truncated_tool_call_discarded():
    cfg = load_config(ROOT / "config.toml")
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}

    # Round A: truncated (516) + a real tool call. Round B: clean message.
    tool = {"id": "fc_a", "type": "function_call", "name": "shell", "call_id": "call_a",
            "arguments": "{\"cmd\":\"ls\"}"}
    rA = FakeResp(_round("rs_a", "ENC_A", 516, extra_items=[tool]))
    rB = FakeResp(_round("rs_b", "ENC_B", 999, msg="done"))

    evs = [e for e in await run_fold(cfg, base_body, rA, [rB]) if isinstance(e, dict)]
    has_fc = any((e.get("item") or {}).get("type") == "function_call" for e in evs)
    fc_args = any(e.get("type") == "response.function_call_arguments.delta" for e in evs)
    check("truncated tool call discarded (no fc item)", not has_fc)
    check("truncated tool call discarded (no fc args)", not fc_args)
    deltas = "".join(e.get("delta", "") for e in evs
                     if e.get("type") == "response.output_text.delta")
    check("clean round message flushed", deltas == "done", deltas)

    # Clean round ending in a tool call → must flush it through.
    rOnly = FakeResp(_round("rs_c", "ENC_C", 999, extra_items=[tool]))
    evs2 = [e for e in await run_fold(cfg, base_body, rOnly, []) if isinstance(e, dict)]
    has_fc2 = any((e.get("item") or {}).get("type") == "function_call" for e in evs2)
    check("clean round tool call flushed", has_fc2)


# --- commentary continuation (default) vs tool_pair --------------------------


async def test_commentary_continuation_payload():
    cfg = load_config(ROOT / "config.toml")  # method = "commentary" by default
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}
    rA = FakeResp(_round("rs_a", "ENC_A", 516, msg="trunc"))  # truncated → continue
    rB = FakeResp(_round("rs_b", "ENC_B", 999, msg="done"))   # clean → stop
    client = FakeClient([rB])
    evs = [e for e in await run_fold_capture(cfg, base_body, rA, client) if isinstance(e, dict)]

    check("commentary: one continuation round opened", len(client.payloads) == 1,
          str(len(client.payloads)))
    inp = (client.payloads[0].get("input") if client.payloads else []) or []
    last = inp[-1] if inp else {}
    check("commentary: marker is a phase:commentary assistant message",
          last.get("type") == "message" and last.get("role") == "assistant"
          and last.get("phase") == "commentary", str(last))
    check("commentary: marker text from config",
          (last.get("content") or [{}])[0].get("text") == cfg.cont.marker_text)
    check("commentary: no function_call injected in replay",
          not any(isinstance(x, dict) and x.get("type") == "function_call" for x in inp))
    check("commentary: prior reasoning replayed (encrypted)",
          any(isinstance(x, dict) and x.get("type") == "reasoning"
              and x.get("encrypted_content") for x in inp))
    # forward_marker defaults false → marker stays hidden from the downstream stream
    check("commentary: marker hidden downstream by default",
          not any((e.get("item") or {}).get("phase") == "commentary" for e in evs))


async def test_commentary_continuation_log_warning():
    """验证自动续发时的 round 决策日志以 WARNING 级别输出。"""
    cfg = load_config(ROOT / "config.toml")
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}
    rA = FakeResp(_round("rs_a", "ENC_A", 516, msg="trunc"))
    rB = FakeResp(_round("rs_b", "ENC_B", 999, msg="done"))
    client = FakeClient([rB])
    capture = CaptureHandler()
    old_level = proxy_module.log.level
    old_propagate = proxy_module.log.propagate
    proxy_module.log.setLevel(logging.INFO)
    proxy_module.log.propagate = False
    proxy_module.log.addHandler(capture)
    try:
        await run_fold_capture(cfg, base_body, rA, client)
    finally:
        proxy_module.log.removeHandler(capture)
        proxy_module.log.setLevel(old_level)
        proxy_module.log.propagate = old_propagate

    continuation_records = [
        record for record in capture.records
        if "round 1:" in record.getMessage() and "-> continue" in record.getMessage()
    ]
    check("commentary continuation warning log count", len(continuation_records) == 1,
          str(len(continuation_records)))
    check("commentary continuation warning log level",
          continuation_records[0].levelno == logging.WARNING if continuation_records else False,
          str(continuation_records[0].levelno if continuation_records else None))


async def test_tool_pair_continuation_payload():
    base = load_config(ROOT / "config.toml")
    cfg = replace(base, cont=replace(base.cont, method="tool_pair"))
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}
    rA = FakeResp(_round("rs_a", "ENC_A", 516, msg="trunc"))
    rB = FakeResp(_round("rs_b", "ENC_B", 999, msg="done"))
    client = FakeClient([rB])
    await run_fold_capture(cfg, base_body, rA, client)

    inp = (client.payloads[0].get("input") if client.payloads else []) or []
    types = [x.get("type") for x in inp if isinstance(x, dict)]
    check("tool_pair: function_call + output injected",
          "function_call" in types and "function_call_output" in types, str(types))
    check("tool_pair: no commentary message in replay",
          not any(isinstance(x, dict) and x.get("phase") == "commentary" for x in inp))


async def test_forward_marker_emits_downstream():
    base = load_config(ROOT / "config.toml")
    cfg = replace(base, cont=replace(base.cont, method="commentary", forward_marker=True))
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}
    rA = FakeResp(_round("rs_a", "ENC_A", 516, msg="trunc"))
    rB = FakeResp(_round("rs_b", "ENC_B", 999, msg="done"))
    evs = [e for e in await run_fold(cfg, base_body, rA, [rB]) if isinstance(e, dict)]

    done = [e for e in evs if e.get("type") == "response.output_item.done"
            and (e.get("item") or {}).get("phase") == "commentary"]
    check("forward_marker: one commentary item emitted downstream", len(done) == 1,
          str(len(done)))
    delta = "".join(e.get("delta", "") for e in evs
                    if e.get("type") == "response.output_text.delta"
                    and e.get("item_id", "").startswith("msg_continue_"))
    check("forward_marker: commentary delta carries marker text",
          delta == cfg.cont.marker_text, delta)
    # reconstructed output carries the commentary item (so the agent echoes it)
    completed = evs[-1]
    out_items = (completed.get("response") or {}).get("output") or []
    phases = [it.get("phase") for it in out_items if isinstance(it, dict)]
    check("forward_marker: commentary in reconstructed output", "commentary" in phases,
          str(phases))
    # sequence numbers stay monotonic 0..n despite the injected item
    seqs = [e["sequence_number"] for e in evs]
    check("forward_marker: seq monotonic with injected marker",
          seqs == list(range(len(evs))), str(seqs[:6]))


# --- 2-fix. header transparency (#2) ----------------------------------------


def test_header_transparency():
    cfg = load_config(ROOT / "config.toml")
    client = _make_client()
    check("client invents no user-agent", "user-agent" not in client.headers)
    check("client invents no accept", "accept" not in client.headers)

    agent = [
        ("Authorization", "Bearer agent"),
        ("Content-Type", "application/json"),
        ("User-Agent", "codex_cli_rs/1.0"),
        ("Host", "drop.me"),
        ("Content-Length", "123"),
        ("Accept-Encoding", "gzip"),
        ("Responses-API-Base", "https://override/responses"),
        ("X-Custom", "keep"),
        ("Upgrade", "websocket"),
        ("Sec-WebSocket-Key", "drop"),
    ]
    out = build_upstream_headers(agent, cfg)
    low = {k.lower(): v for k, v in out.items()}
    check("hdr keeps content-type", low.get("content-type") == "application/json")
    check("hdr keeps user-agent", low.get("user-agent") == "codex_cli_rs/1.0")
    check("hdr keeps custom", low.get("x-custom") == "keep")
    check("hdr keeps authorization", low.get("authorization") == "Bearer agent")
    for dropped in (
        "host",
        "content-length",
        "accept-encoding",
        "responses-api-base",
        "upgrade",
        "sec-websocket-key",
    ):
        check(f"hdr drops {dropped}", dropped not in low)


# --- upstream URL resolution via Responses-API-Base header ------------------


class _Req:
    def __init__(self, headers: dict):
        self.headers = Headers(headers)


def test_upstream_url_resolution():
    base = load_config(ROOT / "config.toml")
    fixed = replace(base, upstream=replace(base.upstream, mode="fixed", url="https://cfg/responses"))
    header = replace(base, upstream=replace(base.upstream, mode="header", url="https://cfg/responses"))
    with_hdr = _Req({"Responses-API-Base": "https://override/v1"})
    no_hdr = _Req({})

    check("fixed ignores header", _resolve_upstream_url(fixed, with_hdr) == "https://cfg/responses")
    check("header appends /responses to base",
          _resolve_upstream_url(header, with_hdr) == "https://override/v1/responses")
    check("header falls back to url",
          _resolve_upstream_url(header, no_hdr) == "https://cfg/responses")
    check("header trims trailing slash + case-insensitive",
          _resolve_upstream_url(header, _Req({"responses-api-base": "https://low/v1/"})) == "https://low/v1/responses")
    check("header full endpoint left as-is",
          _resolve_upstream_url(header, _Req({"Responses-API-Base": "https://x/v1/responses"})) == "https://x/v1/responses")
    check("header blank → fallback",
          _resolve_upstream_url(header, _Req({"Responses-API-Base": "   "})) == "https://cfg/responses")

    # header_required: present → use it; absent/blank → None (caller returns 400)
    req = replace(base, upstream=replace(base.upstream, mode="header_required", url="https://cfg/responses"))
    check("required appends /responses",
          _resolve_upstream_url(req, with_hdr) == "https://override/v1/responses")
    check("required missing → None", _resolve_upstream_url(req, no_hdr) is None)
    check("required blank → None",
          _resolve_upstream_url(req, _Req({"Responses-API-Base": " "})) is None)


# --- security guard: never send config creds to a header-supplied URL --------


def test_auth_safety_guard():
    base = load_config(ROOT / "config.toml")

    def blocked(url_mode, auth_mode, token, has_hdr, has_auth):
        cfg = replace(
            base,
            upstream=replace(base.upstream, mode=url_mode),
            auth=replace(base.auth, mode=auth_mode, access_token=token),
        )
        h = {}
        if has_hdr:
            h["Responses-API-Base"] = "https://external/responses"
        if has_auth:
            h["Authorization"] = "Bearer agent"
        rq = _Req(h)
        from_hdr = _url_is_from_header(cfg, rq)
        inj = would_inject_authorization(
            cfg, agent_has_authorization=rq.headers.get("authorization") is not None
        )
        return from_hdr and inj  # the exact condition handle_responses rejects on

    # fixed url → always safe
    check("guard: fixed+inject allow", not blocked("fixed", "inject", "TOK", True, False))
    # header + passthrough → never injects → allow
    check("guard: header+passthrough allow",
          not blocked("header", "passthrough", "TOK", True, False))
    # header + inject + header present → block (even if agent has its own auth)
    check("guard: header+inject+hdr block (noauth)",
          blocked("header", "inject", "TOK", True, False))
    check("guard: header+inject+hdr block (auth)",
          blocked("header", "inject", "TOK", True, True))
    # header + inject, no header → config url → allow
    check("guard: header+inject no-hdr allow",
          not blocked("header", "inject", "TOK", False, False))
    # header + PtI + header + agent has own auth → allow (uses agent's)
    check("guard: header+PtI+hdr+auth allow",
          not blocked("header", "passthrough_then_inject", "TOK", True, True))
    # header + PtI + header + no agent auth → block (would inject config)
    check("guard: header+PtI+hdr+noauth block",
          blocked("header", "passthrough_then_inject", "TOK", True, False))
    # header_required + inject + header → block
    check("guard: required+inject+hdr block",
          blocked("header_required", "inject", "TOK", True, False))
    # empty configured token → nothing to leak → allow
    check("guard: empty token allow", not blocked("header", "inject", "", True, False))


# --- auth injection from config (#2 follow-up) ------------------------------


def test_auth_injection():
    base = load_config(ROOT / "config.toml")

    def hdrs(cfg, agent):
        return {k.lower(): v for k, v in build_upstream_headers(agent, cfg).items()}

    # passthrough_then_inject: inject token when agent sends none; empty account → no header
    cfg = replace(base, auth=replace(base.auth, mode="passthrough_then_inject",
                                     access_token="TOK", chatgpt_account_id=""))
    out = hdrs(cfg, [("x", "1")])
    check("inject token when missing", out.get("authorization") == "Bearer TOK")
    check("no account header when empty", "chatgpt-account-id" not in out)

    # passthrough_then_inject: agent's auth wins (not overridden)
    out2 = hdrs(cfg, [("Authorization", "Bearer AGENT")])
    check("fallback keeps agent auth", out2.get("authorization") == "Bearer AGENT")

    # inject: config overrides agent + adds account
    cfg2 = replace(base, auth=replace(base.auth, mode="inject",
                                      access_token="TOK", chatgpt_account_id="acct1"))
    out3 = hdrs(cfg2, [("Authorization", "Bearer AGENT")])
    check("inject overrides agent auth", out3.get("authorization") == "Bearer TOK")
    check("inject adds account", out3.get("chatgpt-account-id") == "acct1")

    # passthrough: never inject anything
    cfg3 = replace(base, auth=replace(base.auth, mode="passthrough",
                                      access_token="TOK", chatgpt_account_id="acct1"))
    out4 = hdrs(cfg3, [("x", "1")])
    check("passthrough never injects", "authorization" not in out4 and "chatgpt-account-id" not in out4)


# --- 4-fix. reasoning/stream gating (#4) ------------------------------------


def test_reasoning_gate():
    check("reasoning_enabled dict", reasoning_enabled({"reasoning": {"effort": "high"}}))
    check("reasoning_enabled absent → true", reasoning_enabled({"input": []}))
    check("reasoning_enabled null → true", reasoning_enabled({"reasoning": None}))
    check("reasoning_enabled empty dict → true", reasoning_enabled({"reasoning": {}}))
    check("reasoning_enabled explicit false → false", not reasoning_enabled({"reasoning": False}))


# --- 3-fix. stateful follow-up repair (#3) ----------------------------------


def test_stateful_repair():
    store = IdStore()
    store.add("rs_keep")
    inp = [
        {"role": "user", "content": "q"},
        {"type": "reasoning", "id": "rs_keep", "encrypted_content": "E1"},
        {"type": "reasoning", "id": "rs_natural", "encrypted_content": "E2"},  # not recorded
        {"type": "message", "id": "msg"},
    ]
    out = repair_followup_input(inp, store, tool_name="continue_thinking", output_text="go")

    # pair inserted right after rs_keep only
    idx = next(i for i, x in enumerate(out)
               if isinstance(x, dict) and x.get("id") == "rs_keep")
    nxt = out[idx + 1]
    nxt2 = out[idx + 2]
    cid = continue_call_id("rs_keep")
    check("stateful inserts call after recorded id",
          nxt.get("type") == "function_call" and nxt.get("call_id") == cid, str(nxt))
    check("stateful inserts output after call",
          nxt2.get("type") == "function_call_output" and nxt2.get("call_id") == cid)

    # natural-consecutive reasoning (unrecorded) gets NO splice
    nidx = next(i for i, x in enumerate(out)
                if isinstance(x, dict) and x.get("id") == "rs_natural")
    check("stateful no splice for unrecorded id",
          out[nidx + 1].get("type") == "message", str(out[nidx + 1]))

    # idempotent: re-running adds nothing
    out2 = repair_followup_input(out, store, tool_name="continue_thinking", output_text="go")
    check("stateful idempotent", len(out2) == len(out), f"{len(out)} -> {len(out2)}")


# --- 7-fix. graceful EOF → incomplete (#7) ----------------------------------


async def test_eof_incomplete():
    cfg = load_config(ROOT / "config.toml")
    base_body = {"model": "gpt-5.5", "input": [{"role": "user", "content": "q"}]}
    # A round that streams reasoning + message but NO terminal event.
    events = [
        {"type": "response.created", "response": {"id": "resp_e", "status": "in_progress"}},
        {"type": "response.in_progress", "response": {"id": "resp_e"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"id": "rs_e", "type": "reasoning"}},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {"id": "rs_e", "type": "reasoning", "encrypted_content": "E"}},
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"id": "msg_e", "type": "message"}},
        {"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_e",
         "content_index": 0, "delta": "partial"},
        {"type": "response.output_item.done", "output_index": 1,
         "item": {"id": "msg_e", "type": "message"}},
        # <-- no response.completed
    ]
    evs = [e for e in await run_fold(cfg, base_body, FakeResp(make_sse(events)), [])
           if isinstance(e, dict)]
    term = evs[-1]
    check("eof terminal is incomplete", term.get("type") == "response.incomplete",
          term.get("type"))
    reason = ((term.get("response") or {}).get("incomplete_details") or {}).get("reason")
    check("eof reason upstream_eof", reason == "upstream_eof", str(reason))

    # buffered tentative output must NOT leak on EOF (only reasoning survives)
    leaked = any(e.get("type") == "response.output_text.delta" for e in evs)
    check("eof does not leak buffered message", not leaked)
    out_items = (term.get("response") or {}).get("output") or []
    check("eof output is reasoning only",
          all(it.get("type") == "reasoning" for it in out_items) and len(out_items) == 1,
          str([it.get("type") for it in out_items]))


async def test_stream_idle_timeout_emits_incomplete_reason():
    cfg = load_config(ROOT / "config.toml")
    base_body = {"model": "gpt-5.5", "input": []}

    class IdleFoldResponse(FakeResp):
        async def aiter_bytes(self):
            yield make_sse([
                {
                    "type": "response.created",
                    "response": {"id": "resp_idle", "status": "in_progress"},
                },
                {
                "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "working",
                },
            ])
            await asyncio.sleep(1.0)

    response = IdleFoldResponse(b"")
    setattr(response, proxy_module._DIAG_ATTR, {
        "request_id": "up_fold_idle",
        "trace_id": "trace_fold_idle",
        "phase": "fold_round_1",
        "started": __import__("time").perf_counter(),
        "stream_idle_timeout_s": 0.05,
    })
    events = [
        event for event in await run_fold(cfg, base_body, response, [])
        if isinstance(event, dict)
    ]
    terminal = events[-1]
    reason = (
        ((terminal.get("response") or {}).get("incomplete_details") or {})
        .get("reason")
    )
    check(
        "stream idle timeout emits incomplete terminal",
        terminal.get("type") == "response.incomplete",
        str(terminal),
    )
    check(
        "stream idle timeout emits specific reason",
        reason == "stream_idle_timeout",
        str(reason),
    )

    class MetadataOnlyResponse(FakeResp):
        async def aiter_bytes(self):
            yield make_sse([
                {
                    "type": "response.created",
                    "response": {"id": "resp_metadata", "status": "in_progress"},
                },
                {
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": 0,
                    "delta": "working",
                },
            ])
            await asyncio.sleep(1.0)

    metadata_response = MetadataOnlyResponse(b"")
    setattr(metadata_response, proxy_module._DIAG_ATTR, {
        "request_id": "up_fold_metadata",
        "trace_id": "trace_fold_metadata",
        "phase": "fold_round_1",
        "started": __import__("time").perf_counter(),
        "first_byte_timeout_s": 0.05,
        "stream_idle_timeout_s": 0.01,
        "generation_started": False,
    })
    metadata_events = [
        event for event in await run_fold(
            cfg, base_body, metadata_response, []
        )
        if isinstance(event, dict)
    ]
    metadata_terminal = metadata_events[-1]
    metadata_reason = (
        ((metadata_terminal.get("response") or {}).get("incomplete_details") or {})
        .get("reason")
    )
    check(
        "reasoning summary preserves generation timeout reason",
        metadata_reason == "generation_timeout",
        str(metadata_reason),
    )

    class ActiveCustomToolResponse(FakeResp):
        async def aiter_bytes(self):
            yield make_sse([
                {
                    "type": "response.created",
                    "response": {"id": "resp_active", "status": "in_progress"},
                },
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "started",
                },
            ])
            for index in range(4):
                await asyncio.sleep(0.02)
                yield make_sse([{
                    "type": "response.custom_tool_call_input.delta",
                    "output_index": 1,
                    "delta": str(index),
                }])
            yield make_sse([{
                "type": "response.completed",
                "response": {
                    "id": "resp_active",
                    "status": "completed",
                    "output": [],
                },
            }])

    active_response = ActiveCustomToolResponse(b"")
    setattr(active_response, proxy_module._DIAG_ATTR, {
        "request_id": "up_fold_active_custom_tool",
        "trace_id": "trace_fold_active_custom_tool",
        "phase": "fold_round_1",
        "started": __import__("time").perf_counter(),
        "first_byte_timeout_s": 0.2,
        "stream_idle_timeout_s": 0.05,
        "generation_started": False,
    })
    active_events = [
        event for event in await run_fold(cfg, base_body, active_response, [])
        if isinstance(event, dict)
    ]
    check(
        "HTTP custom-tool events reset active stream idle budget",
        active_events[-1].get("type") == "response.completed",
        str(active_events[-1]),
    )


async def test_native_upstream_websocket_reuses_connection():
    captured: dict = {}

    class NativeConnection:
        def __init__(self):
            self.sent: list[str] = []
            self.closed = False
            self.response = SimpleNamespace(status_code=101, headers={"cf-ray": "test-ray"})
            self.incoming = [
                json.dumps({
                    "type": "response.created",
                    "response": {"id": "resp_one", "status": "in_progress"},
                }),
                json.dumps({
                    "type": "response.completed",
                    "response": {"id": "resp_one", "status": "completed", "output": []},
                }),
                json.dumps({
                    "type": "response.created",
                    "response": {"id": "resp_two", "status": "in_progress"},
                }),
                json.dumps({
                    "type": "response.completed",
                    "response": {"id": "resp_two", "status": "completed", "output": []},
                }),
            ]

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            return self.incoming.pop(0)

        async def close(self):
            self.closed = True

    native = NativeConnection()

    async def fake_connect(uri, **kwargs):
        captured["uri"] = uri
        captured["kwargs"] = kwargs
        return native

    session = await connect_upstream_websocket(
        url="https://chatgpt.com/backend-api/codex/responses?test=1",
        headers={
            "Authorization": "Bearer test",
            "User-Agent": "codex-test",
            "OpenAI-Beta": "another_feature=v1",
        },
        payload_logger=None,
        connection_id="ws_test",
        connector=fake_connect,
    )
    first = await session.open_round(
        {"model": "gpt-test", "stream": True, "input": [{"role": "user"}]},
        phase="fold_round_1",
        trace_id="trace_one",
    )
    first_events = [event async for event in first.aiter_events()]
    await first.aclose()
    second = await session.open_round(
        {
            "model": "gpt-test",
            "stream": True,
            "background": False,
            "previous_response_id": "resp_one",
            "input": [{"role": "user"}],
        },
        phase="passthrough",
        trace_id="trace_two",
    )
    second_events = [event async for event in second.aiter_events()]
    await second.aclose()
    await session.aclose()

    sent = [json.loads(raw) for raw in native.sent]
    beta = captured.get("kwargs", {}).get("additional_headers", {}).get("OpenAI-Beta", "")
    check("http upstream URL maps to wss for ws downstream",
          captured.get("uri") == "wss://chatgpt.com/backend-api/codex/responses?test=1",
          str(captured.get("uri")))
    check("native ws handshake includes required beta",
          "responses_websockets=2026-02-06" in beta, beta)
    check("native ws handshake preserves existing beta",
          "another_feature=v1" in beta, beta)
    check("native ws forwards user agent without duplicate header",
          captured.get("kwargs", {}).get("user_agent_header") == "codex-test"
          and not any(
              key.lower() == "user-agent"
              for key in captured.get("kwargs", {}).get("additional_headers", {})
          ),
          str(captured.get("kwargs")))
    check("native ws reuses one connection for two rounds",
          len(sent) == 2 and len(first_events) == 2 and len(second_events) == 2,
          str(sent))
    check("native ws sends response.create without HTTP-only fields",
          all(item.get("type") == "response.create" for item in sent)
          and all("stream" not in item and "background" not in item for item in sent),
          str(sent))
    check("native ws preserves previous_response_id on later round",
          sent[1].get("previous_response_id") == "resp_one", str(sent[1]))
    check("native ws session closes its upstream connection", native.closed)


async def test_native_upstream_websocket_stream_idle_timeout():
    class IdleConnection:
        def __init__(self, first_event):
            self.response = SimpleNamespace(status_code=101, headers={})
            self.recv_count = 0
            self.closed = False
            self.first_event = first_event

        async def send(self, data):
            pass

        async def recv(self):
            self.recv_count += 1
            if self.recv_count == 1:
                return json.dumps(self.first_event)
            await asyncio.sleep(1.0)
            raise AssertionError("idle receive should have timed out")

        async def close(self):
            self.closed = True

    connection = IdleConnection({
        "type": "response.output_text.delta",
        "output_index": 0,
        "delta": "started",
    })

    async def fake_connect(*args, **kwargs):
        return connection

    timeout_cfg = replace(
        load_config(ROOT / "config.toml").timeouts,
        stream_idle_timeout_s=0.05,
    )
    session = await connect_upstream_websocket(
        url="https://example.test/v1/responses",
        headers={},
        payload_logger=None,
        connection_id="ws_idle_test",
        timeout_cfg=timeout_cfg,
        connector=fake_connect,
    )
    response = await session.open_round(
        {"model": "gpt-5.5", "input": []},
        phase="fold_round_1",
        trace_id="trace_ws_idle",
    )
    raised = None
    try:
        async for _ in response.aiter_events():
            pass
    except Exception as exc:
        raised = exc
    check(
        "native websocket silence raises stream idle timeout",
        isinstance(raised, proxy_module.UpstreamStreamIdleTimeout),
        repr(raised),
    )
    check(
        "native websocket idle timeout closes upstream connection",
        connection.closed,
    )

    metadata_connection = IdleConnection({
        "type": "response.reasoning_summary_text.delta",
        "output_index": 0,
        "delta": "working",
    })

    async def metadata_connect(*args, **kwargs):
        return metadata_connection

    metadata_timeout_cfg = replace(
        timeout_cfg,
        small_timeout_s=0.05,
    )
    metadata_session = await connect_upstream_websocket(
        url="https://example.test/v1/responses",
        headers={},
        payload_logger=None,
        connection_id="ws_metadata_test",
        timeout_cfg=metadata_timeout_cfg,
        connector=metadata_connect,
    )
    metadata_response = await metadata_session.open_round(
        {"model": "gpt-5.5", "input": []},
        phase="fold_round_1",
        trace_id="trace_ws_metadata",
    )
    raised = None
    try:
        async for _ in metadata_response.aiter_events():
            pass
    except Exception as exc:
        raised = exc
    check(
        "websocket reasoning summary stays in generation timeout budget",
        isinstance(raised, proxy_module.UpstreamGenerationTimeout),
        repr(raised),
    )

    class KeepaliveConnection(IdleConnection):
        async def recv(self):
            self.recv_count += 1
            if self.recv_count == 1:
                return json.dumps(self.first_event)
            await asyncio.sleep(0.02)
            if self.recv_count < 6:
                return json.dumps({"type": "keepalive"})
            return json.dumps({
                "type": "response.completed",
                "response": {"id": "resp_keepalive", "status": "completed"},
            })

    keepalive_connection = KeepaliveConnection({
        "type": "response.output_text.delta",
        "output_index": 0,
        "delta": "started",
    })

    async def keepalive_connect(*args, **kwargs):
        return keepalive_connection

    keepalive_session = await connect_upstream_websocket(
        url="https://example.test/v1/responses",
        headers={},
        payload_logger=None,
        connection_id="ws_keepalive_test",
        timeout_cfg=timeout_cfg,
        connector=keepalive_connect,
    )
    keepalive_response = await keepalive_session.open_round(
        {"model": "gpt-5.5", "input": []},
        phase="fold_round_1",
        trace_id="trace_ws_keepalive",
    )
    raised = None
    try:
        async for _ in keepalive_response.aiter_events():
            pass
    except Exception as exc:
        raised = exc
    check(
        "websocket events reset active stream idle budget",
        raised is None,
        repr(raised),
    )


def test_http_downstream_keeps_http_upstream():
    base = load_config(ROOT / "config.toml")
    cfg = replace(base, cont=replace(base.cont, enabled=False))
    app = create_app(cfg)
    http_client = FakeClient([
        FakeResp(_round("rs_http", "ENC_HTTP", 999, msg="http done"))
    ])
    ws_calls: list[dict] = []

    async def forbidden_ws_connector(**kwargs):
        ws_calls.append(kwargs)
        raise AssertionError("HTTP downstream must not open an upstream WebSocket")

    with TestClient(app) as client:
        app.state.client = http_client
        app.state.ws_connector = forbidden_ws_connector
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.5",
                "stream": True,
                "input": [{"role": "user", "content": "http"}],
            },
        )

    check("http downstream receives upstream http response", response.status_code == 200)
    check("http downstream uses the http client exactly once",
          len(http_client.payloads) == 1, str(http_client.payloads))
    check("http downstream never opens upstream ws", not ws_calls, str(ws_calls))


class _RaisingClient(FakeClient):
    """HTTP client that fails before any upstream response is available."""

    def __init__(self, exc: Exception):
        super().__init__([])
        self._exc = exc

    async def send(self, req, stream=True):
        raise self._exc


def test_http_upstream_read_error_returns_502_without_asgi_crash():
    """httpx.ReadError on open_round must become a normal 502 JSON response.

    Regression: previously the exception escaped handle_responses and uvicorn
    logged ``Exception in ASGI application`` with a full traceback.
    """
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    failing = _RaisingClient(httpx.ReadError(""))

    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.client = failing
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.5",
                "stream": True,
                "input": [{"role": "user", "content": "hi"}],
            },
        )

    check("http read error returns 502", response.status_code == 502, str(response.status_code))
    body = response.json()
    err = body.get("error") if isinstance(body, dict) else None
    check(
        "http read error body is structured JSON",
        isinstance(err, dict)
        and err.get("code") == "upstream_transport_error"
        and err.get("type") == "ReadError",
        str(body),
    )


def test_http_upstream_timeout_returns_504():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    failing = _RaisingClient(httpx.ReadTimeout("timed out"))

    with TestClient(app, raise_server_exceptions=True) as client:
        app.state.client = failing
        response = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.5",
                "stream": True,
                "input": [{"role": "user", "content": "hi"}],
            },
        )

    check("http timeout returns 504", response.status_code == 504, str(response.status_code))
    body = response.json()
    err = body.get("error") if isinstance(body, dict) else None
    check(
        "http timeout body names Timeout type",
        isinstance(err, dict) and err.get("type") == "ReadTimeout",
        str(body),
    )


def test_models_config_and_url_resolution():
    """Models config is independent from Responses-API-Base and keeps query pairs."""
    cfg = load_config(ROOT / "config.example.toml")
    check("models config paths loaded", cfg.server.models_paths == ("/v1/models",),
          str(cfg.server.models_paths))
    check(
        "models config URL loaded",
        cfg.upstream.models_url == "https://chatgpt.com/backend-api/codex/models",
        cfg.upstream.models_url,
    )
    check("models timeout loaded", cfg.timeouts.models_timeout_s == 30.0,
          str(cfg.timeouts.models_timeout_s))

    configured = replace(
        cfg,
        upstream=replace(
            cfg.upstream,
            models_url="https://models.test/backend-api/codex/models?existing=1&dup=base",
        ),
    )
    request = _Req({"Responses-API-Base": "https://wrong.test/v1"})
    request.query_params = httpx.QueryParams(
        [("client_version", "0.149.1"), ("dup", "first value"), ("dup", "second+value")]
    )
    resolved = _resolve_models_upstream_url(configured, request)
    parsed = httpx.URL(resolved)
    pairs = list(httpx.QueryParams(parsed.query).multi_items())
    check(
        "models URL ignores Responses-API-Base",
        parsed.host == "models.test" and parsed.path == "/backend-api/codex/models",
        resolved,
    )
    check(
        "models URL preserves duplicate query pairs",
        pairs == [
            ("existing", "1"),
            ("dup", "base"),
            ("dup", "first value"),
            ("dup", "second+value"),
            ("client_version", "0.149.1"),
        ],
        str(pairs),
    )


def test_models_get_proxy_transparent_response_and_headers():
    cfg = load_config(ROOT / "config.example.toml")
    cfg = replace(
        cfg,
        auth=replace(cfg.auth, mode="passthrough"),
        upstream=replace(
            cfg.upstream,
            models_url="https://chatgpt.com/backend-api/codex/models?source=cfg",
        ),
    )
    raw = b'{"models":[{"id":"codex-special","capabilities":{"reasoning":true}}]}'
    app = create_app(cfg)
    fake = FakeClient([
        FakeResp(
            raw,
            headers={
                "content-type": "application/json; charset=utf-8",
                "etag": '"models-v1"',
                "set-cookie": "should-not-forward",
                "connection": "close",
            },
        )
    ])

    with TestClient(app) as client:
        app.state.client = fake
        response = client.get(
            "/v1/models?client_version=0.149.1&scope=codex&scope=full%20scope",
            headers={
                "Authorization": "Bearer agent-token",
                "chatgpt-account-id": "acct-models",
                "User-Agent": "codex-test/0.149.1",
                "Responses-API-Base": "https://request-controlled.invalid/v1",
                "Content-Length": "0",
            },
        )

    check("models route GET returns 200", response.status_code == 200,
          str(response.status_code))
    check("models body is byte-for-byte unchanged", response.content == raw,
          repr(response.content))
    check(
        "models content-type and etag preserved",
        response.headers.get("content-type") == "application/json; charset=utf-8"
        and response.headers.get("etag") == '"models-v1"',
        str(dict(response.headers)),
    )
    check("models does not forward Set-Cookie", "set-cookie" not in response.headers)

    request = fake.requests[0] if fake.requests else {}
    headers = {str(k).lower(): str(v) for k, v in (request.get("headers") or {}).items()}
    check(
        "models forwards auth account and user agent",
        headers.get("authorization") == "Bearer agent-token"
        and headers.get("chatgpt-account-id") == "acct-models"
        and headers.get("user-agent") == "codex-test/0.149.1",
        str(headers),
    )
    check(
        "models strips owned and control headers",
        all(name not in headers for name in
            ("host", "content-length", "responses-api-base")),
        str(headers),
    )
    check("models sends GET with configured timeout",
          request.get("method") == "GET" and request.get("timeout") == 30.0,
          str(request))
    upstream_url = request.get("url") or ""
    query_pairs = list(httpx.QueryParams(httpx.URL(upstream_url).query).multi_items())
    check(
        "models forwards all query pairs",
        query_pairs == [
            ("source", "cfg"),
            ("client_version", "0.149.1"),
            ("scope", "codex"),
            ("scope", "full scope"),
        ],
        str(query_pairs),
    )
    check("models does not add Responses payload", fake.payloads == [], str(fake.payloads))


def test_models_upstream_status_body_and_transport_errors():
    cfg = load_config(ROOT / "config.example.toml")
    raw = b'{"error":{"type":"upstream","message":"unchanged"}}'
    for status in (401, 403, 500):
        app = create_app(cfg)
        fake = FakeClient([FakeResp(raw, status=status, headers={"content-type": "application/json"})])
        with TestClient(app, raise_server_exceptions=True) as client:
            app.state.client = fake
            response = client.get("/v1/models?client_version=0.149.1")
        check(f"models preserves upstream status {status}", response.status_code == status,
              str(response.status_code))
        check(f"models preserves upstream body {status}", response.content == raw,
              repr(response.content))

    for exc, expected_status, label in (
        (httpx.ReadError("read failed"), 502, "ReadError"),
        (httpx.ReadTimeout("timed out"), 504, "ReadTimeout"),
    ):
        app = create_app(cfg)
        failing = _RaisingClient(exc)
        with TestClient(app, raise_server_exceptions=True) as client:
            app.state.client = failing
            response = client.get("/v1/models?client_version=0.149.1")
        body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        check(f"models {label} maps status", response.status_code == expected_status,
              str(response.status_code))
        check(
            f"models {label} maps structured error",
            isinstance(err, dict)
            and err.get("code") == "upstream_transport_error"
            and err.get("type") == label,
            str(body),
        )


def test_first_byte_timeout_policy_uses_serialized_body_size():
    cfg = load_config(ROOT / "config.example.toml")
    policy = cfg.timeouts
    small = policy.small_body_max_bytes
    large = policy.large_body_max_bytes

    check(
        "small request uses 60s first-byte budget",
        proxy_module.first_byte_timeout_for_body(small, policy) == 60.0,
        str(policy),
    )
    check(
        "medium request uses 180s first-byte budget",
        proxy_module.first_byte_timeout_for_body(small + 1, policy) == 180.0
        and proxy_module.first_byte_timeout_for_body(large, policy) == 180.0,
        str(policy),
    )
    check(
        "large request uses 300s first-byte budget",
        proxy_module.first_byte_timeout_for_body(large + 1, policy) == 300.0,
        str(policy),
    )
    check(
        "stream idle timeout defaults to 5s",
        policy.stream_idle_timeout_s == 5.0,
        str(policy),
    )
    check(
        "metadata events do not count as model generation",
        not proxy_module.is_generation_delta_event({"type": "codex.rate_limits"})
        and not proxy_module.is_generation_delta_event({"type": "response.created"}),
    )
    check(
        "reasoning summary stays in generation budget",
        not proxy_module.is_generation_delta_event({
            "type": "response.reasoning_summary_text.delta"
        }),
    )
    check(
        "final output deltas start active stream phase",
        proxy_module.is_generation_delta_event({
            "type": "response.output_text.delta"
        })
        and proxy_module.is_generation_delta_event({
            "type": "response.function_call_arguments.delta"
        })
        and proxy_module.is_generation_delta_event({
            "type": "response.custom_tool_call_input.delta"
        }),
    )

    diagnostic_response = FakeResp(b"")
    diagnostic_started = __import__("time").perf_counter()
    diagnostic_state = {
        "started": diagnostic_started,
        "generation_started": False,
    }
    setattr(diagnostic_response, proxy_module._DIAG_ATTR, diagnostic_state)
    proxy_module._record_upstream_chunk(diagnostic_response, 11)
    for event in (
        {"type": "response.created"},
        {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
        {"type": "response.output_text.delta", "delta": "answer"},
        {"type": "response.function_call_arguments.delta", "delta": "{}"},
        {"type": "response.custom_tool_call_input.delta", "delta": "input"},
        {"type": "response.output_item.done"},
    ):
        proxy_module._record_upstream_event(diagnostic_response, event)
    proxy_module._record_upstream_chunk(diagnostic_response, 23)
    timeout_diag = proxy_module._timeout_event_diagnostics(
        diagnostic_state, diagnostic_started, "stream_idle"
    )
    required_timeout_fields = {
        "timeout_phase",
        "last_event_type",
        "last_event_at",
        "last_reasoning_delta_at",
        "last_output_delta_type",
        "last_output_delta_at",
        "reasoning_delta_count",
        "output_delta_count",
        "ms_since_last_output_delta",
        "last_raw_chunk_at",
        "raw_chunk_count",
        "last_raw_chunk_bytes",
        "ms_since_last_raw_chunk",
        "raw_bytes_since_last_event",
        "generation_elapsed_ms",
        "recent_event_types",
    }
    check(
        "timeout diagnostics expose every promised evidence field",
        required_timeout_fields <= timeout_diag.keys(),
        str(timeout_diag),
    )
    check(
        "timeout diagnostics record raw transport activity",
        timeout_diag["last_raw_chunk_at"] > 0
        and timeout_diag["raw_chunk_count"] == 2
        and timeout_diag["last_raw_chunk_bytes"] == 23
        and timeout_diag["ms_since_last_raw_chunk"] >= 0
        and timeout_diag["raw_bytes_since_last_event"] == 23,
        str(timeout_diag),
    )
    check(
        "timeout diagnostics record reasoning and final output events",
        timeout_diag["last_event_type"]
        == "response.output_item.done"
        and timeout_diag["last_output_delta_type"]
        == "response.custom_tool_call_input.delta"
        and timeout_diag["reasoning_delta_count"] == 1
        and timeout_diag["output_delta_count"] == 3
        and diagnostic_state["last_generation_at"]
        > diagnostic_state["last_output_delta_monotonic"]
        and "response.reasoning_summary_text.delta"
        in timeout_diag["recent_event_types"],
        str(timeout_diag),
    )


async def test_first_byte_timeout_on_headers():
    """send() hanging past FIRST_BYTE_TIMEOUT_S becomes ReadTimeout."""
    old = proxy_module.FIRST_BYTE_TIMEOUT_S
    proxy_module.FIRST_BYTE_TIMEOUT_S = 0.05

    class SlowHeaderClient(FakeClient):
        async def send(self, req, stream=True):
            await asyncio.sleep(1.0)
            return await super().send(req, stream=stream)

    client = SlowHeaderClient([FakeResp(b"data: {}\n\n")])
    try:
        raised = None
        try:
            await open_round(
                client,
                "https://example.test/responses",
                {"model": "test", "input": []},
                {},
                transport="http",
                phase="fold_round_1",
                trace_id="trace_fb_headers",
            )
        except Exception as exc:
            raised = exc
    finally:
        proxy_module.FIRST_BYTE_TIMEOUT_S = old

    check(
        "first-byte header hang raises ReadTimeout",
        isinstance(raised, httpx.ReadTimeout) and "first-byte timeout" in str(raised),
        repr(raised),
    )


async def test_first_byte_and_stream_idle_timeouts():
    """First-byte and reset-on-each-chunk idle budgets are independent."""
    old = proxy_module.FIRST_BYTE_TIMEOUT_S
    proxy_module.FIRST_BYTE_TIMEOUT_S = 0.05

    class SlowFirstChunk(FakeResp):
        async def aiter_bytes(self):
            await asyncio.sleep(1.0)
            yield b"event: response.created\ndata: {}\n\n"

    class IdleAfterFirst(FakeResp):
        async def aiter_bytes(self):
            yield b"event: response.created\ndata: {\"type\":\"response.created\"}\n\n"
            await asyncio.sleep(0.2)
            yield b"event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"

    class DeltaThenIdle(FakeResp):
        async def aiter_bytes(self):
            yield b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n"
            await asyncio.sleep(0.2)
            yield b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"b\"}\n\n"

    class ContinuouslyActive(FakeResp):
        async def aiter_bytes(self):
            for index in range(4):
                await asyncio.sleep(0.02)
                yield (
                    "data: {\"type\":\"response.output_text.delta\","
                    f"\"delta\":\"{index}\"}}\n\n"
                ).encode()

    def diagnostics(trace_id: str, *, idle_timeout_s: float = 0.05):
        return {
            "request_id": f"up_{trace_id}",
            "trace_id": trace_id,
            "phase": "fold_round_1",
            "started": __import__("time").perf_counter(),
            "first_byte_timeout_s": 0.05,
            "stream_idle_timeout_s": idle_timeout_s,
            "generation_started": False,
        }

    try:
        slow = SlowFirstChunk(b"")
        setattr(slow, proxy_module._DIAG_ATTR, diagnostics("slow"))
        raised = None
        try:
            async for _ in proxy_module.observed_response_bytes(slow):
                pass
        except Exception as exc:
            raised = exc
        check(
            "first-byte body hang raises ReadTimeout",
            isinstance(raised, httpx.ReadTimeout) and "first-byte timeout" in str(raised),
            repr(raised),
        )

        idle = IdleAfterFirst(b"")
        setattr(idle, proxy_module._DIAG_ATTR, diagnostics("idle"))
        raised = None
        try:
            async for _ in proxy_module.observed_response_bytes(idle):
                pass
        except Exception as exc:
            raised = exc
        check(
            "metadata does not end generation timeout budget",
            isinstance(raised, proxy_module.UpstreamGenerationTimeout),
            repr(raised),
        )

        delta_idle = DeltaThenIdle(b"")
        delta_diag = diagnostics("delta_idle")
        setattr(delta_idle, proxy_module._DIAG_ATTR, delta_diag)
        delta_iter = proxy_module.observed_response_bytes(delta_idle).__aiter__()
        await delta_iter.__anext__()
        delta_diag["generation_started"] = True
        delta_diag["last_generation_at"] = __import__("time").perf_counter()
        raised = None
        try:
            await delta_iter.__anext__()
        except Exception as exc:
            raised = exc
        check(
            "post-generation silence raises stream idle timeout",
            isinstance(raised, proxy_module.UpstreamStreamIdleTimeout),
            repr(raised),
        )

        active = ContinuouslyActive(b"")
        active_diag = diagnostics("active")
        setattr(active, proxy_module._DIAG_ATTR, active_diag)
        active_iter = proxy_module.observed_response_bytes(active).__aiter__()
        chunks = [await active_iter.__anext__()]
        proxy_module._mark_generation_started(active)
        while True:
            try:
                chunk = await active_iter.__anext__()
            except StopAsyncIteration:
                break
            chunks.append(chunk)
            proxy_module._mark_generation_started(active)
        check(
            "active stream resets idle timeout after each chunk",
            len(chunks) == 4,
            str(chunks),
        )
    finally:
        proxy_module.FIRST_BYTE_TIMEOUT_S = old


def test_websocket_handshake_504_is_returned_without_crashing():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)

    async def failing_connector(**kwargs):
        raise UpstreamWebSocketError(
            "Upstream WebSocket handshake failed", status_code=504
        )

    with TestClient(app) as client:
        app.state.ws_connector = failing_connector
        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json({
                "type": "response.create",
                "model": "gpt-5.5",
                "input": [{"role": "user", "content": "trigger handshake"}],
            })
            event = ws.receive_json()

    check("ws upstream handshake 504 is returned as an error event",
          event.get("type") == "error" and event.get("status") == 504,
          str(event))
    check("ws upstream handshake 504 has a stable proxy error code",
          (event.get("error") or {}).get("code") == "upstream_connection_error",
          str(event))


def test_websocket_route_streams_events():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    connector = FakeWsConnector([
        FakeResp(_round("rs_ws", "ENC_WS", 999, msg="done"))
    ])

    with TestClient(app) as client:
        app.state.ws_connector = connector
        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "input": [{"role": "user", "content": "q"}],
                }
            )
            evs = []
            while True:
                ev = ws.receive_json()
                evs.append(ev)
                if ev.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                    break

    types = [e.get("type") for e in evs]
    check("ws route emits created", "response.created" in types, str(types[:4]))
    check("ws route emits terminal", "response.completed" in types, str(types[-2:]))
    deltas = "".join(e.get("delta", "") for e in evs if e.get("type") == "response.output_text.delta")
    check("ws route forwards text delta", deltas == "done", deltas)
    check("ws route opens exactly one upstream ws session",
          len(connector.sessions) == 1, str(len(connector.sessions)))
    check("ws route sends response.create over upstream ws",
          len(connector.payloads) == 1
          and connector.payloads[0].get("type") == "response.create",
          str(connector.payloads))
    check("ws wire payload omits HTTP stream field",
          "stream" not in connector.payloads[0], str(connector.payloads[0]))


async def test_websocket_disconnect_cancels_waiting_upstream():
    cfg = load_config(ROOT / "config.toml")
    upstream = BlockingWsConnector()
    app = SimpleNamespace(state=SimpleNamespace(
        cfg=cfg,
        client=FakeClient([]),
        ws_connector=upstream,
        id_store=IdStore(),
        payload_logger=None,
    ))
    websocket = FakeDownstreamWebSocket(app)
    handler = asyncio.create_task(handle_responses_ws(websocket))

    await websocket.send_response_create()
    await asyncio.wait_for(upstream.session.started.wait(), timeout=1)
    await websocket.disconnect()

    done, _ = await asyncio.wait({handler}, timeout=1)
    completed_promptly = handler in done
    handler_error = None
    if completed_promptly:
        try:
            await handler
        except BaseException as exc:
            handler_error = exc
    else:
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)

    check("ws disconnect cancels upstream waiting for headers",
          upstream.session.cancelled.is_set())
    check("ws disconnect closes corresponding upstream ws session",
          upstream.session.closed.is_set())
    check("ws disconnect ends handler promptly", completed_promptly)
    check("ws disconnect cancellation is handled cleanly",
          handler_error is None, repr(handler_error))


def test_websocket_warmup_generate_false():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    warmup_response = make_sse([
        {
            "type": "response.created",
            "response": {"id": "resp_warm_native", "status": "in_progress"},
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_warm_native",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        },
    ])
    connector = FakeWsConnector([
        FakeResp(warmup_response),
        FakeResp(_round("rs_after_warm", "ENC", 999, msg="after warmup")),
    ])

    with TestClient(app) as client:
        app.state.ws_connector = connector
        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "generate": False,
                    "tools": [{"type": "function", "name": "shell"}],
                    "input": [{"type": "message", "role": "developer", "content": "warm"}],
                }
            )
            ev1 = ws.receive_json()
            ev2 = ws.receive_json()
            warm_id = (ev2.get("response") or {}).get("id")
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "previous_response_id": warm_id,
                    "input": [{"role": "user", "content": "run"}],
                }
            )
            while True:
                ev = ws.receive_json()
                if ev.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                    break

    check("ws warmup emits created", ev1.get("type") == "response.created", str(ev1))
    check("ws warmup emits completed", ev2.get("type") == "response.completed", str(ev2))
    check("ws warmup returns native upstream response id",
          (ev2.get("response") or {}).get("id") == "resp_warm_native",
          str((ev2.get("response") or {}).get("id")))
    check("ws warmup is forwarded upstream",
          connector.phases[:1] == ["warmup"]
          and connector.payloads[0].get("generate") is False,
          str(connector.payloads[:1]))
    merged = connector.payloads[1] if len(connector.payloads) > 1 else {}
    minput = merged.get("input") or []
    check("ws warmup merges cached input into next upstream request", len(minput) == 2, str(minput))
    check("ws warmup preserves cached tools on next upstream request",
          (merged.get("tools") or [{}])[0].get("name") == "shell",
          str(merged.get("tools")))
    check("ws warmup strips synthetic previous_response_id before upstream",
          "previous_response_id" not in merged, str(merged))


def test_websocket_preserves_native_previous_response_id():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    connector = FakeWsConnector([
        FakeResp(_round("rs_prev", "ENC", 999, msg="done"))
    ])

    with TestClient(app) as client:
        app.state.ws_connector = connector
        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "previous_response_id": "resp_existing",
                    "input": [{"role": "user", "content": "continue"}],
                }
            )
            while True:
                ev = ws.receive_json()
                if ev.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                    break

    sent = connector.payloads[0] if connector.payloads else {}
    check("ws preserves native previous_response_id for codex upstream",
          sent.get("previous_response_id") == "resp_existing", str(sent))


def test_websocket_replays_previous_tool_call_for_output():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    tool = {
        "id": "fc_replay",
        "type": "function_call",
        "name": "shell",
        "call_id": "call_replay",
        "arguments": "{\"cmd\":\"pwd\"}",
    }
    connector = FakeWsConnector([
        FakeResp(_round("rs_replay", "ENC_REPLAY", 999, extra_items=[tool])),
        FakeResp(_round("rs_done", "ENC_DONE", 999, msg="done")),
    ])

    with TestClient(app) as client:
        app.state.ws_connector = connector
        with client.websocket_connect("/v1/responses") as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "input": [{"role": "user", "content": "run pwd"}],
                }
            )
            response_id = None
            while True:
                ev = ws.receive_json()
                if ev.get("type") in (
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                ):
                    response_id = (ev.get("response") or {}).get("id")
                    break

            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "previous_response_id": response_id,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_replay",
                            "output": "/app",
                        }
                    ],
                }
            )
            while True:
                ev = ws.receive_json()
                if ev.get("type") in (
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                ):
                    break

    replayed = connector.payloads[1] if len(connector.payloads) > 1 else {}
    replayed_input = replayed.get("input") or []
    replayed_types = [
        item.get("type") for item in replayed_input if isinstance(item, dict)
    ]
    check("tool output replay strips unsupported previous_response_id",
          "previous_response_id" not in replayed, str(replayed))
    check("tool output replay includes matching function_call",
          "function_call" in replayed_types, str(replayed_types))
    check("tool output replay keeps function_call_output",
          replayed_types[-1:] == ["function_call_output"], str(replayed_types))
    call_ids = [
        item.get("call_id")
        for item in replayed_input
        if isinstance(item, dict) and item.get("type") in (
            "function_call", "function_call_output"
        )
    ]
    check("tool output replay call ids match",
          call_ids == ["call_replay", "call_replay"], str(call_ids))


def test_websocket_response_contexts_are_connection_local():
    cfg = load_config(ROOT / "config.toml")
    app = create_app(cfg)
    tool_a = {
        "id": "fc_a",
        "type": "function_call",
        "name": "shell",
        "call_id": "call_connection_a",
        "arguments": "{}",
    }
    tool_b = {
        "id": "fc_b",
        "type": "function_call",
        "name": "shell",
        "call_id": "call_connection_b",
        "arguments": "{}",
    }
    connector = FakeWsConnector([
        FakeResp(_round("rs_a", "ENC_A", 999, extra_items=[tool_a])),
        FakeResp(_round("rs_b", "ENC_B", 999, extra_items=[tool_b])),
        FakeResp(_round("rs_done_a", "ENC_DONE_A", 999, msg="a done")),
        FakeResp(_round("rs_done_b", "ENC_DONE_B", 999, msg="b done")),
    ])

    def receive_terminal(ws):
        while True:
            ev = ws.receive_json()
            if ev.get("type") in (
                "response.completed",
                "response.failed",
                "response.incomplete",
            ):
                return (ev.get("response") or {}).get("id")

    with TestClient(app) as client:
        app.state.ws_connector = connector
        with client.websocket_connect("/v1/responses") as ws_a:
            with client.websocket_connect("/v1/responses") as ws_b:
                ws_a.send_json({
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "input": [{"role": "user", "content": "connection a"}],
                })
                response_a = receive_terminal(ws_a)
                ws_b.send_json({
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "input": [{"role": "user", "content": "connection b"}],
                })
                response_b = receive_terminal(ws_b)

                ws_a.send_json({
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "previous_response_id": response_a,
                    "input": [{
                        "type": "function_call_output",
                        "call_id": "call_connection_a",
                        "output": "a",
                    }],
                })
                receive_terminal(ws_a)
                ws_b.send_json({
                    "type": "response.create",
                    "model": "gpt-5.5",
                    "previous_response_id": response_b,
                    "input": [{
                        "type": "function_call_output",
                        "call_id": "call_connection_b",
                        "output": "b",
                    }],
                })
                receive_terminal(ws_b)

    def replayed_call_ids(payload):
        return [
            item.get("call_id")
            for item in payload.get("input") or []
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]

    payload_a = connector.payloads[2] if len(connector.payloads) > 2 else {}
    payload_b = connector.payloads[3] if len(connector.payloads) > 3 else {}
    check("connection a replays only its own function call",
          replayed_call_ids(payload_a) == ["call_connection_a"], str(payload_a))
    check("connection b replays only its own function call",
          replayed_call_ids(payload_b) == ["call_connection_b"], str(payload_b))


def test_payload_sqlite_records_ws_rounds():
    base = load_config(ROOT / "config.toml")
    if not hasattr(base.log, "payload_sqlite_path"):
        check("payload sqlite config supported", False, "LogCfg.payload_sqlite_path missing")
        return

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "payloads.sqlite3"
        cfg = replace(base, log=replace(base.log, payload_sqlite_path=str(db_path)))
        app = create_app(cfg)
        connector = FakeWsConnector([
            FakeResp(_round("rs_log_a", "ENC_A", 516, msg="trunc")),
            FakeResp(_round("rs_log_b", "ENC_B", 999, msg="done")),
        ])

        with TestClient(app) as client:
            app.state.ws_connector = connector
            with client.websocket_connect("/v1/responses") as ws:
                ws.send_json(
                    {
                        "type": "response.create",
                        "model": "gpt-5.5",
                        "previous_response_id": "resp_existing",
                        "input": [{"role": "user", "content": "log me"}],
                    }
                )
                while True:
                    ev = ws.receive_json()
                    if ev.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                        break

        db = sqlite3.connect(db_path)
        try:
            rows = db.execute(
                "select transport, phase, url, status_code, model, payload_json "
                "from outbound_payloads order by id"
            ).fetchall()
        finally:
            db.close()

        check("payload sqlite records two ws upstream requests", len(rows) == 2, str(rows))
        phases = [r[1] for r in rows]
        check("payload sqlite records first and continuation phases",
              phases == ["fold_round_1", "continuation"], str(phases))
        first_payload = json.loads(rows[0][5]) if rows else {}
        check("payload sqlite stores native ws outbound body",
              first_payload.get("type") == "response.create"
              and first_payload.get("previous_response_id") == "resp_existing"
              and "stream" not in first_payload,
              str(first_payload))
        check("payload sqlite records status code",
              rows[0][3] == 200 and rows[1][3] == 200 if rows else False, str(rows))


def test_payload_sqlite_initialized_on_startup():
    base = load_config(ROOT / "config.toml")
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "payloads.sqlite3"
        cfg = replace(base, log=replace(base.log, payload_sqlite_path=str(db_path)))
        app = create_app(cfg)

        with TestClient(app):
            pass

        created_on_startup = db_path.exists()
        names = []
        if created_on_startup:
            db = sqlite3.connect(db_path)
            try:
                names = [
                    row[0]
                    for row in db.execute(
                        "select name from sqlite_master where type = 'table' order by name"
                    ).fetchall()
                ]
            finally:
                db.close()

        check("payload sqlite initializes database on startup", created_on_startup, str(db_path))
        check("payload sqlite creates outbound_payloads table",
              "outbound_payloads" in names, str(names))


async def test_upstream_diagnostic_logs_correlate_504():
    response = FakeResp(b'{"error":{"message":"gateway timeout"}}', status=504)
    response.headers = {"x-request-id": "upstream-test-id"}
    client = FakeClient([response])
    capture = CaptureHandler()
    logger = proxy_module.log
    old_level = logger.level
    old_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(capture)
    try:
        opened = await open_round(
            client,
            "https://example.test/responses",
            {"model": "test-model", "input": [{"role": "user", "content": "q"}]},
            {},
            transport="ws",
            phase="fold_round_1",
            trace_id="trace_test_504",
        )
        await read_upstream_error(opened)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    starts = [m for m in capture.messages if "event=upstream_request_start" in m]
    headers = [m for m in capture.messages if "event=upstream_response_headers" in m]
    bodies = [m for m in capture.messages if "event=upstream_error_body_end" in m]

    def field(message: str, name: str) -> str | None:
        prefix = f"{name}="
        return next(
            (part[len(prefix):] for part in message.split() if part.startswith(prefix)),
            None,
        )

    ids = [field(messages[0], "request_id") for messages in (starts, headers, bodies) if messages]
    check("diagnostic logs emit start, headers, and error body",
          len(starts) == len(headers) == len(bodies) == 1, str(capture.messages))
    check("diagnostic logs correlate 504 with one request id",
          len(ids) == 3 and len(set(ids)) == 1 and ids[0] not in (None, "-"), str(ids))
    check("diagnostic start log records selected timeout",
          bool(starts) and "timeout_s=60.0" in starts[0],
          starts[0] if starts else "missing")
    check("diagnostic headers log trace, status, elapsed, and upstream id",
          bool(headers)
          and "trace_id=trace_test_504" in headers[0]
          and "status=504" in headers[0]
          and "elapsed_ms=" in headers[0]
          and "x-request-id:upstream-test-id" in headers[0],
          headers[0] if headers else "missing")


async def test_upstream_failed_terminal_logs_error():
    cfg = load_config(ROOT / "config.toml")
    response = FakeResp(
        make_sse([
            {
                "type": "response.created",
                "response": {"id": "resp_failed", "status": "in_progress"},
            },
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_failed",
                    "status": "failed",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                        "message": "request is too large",
                    },
                    "incomplete_details": {"reason": "error"},
                },
            },
        ])
    )
    response.headers = {"cf-ray": "ray-terminal-test"}
    capture = CaptureHandler()
    logger = proxy_module.log
    old_level = logger.level
    old_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(capture)
    try:
        await run_fold(
            cfg,
            {
                "model": "gpt-5.5",
                "previous_response_id": "resp_previous",
                "input": [{"type": "message", "status": "incomplete"}],
            },
            response,
            [],
        )
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    failures = [
        message
        for message in capture.messages
        if "event=upstream_terminal_failure" in message
    ]
    check("failed terminal emits one diagnostic log", len(failures) == 1,
          str(capture.messages))
    message = failures[0] if failures else ""
    check(
        "failed terminal log preserves upstream error fields",
        "terminal_type=response.failed" in message
        and "response_status=failed" in message
        and "error_type=invalid_request_error" in message
        and "error_code=context_length_exceeded" in message
        and "error_message='request is too large'" in message
        and "incomplete_reason=error" in message
        and "previous_response_id_present=yes" in message
        and "input_items=1" in message
        and "input_tail_shape=message:incomplete" in message
        and "cf-ray:ray-terminal-test" in message,
        message,
    )


# --- runner -----------------------------------------------------------------


async def _main():
    test_truncation_math()
    await test_sse_framing()
    await test_fold_real_captures()
    await test_truncated_tool_call_discarded()
    await test_commentary_continuation_payload()
    await test_commentary_continuation_log_warning()
    await test_tool_pair_continuation_payload()
    await test_forward_marker_emits_downstream()
    test_header_transparency()
    test_upstream_url_resolution()
    test_auth_safety_guard()
    test_auth_injection()
    test_reasoning_gate()
    test_stateful_repair()
    await test_eof_incomplete()
    await test_stream_idle_timeout_emits_incomplete_reason()
    await test_native_upstream_websocket_reuses_connection()
    await test_native_upstream_websocket_stream_idle_timeout()
    test_http_downstream_keeps_http_upstream()
    test_http_upstream_read_error_returns_502_without_asgi_crash()
    test_http_upstream_timeout_returns_504()
    test_models_config_and_url_resolution()
    test_models_get_proxy_transparent_response_and_headers()
    test_models_upstream_status_body_and_transport_errors()
    test_first_byte_timeout_policy_uses_serialized_body_size()
    await test_first_byte_timeout_on_headers()
    await test_first_byte_and_stream_idle_timeouts()
    test_websocket_handshake_504_is_returned_without_crashing()
    test_websocket_route_streams_events()
    await test_websocket_disconnect_cancels_waiting_upstream()
    test_websocket_warmup_generate_false()
    test_websocket_preserves_native_previous_response_id()
    test_websocket_replays_previous_tool_call_for_output()
    test_websocket_response_contexts_are_connection_local()
    test_payload_sqlite_records_ws_rounds()
    test_payload_sqlite_initialized_on_startup()
    await test_upstream_diagnostic_logs_correlate_504()
    await test_upstream_failed_terminal_logs_error()


def main():
    asyncio.run(_main())
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {name}"
        if not ok and detail:
            line += f"  -- {detail}"
        print(line)
    print(f"\n{passed}/{len(_RESULTS)} checks passed")
    sys.exit(0 if passed == len(_RESULTS) else 1)


if __name__ == "__main__":
    main()
