"""Regression coverage for reasoning item activity and stream idle timeouts."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

from middleware import proxy as proxy_module
from middleware.config import Config, TimeoutCfg
from middleware.proxy import (
    UpstreamStreamIdleTimeout,
    _record_upstream_event,
)
from middleware.ws_upstream import UpstreamWebSocketSession
from test_middleware import FakeResp, make_sse, run_fold


# Reasoning event timing stays below idle timeout while exceeding first-byte budget.
REASONING_INTERVAL_S = 0.02
# Short first-byte budget makes the long-activity regression deterministic.
INITIAL_BUDGET_S = 0.06
# Active streams tolerate each individual reasoning event interval.
ACTIVE_IDLE_BUDGET_S = 0.12


def make_reasoning_lifecycle_events(prefix: str, pair_count: int = 4) -> list[dict]:
    """Build reasoning item lifecycle events used to exercise streamed activity."""
    events: list[dict] = []
    for index in range(pair_count):
        item = {"id": f"{prefix}_{index}", "type": "reasoning"}
        events.append({
            "type": "response.output_item.added",
            "output_index": index,
            "item": item,
        })
        events.append({
            "type": "response.output_item.done",
            "output_index": index,
            "item": {**item, "encrypted_content": "FAKE_encrypted_content"},
        })
    return events


def make_timeout_cfg(
    first_byte_timeout_s: float,
    stream_idle_timeout_s: float,
) -> TimeoutCfg:
    """Create isolated HTTP/WebSocket timeout budgets for regression scenarios."""
    return replace(
        TimeoutCfg(),
        small_timeout_s=first_byte_timeout_s,
        medium_timeout_s=first_byte_timeout_s,
        large_timeout_s=first_byte_timeout_s,
        stream_idle_timeout_s=stream_idle_timeout_s,
    )


def attach_http_timeout_diagnostics(
    response: FakeResp,
    *,
    first_byte_timeout_s: float,
    stream_idle_timeout_s: float,
    request_id: str,
) -> dict:
    """Attach the same timeout state used by the real HTTP upstream path."""
    diagnostics = {
        "request_id": request_id,
        "trace_id": request_id,
        "phase": "fold_round_1",
        "started": time.perf_counter(),
        "first_byte_timeout_s": first_byte_timeout_s,
        "stream_idle_timeout_s": stream_idle_timeout_s,
        "track_generation_start": True,
        "generation_started": False,
    }
    setattr(response, proxy_module._DIAG_ATTR, diagnostics)
    return diagnostics


async def collect_ws_events(response: Any) -> list[dict]:
    """Consume one native WebSocket response through its event streaming API."""
    events: list[dict] = []
    async for event in response.aiter_events():
        events.append(event)
    return events


class TimedHTTPResponse(FakeResp):
    """Yield controlled SSE activity chunks and record HTTP response cleanup."""

    def __init__(self, chunks: list[bytes], *, interval_s: float = 0.0,
                 block_after_chunks: bool = False):
        """Keep the chosen SSE chunks and timing for a fake HTTP response."""
        super().__init__(b"")
        self._chunks = chunks  # SSE chunks returned to the fold state machine.
        self._interval_s = interval_s  # Delay between upstream activity chunks.
        self._block_after_chunks = block_after_chunks  # Simulate upstream silence.
        self.closed = False  # Records whether the fold path released this response.

    async def aiter_bytes(self):
        """Return activity chunks with the configured delay or silent wait."""
        for index, chunk in enumerate(self._chunks):
            if index > 0 and self._interval_s > 0:
                await asyncio.sleep(self._interval_s)
            yield chunk
        if self._block_after_chunks:
            await asyncio.sleep(1.0)

    async def aclose(self) -> None:
        """Record release of the HTTP stream after terminal or timeout handling."""
        self.closed = True


class FakeWebSocketConnection:
    """Supply timed JSON events to the native upstream WebSocket session."""

    def __init__(self, events: list[dict], *, interval_s: float = 0.0):
        """Initialize the finite upstream response and connection state."""
        self.events = events  # Ordered native response events consumed by recv.
        self.interval_s = interval_s  # Delay applied before each event after the first.
        self.sent: list[str] = []  # Encoded response.create frames sent by the session.
        self.index = 0  # Index of the next native response event.
        self.closed = False  # Records whether the session closed the connection.
        self.response = SimpleNamespace(status_code=101, headers={})  # Handshake metadata.

    async def send(self, data: str) -> None:
        """Record the response.create frame sent for this upstream round."""
        self.sent.append(data)

    async def recv(self) -> str:
        """Return the next event or remain silent until the session times out."""
        if self.index < len(self.events):
            if self.index > 0 and self.interval_s > 0:
                await asyncio.sleep(self.interval_s)
            event = self.events[self.index]
            self.index += 1
            return json.dumps(event)
        await asyncio.sleep(1.0)
        raise AssertionError("upstream silence should have been cancelled")

    async def close(self) -> None:
        """Record that the session released the fake upstream connection."""
        self.closed = True


class ReasoningActivityTests(unittest.TestCase):
    """Verify the exact event predicate and shared output diagnostics."""

    def test_exact_activity_predicate(self) -> None:
        """Accept only existing output deltas and reasoning item added/done events."""
        accepted = [
            ("output_text_delta", {"type": "response.output_text.delta"}),
            ("function_arguments_delta", {
                "type": "response.function_call_arguments.delta",
            }),
            ("custom_tool_input_delta", {
                "type": "response.custom_tool_call_input.delta",
            }),
            ("reasoning_item_added", {
                "type": "response.output_item.added",
                "item": {"id": "reasoning_added", "type": "reasoning"},
            }),
            ("reasoning_item_done", {
                "type": "response.output_item.done",
                "item": {"id": "reasoning_done", "type": "reasoning"},
            }),
        ]
        rejected = [
            ("response_created", {"type": "response.created"}),
            ("rate_limits", {"type": "codex.rate_limits"}),
            ("reasoning_summary", {
                "type": "response.reasoning_summary_text.delta",
            }),
            ("unlisted_custom_delta", {"type": "response.custom.delta"}),
            ("message_added", {
                "type": "response.output_item.added",
                "item": {"id": "message", "type": "message"},
            }),
            ("message_done", {
                "type": "response.output_item.done",
                "item": {"id": "message", "type": "message"},
            }),
            ("function_call_added", {
                "type": "response.output_item.added",
                "item": {"id": "call", "type": "function_call"},
            }),
            ("function_call_done", {
                "type": "response.output_item.done",
                "item": {"id": "call", "type": "function_call"},
            }),
            ("missing_item", {"type": "response.output_item.added"}),
            ("null_item", {"type": "response.output_item.added", "item": None}),
            ("string_item", {"type": "response.output_item.added", "item": "reasoning"}),
            ("list_item", {"type": "response.output_item.done", "item": []}),
            ("missing_item_type", {
                "type": "response.output_item.added", "item": {"id": "missing_type"},
            }),
            ("dict_item_type", {
                "type": "response.output_item.added",
                "item": {"type": {"reasoning": True}},
            }),
            ("list_item_type", {
                "type": "response.output_item.done", "item": {"type": ["reasoning"]},
            }),
            ("integer_item_type", {
                "type": "response.output_item.done", "item": {"type": 7},
            }),
            ("null_event_type", {"type": None, "item": {"type": "reasoning"}}),
            ("list_event_type", {"type": ["response.output_item.added"]}),
            ("dict_event_type", {"type": {"event": "added"}}),
            ("integer_event_type", {"type": 7}),
            ("null_event", None),
            ("list_event", []),
        ]
        for case_name, event in accepted:
            with self.subTest(case=case_name):
                self.assertTrue(proxy_module.is_generation_delta_event(event))
        for case_name, event in rejected:
            with self.subTest(case=case_name):
                self.assertFalse(proxy_module.is_generation_delta_event(event))

    def test_reasoning_activity_does_not_inflate_output_diagnostics(self) -> None:
        """Start the shared stream phase while keeping output delta diagnostics exact."""
        response = FakeResp(b"")
        diagnostics = {
            "started": time.perf_counter(),
            "generation_started": False,
        }
        setattr(response, proxy_module._DIAG_ATTR, diagnostics)

        _record_upstream_event(response, {
            "type": "response.reasoning_summary_text.delta",
        })
        self.assertEqual(diagnostics.get("reasoning_delta_count"), 1)
        summary_at = diagnostics["last_reasoning_delta_monotonic"]
        _record_upstream_event(response, {
            "type": "response.output_item.added",
            "item": {"id": "reasoning_diag", "type": "reasoning"},
        })
        reasoning_added_at = diagnostics["last_generation_at"]
        self.assertTrue(diagnostics["generation_started"])
        self.assertGreater(reasoning_added_at, summary_at)
        self.assertEqual(diagnostics.get("output_delta_count", 0), 0)
        self.assertNotIn("last_output_delta_type", diagnostics)

        _record_upstream_event(response, {
            "type": "response.output_item.done",
            "item": {"id": "reasoning_diag", "type": "reasoning"},
        })
        reasoning_done_at = diagnostics["last_generation_at"]
        self.assertGreaterEqual(reasoning_done_at, reasoning_added_at)
        self.assertEqual(diagnostics.get("output_delta_count", 0), 0)
        self.assertEqual(diagnostics["reasoning_delta_count"], 1)
        self.assertEqual(
            diagnostics["recent_event_types"],
            [
                "response.reasoning_summary_text.delta",
                "response.output_item.added",
                "response.output_item.done",
            ],
        )

        for event_type in (
            "response.output_text.delta",
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        ):
            _record_upstream_event(response, {"type": event_type})
        self.assertEqual(diagnostics["output_delta_count"], 3)
        self.assertEqual(
            diagnostics["last_output_delta_type"],
            "response.custom_tool_call_input.delta",
        )


class ReasoningHTTPTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Verify HTTP reasoning activity renews idle time and silence still expires."""

    async def test_reasoning_activity_outlives_initial_budget(self) -> None:
        """Fold a long reasoning-only HTTP response through the real SSE path."""
        lifecycle = make_reasoning_lifecycle_events("http_reasoning")
        all_events = [
            {"type": "response.created", "response": {
                "id": "resp_http_reasoning", "status": "in_progress",
            }},
            *lifecycle,
            {"type": "response.completed", "response": {
                "id": "resp_http_reasoning",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 17,
                    "total_tokens": 18,
                    "output_tokens_details": {"reasoning_tokens": 17},
                },
            }},
        ]
        chunks = [make_sse(all_events[:2])]
        chunks.extend(make_sse([event]) for event in all_events[2:])
        response = TimedHTTPResponse(chunks, interval_s=REASONING_INTERVAL_S)
        diagnostics = attach_http_timeout_diagnostics(
            response,
            first_byte_timeout_s=INITIAL_BUDGET_S,
            stream_idle_timeout_s=ACTIVE_IDLE_BUDGET_S,
            request_id="http_reasoning_long",
        )
        cfg = Config()
        started = time.perf_counter()
        events = await asyncio.wait_for(
            run_fold(cfg, {"model": "gpt-5.5", "input": []}, response, []),
            timeout=1.0,
        )
        elapsed = time.perf_counter() - started
        terminal = events[-1]
        output = (terminal.get("response") or {}).get("output") or []

        self.assertGreater(elapsed, INITIAL_BUDGET_S)
        self.assertEqual(terminal.get("type"), "response.completed")
        self.assertTrue(any(item.get("type") == "reasoning" for item in output))
        self.assertEqual(diagnostics.get("output_delta_count", 0), 0)
        self.assertTrue(response.closed)

    async def test_reasoning_then_silence_uses_idle_timeout(self) -> None:
        """Convert post-reasoning HTTP silence to the stream idle incomplete reason."""
        first_events = [
            {"type": "response.created", "response": {
                "id": "resp_http_silent", "status": "in_progress",
            }},
            {
                "type": "response.output_item.added",
                "item": {"id": "http_silent_reasoning", "type": "reasoning"},
            },
        ]
        response = TimedHTTPResponse(
            [make_sse(first_events)],
            block_after_chunks=True,
        )
        attach_http_timeout_diagnostics(
            response,
            first_byte_timeout_s=0.5,
            stream_idle_timeout_s=0.03,
            request_id="http_reasoning_silence",
        )
        events = await asyncio.wait_for(
            run_fold(Config(), {"model": "gpt-5.5", "input": []}, response, []),
            timeout=1.0,
        )
        terminal = events[-1]
        reason = (
            ((terminal.get("response") or {}).get("incomplete_details") or {})
            .get("reason")
        )

        self.assertEqual(terminal.get("type"), "response.incomplete")
        self.assertEqual(reason, "stream_idle_timeout")
        self.assertTrue(response.closed)


class ReasoningWSTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Verify native WebSocket reasoning activity shares HTTP timeout semantics."""

    async def test_reasoning_activity_outlives_initial_budget(self) -> None:
        """Consume long reasoning-only activity through a native WebSocket round."""
        events = [
            {"type": "response.created", "response": {
                "id": "resp_ws_reasoning", "status": "in_progress",
            }},
            *make_reasoning_lifecycle_events("ws_reasoning"),
            {"type": "response.completed", "response": {
                "id": "resp_ws_reasoning", "status": "completed",
            }},
        ]
        connection = FakeWebSocketConnection(events, interval_s=REASONING_INTERVAL_S)
        session = UpstreamWebSocketSession(
            connection,
            url="wss://example.test/v1/responses",
            connection_id="ws_reasoning_long",
            timeout_cfg=make_timeout_cfg(INITIAL_BUDGET_S, ACTIVE_IDLE_BUDGET_S),
        )
        started = time.perf_counter()
        response = await session.open_round(
            {"model": "gpt-5.5", "input": []},
            phase="fold_round_1",
            trace_id="ws_reasoning_long",
        )
        try:
            received = await asyncio.wait_for(collect_ws_events(response), timeout=1.0)
            elapsed = time.perf_counter() - started
            diagnostics = getattr(response, proxy_module._DIAG_ATTR)
            self.assertGreater(elapsed, INITIAL_BUDGET_S)
            self.assertEqual(received[-1].get("type"), "response.completed")
            self.assertEqual(diagnostics.get("output_delta_count", 0), 0)
            self.assertEqual(len(received), len(events))
        finally:
            await response.aclose()
            await session.aclose()

    async def test_first_reasoning_item_then_silence_uses_idle_timeout(self) -> None:
        """Start idle timing from either reasoning lifecycle event received first."""
        for first_type in (
            "response.output_item.added",
            "response.output_item.done",
        ):
            with self.subTest(first_type=first_type):
                first_event = {
                    "type": first_type,
                    "item": {
                        "id": f"ws_silent_{first_type.rsplit('.', 1)[-1]}",
                        "type": "reasoning",
                        "encrypted_content": "FAKE_encrypted_content",
                    },
                }
                connection = FakeWebSocketConnection([first_event])
                session = UpstreamWebSocketSession(
                    connection,
                    url="wss://example.test/v1/responses",
                    connection_id=f"ws_silent_{first_type}",
                    timeout_cfg=make_timeout_cfg(0.5, 0.03),
                )
                response = await session.open_round(
                    {"model": "gpt-5.5", "input": []},
                    phase="fold_round_1",
                    trace_id=f"ws_silent_{first_type}",
                )
                try:
                    with self.assertRaises(UpstreamStreamIdleTimeout):
                        await asyncio.wait_for(
                            collect_ws_events(response),
                            timeout=1.0,
                        )
                    self.assertTrue(connection.closed)
                finally:
                    await response.aclose()
                    await session.aclose()
