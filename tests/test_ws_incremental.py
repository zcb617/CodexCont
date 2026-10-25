"""WebSocket 增量输入与隐藏续写历史边界的回归验证。"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_middleware as middleware_tests
from middleware.config import Config
from middleware.proxy import FoldState, fold_stream


def _sse_events(data: bytes) -> list[dict[str, Any]]:
    """读取测试 SSE 的 data 行，供线缆响应构造与断言复用。"""
    events: list[dict[str, Any]] = []
    for line in data.splitlines():
        if line.startswith(b"data:"):
            value = json.loads(line[5:].strip())
            if isinstance(value, dict):
                events.append(value)
    return events


def named_round(
    response_id: str,
    reasoning_tokens: int,
    *,
    encrypted_content: str | None = None,
    extra_items: list[dict[str, Any]] | None = None,
    msg: str | None = None,
) -> bytes:
    """构造可保存终态 output 的响应流，并将该轮原生 ID 统一为指定值。"""
    data = middleware_tests._round(
        f"rs_{response_id}",
        encrypted_content if encrypted_content is not None else f"enc_{response_id}",
        reasoning_tokens,
        extra_items=extra_items,
        msg=msg,
    )
    events = _sse_events(data)
    output_items: list[dict[str, Any]] = []
    for event in events:
        response = event.get("response")
        if isinstance(response, dict):
            response["id"] = response_id
        if event.get("type") == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        if event.get("type") in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            response = event.setdefault("response", {})
            response["output"] = output_items
    return middleware_tests.make_sse(events)


def exchange_ws(
    websocket: Any,
    input_items: list[dict[str, Any]],
    *,
    previous_response_id: str | None = None,
    reasoning: dict[str, Any] | bool | None = None,
) -> list[dict[str, Any]]:
    """提交一轮 WS 请求并收集至终态，供真实路由集成用例复用。"""
    request: dict[str, Any] = {
        "type": "response.create",
        "model": "gpt-5.5",
        "input": input_items,
    }
    if reasoning is not None:
        request["reasoning"] = reasoning
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    websocket.send_json(request)
    events: list[dict[str, Any]] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event.get("type") in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            return events


class GatedFakeResponse:
    """让测试在 created 后暂停上游终态，检查首事件无需历史解析。"""

    def __init__(self, gate: asyncio.Event, first_chunk: bytes, terminal_chunk: bytes):
        self.status_code = 200  # 供 fold 状态机识别为成功的上游响应。
        self.headers: dict[str, str] = {}  # 提供与普通 HTTP 响应一致的头部接口。
        self.gate = gate  # 控制测试何时允许上游继续发送终态数据。
        self.first_chunk = first_chunk  # 只包含立即转发的 response.created 事件。
        self.terminal_chunk = terminal_chunk  # 包含被 gate 延迟的剩余事件与终态。
        self.closed = False  # 记录响应资源是否已按状态机生命周期关闭。

    async def aiter_bytes(self):
        """先发送首 metadata，再等待测试开放上游终态。"""
        yield self.first_chunk
        await self.gate.wait()
        yield self.terminal_chunk

    async def aclose(self) -> None:
        """记录伪响应关闭，维持上游响应的资源接口。"""
        self.closed = True


class WsIncrementalTests(unittest.TestCase):
    """验证正常 WS 请求的增量线缆语义及隐藏续写上下文映射。"""

    def test_normal_turns_keep_delta_with_large_history(self) -> None:
        """确保大历史不会进入禁用、未触发续写或关闭 reasoning 的首包。"""
        history = "H" * (1024 * 1024)
        new_item = {"role": "user", "content": "small delta"}
        cases = (
            ("continuation disabled", False, {"effort": "high"}),
            ("enabled without truncation", True, {"effort": "high"}),
            ("reasoning disabled", True, False),
        )
        for case_name, continuation_enabled, reasoning in cases:
            with self.subTest(case=case_name):
                base = Config()
                cfg = replace(
                    base,
                    cont=replace(base.cont, enabled=continuation_enabled),
                )
                connector = middleware_tests.FakeWsConnector([
                    middleware_tests.FakeResp(
                        named_round("resp_seed", 999, msg="seed answer")
                    ),
                    middleware_tests.FakeResp(
                        named_round("resp_next", 999, msg="next answer")
                    ),
                ])
                app = middleware_tests.create_app(cfg)
                with middleware_tests.TestClient(app) as client:
                    app.state.ws_connector = connector
                    with client.websocket_connect("/v1/responses") as websocket:
                        seed_events = exchange_ws(
                            websocket,
                            [{"role": "user", "content": history}],
                            reasoning=reasoning,
                        )
                        seed_id = (seed_events[-1].get("response") or {}).get("id")
                        exchange_ws(
                            websocket,
                            [new_item],
                            previous_response_id=seed_id,
                            reasoning=reasoning,
                        )

                second_payload = connector.payloads[1]
                self.assertEqual(second_payload.get("input"), [new_item])
                self.assertEqual(second_payload.get("previous_response_id"), "resp_seed")
                self.assertLess(
                    len(json.dumps(second_payload, ensure_ascii=False).encode("utf-8")),
                    4096,
                )
                self.assertEqual(len(connector.sessions), 1)
                self.assertNotIn("continuation", connector.phases)
                self.assertEqual(
                    (seed_events[-1].get("response") or {}).get("id"), "resp_seed"
                )

    def test_hidden_continuation_replays_history_and_maps_next_id(self) -> None:
        """验证历史只进入续写请求，成功后下一轮映射至最后 native ID。"""
        seed_input = {"role": "user", "content": "seed context"}
        seed_tool = {
            "id": "fc_seed",
            "type": "function_call",
            "name": "shell",
            "call_id": "call_seed",
            "arguments": "{}",
        }
        discarded_tool = {
            "id": "fc_discarded",
            "type": "function_call",
            "name": "shell",
            "call_id": "call_discarded",
            "arguments": "{}",
        }
        retained_tool = {
            "id": "fc_retained",
            "type": "function_call",
            "name": "shell",
            "call_id": "call_retained",
            "arguments": "{}",
        }
        current_output = {
            "type": "function_call_output",
            "call_id": "call_seed",
            "output": "seed result",
        }
        next_output = {
            "type": "function_call_output",
            "call_id": "call_retained",
            "output": "retained result",
        }
        cfg = Config()
        connector = middleware_tests.FakeWsConnector([
            middleware_tests.FakeResp(
                named_round("resp_seed", 999, extra_items=[seed_tool])
            ),
            middleware_tests.FakeResp(
                named_round(
                    "resp_visible",
                    516,
                    encrypted_content="ENC_TRUNCATED",
                    extra_items=[discarded_tool],
                    msg="temporary answer",
                )
            ),
            middleware_tests.FakeResp(
                named_round(
                    "resp_hidden",
                    17,
                    encrypted_content="ENC_FINAL",
                    extra_items=[retained_tool],
                )
            ),
            middleware_tests.FakeResp(
                named_round("resp_next", 999, msg="next answer")
            ),
        ])
        app = middleware_tests.create_app(cfg)
        with middleware_tests.TestClient(app) as client:
            app.state.ws_connector = connector
            with client.websocket_connect("/v1/responses") as websocket:
                exchange_ws(websocket, [seed_input])
                folded_events = exchange_ws(
                    websocket,
                    [current_output],
                    previous_response_id="resp_seed",
                )
                next_events = exchange_ws(
                    websocket,
                    [next_output],
                    previous_response_id="resp_visible",
                )

        first_fold_payload = connector.payloads[1]
        continuation_payload = connector.payloads[2]
        next_payload = connector.payloads[3]
        self.assertEqual(first_fold_payload.get("previous_response_id"), "resp_seed")
        self.assertEqual(first_fold_payload.get("input"), [current_output])
        continuation_input = continuation_payload.get("input") or []
        self.assertIn(seed_input, continuation_input)
        self.assertIn(seed_tool, continuation_input)
        self.assertIn(current_output, continuation_input)
        self.assertTrue(
            any(item.get("encrypted_content") == "ENC_TRUNCATED"
                for item in continuation_input if isinstance(item, dict))
        )
        self.assertTrue(
            any(
                item.get("type") == "message"
                and "Continue thinking..." in json.dumps(item)
                for item in continuation_input if isinstance(item, dict)
            )
        )
        self.assertFalse(
            any(
                item.get("call_id") == "call_discarded"
                or "temporary answer" in json.dumps(item)
                for item in continuation_input if isinstance(item, dict)
            )
        )
        self.assertEqual(next_payload.get("previous_response_id"), "resp_hidden")
        self.assertEqual(next_payload.get("input"), [next_output])
        created_ids = [
            (event.get("response") or {}).get("id")
            for event in folded_events if event.get("type") == "response.created"
        ]
        self.assertEqual(created_ids, ["resp_visible"])
        self.assertEqual(
            (folded_events[-1].get("response") or {}).get("id"), "resp_visible"
        )
        self.assertEqual(
            ["fold_round_1", "fold_round_1", "continuation", "fold_round_1"],
            connector.phases,
        )
        self.assertEqual((next_events[-1].get("response") or {}).get("id"), "resp_next")

    def test_failed_continuation_replays_only_reconstructed_context(self) -> None:
        """验证续写上游失败后只回放已保留上下文，不发送废弃输出或失效 ID。"""
        seed_input = {"role": "user", "content": "recoverable seed"}
        current_output = {
            "type": "function_call_output",
            "call_id": "call_seed",
            "output": "seed tool result",
        }
        discarded_tool = {
            "id": "fc_discarded",
            "type": "function_call",
            "name": "shell",
            "call_id": "call_discarded",
            "arguments": "{}",
        }
        seed_tool = {
            "id": "fc_seed",
            "type": "function_call",
            "name": "shell",
            "call_id": "call_seed",
            "arguments": "{}",
        }
        recovery_input = {"role": "user", "content": "continue after failure"}
        cfg = Config()
        connector = middleware_tests.FakeWsConnector([
            middleware_tests.FakeResp(
                named_round("resp_seed", 999, extra_items=[seed_tool])
            ),
            middleware_tests.FakeResp(
                named_round(
                    "resp_visible",
                    516,
                    encrypted_content="ENC_KEEP",
                    extra_items=[discarded_tool],
                    msg="temporary text",
                )
            ),
            middleware_tests.FakeResp(
                b'{"error":{"message":"continuation rejected"}}', status=500
            ),
            middleware_tests.FakeResp(
                named_round("resp_recovered", 999, msg="recovered")
            ),
        ])
        app = middleware_tests.create_app(cfg)
        with middleware_tests.TestClient(app) as client:
            app.state.ws_connector = connector
            with client.websocket_connect("/v1/responses") as websocket:
                exchange_ws(websocket, [seed_input])
                failed_events = exchange_ws(
                    websocket,
                    [current_output],
                    previous_response_id="resp_seed",
                )
                exchange_ws(
                    websocket,
                    [recovery_input],
                    previous_response_id="resp_visible",
                )

        failed_terminal = failed_events[-1]
        self.assertEqual(failed_terminal.get("type"), "response.incomplete")
        self.assertEqual(
            (failed_terminal.get("response") or {}).get("id"), "resp_visible"
        )
        self.assertEqual(
            (failed_terminal.get("response") or {}).get("incomplete_details", {}).get("reason"),
            "upstream_error",
        )
        recovered_payload = connector.payloads[3]
        recovered_input = recovered_payload.get("input") or []
        self.assertNotIn("previous_response_id", recovered_payload)
        self.assertIn(seed_input, recovered_input)
        self.assertIn(current_output, recovered_input)
        self.assertIn(recovery_input, recovered_input)
        self.assertTrue(
            any(item.get("encrypted_content") == "ENC_KEEP"
                for item in recovered_input if isinstance(item, dict))
        )
        self.assertFalse(
            any(
                item.get("call_id") == "call_discarded"
                or "temporary text" in json.dumps(item)
                for item in recovered_input if isinstance(item, dict)
            )
        )

    def test_aliases_do_not_cross_connections(self) -> None:
        """验证 A 连接的 visible 到 native 别名不会影响 B 连接请求。"""
        private_history = {"role": "user", "content": "A private history"}
        b_input = {"role": "user", "content": "B only delta"}
        cfg = Config()
        connector = middleware_tests.FakeWsConnector([
            middleware_tests.FakeResp(
                named_round("resp_a_visible", 516, encrypted_content="ENC_A")
            ),
            middleware_tests.FakeResp(
                named_round("resp_a_native", 17, encrypted_content="ENC_A_FINAL")
            ),
            middleware_tests.FakeResp(
                named_round("resp_b", 999, msg="B answer")
            ),
        ])
        app = middleware_tests.create_app(cfg)
        with middleware_tests.TestClient(app) as client:
            app.state.ws_connector = connector
            with client.websocket_connect("/v1/responses") as websocket_a:
                a_events = exchange_ws(websocket_a, [private_history])
                visible_id = (a_events[-1].get("response") or {}).get("id")
                with client.websocket_connect("/v1/responses") as websocket_b:
                    exchange_ws(
                        websocket_b,
                        [b_input],
                        previous_response_id=visible_id,
                    )

        payload_b = connector.payloads[2]
        self.assertEqual(payload_b.get("previous_response_id"), "resp_a_visible")
        self.assertEqual(payload_b.get("input"), [b_input])
        self.assertNotIn("A private history", json.dumps(payload_b))
        self.assertEqual(len(connector.sessions), 2)
        self.assertEqual(len(connector.calls), 2)
        self.assertNotEqual(
            connector.calls[0].get("connection_id"),
            connector.calls[1].get("connection_id"),
        )


class LazyFoldInputTests(unittest.IsolatedAsyncioTestCase):
    """验证 fold_stream 只在隐藏续写时解析上下文并先转发 created。"""

    async def test_provider_is_lazy_and_resolved_once(self) -> None:
        """无续写不调用历史 provider，多隐藏轮仅解析一次且复用其结果。"""
        cfg = Config()
        provider_calls = 0

        def forbidden_provider() -> list[Any]:
            nonlocal provider_calls
            provider_calls += 1
            raise AssertionError("clean response must not resolve local history")

        clean_client = middleware_tests.FakeClient([])
        clean_state = FoldState()
        clean_stream = fold_stream(
            clean_client,
            cfg,
            {"model": "gpt-5.5", "input": [{"role": "user", "content": "clean"}]},
            {},
            middleware_tests.FakeResp(named_round("resp_clean", 999)),
            continuation_input=forbidden_provider,
            state=clean_state,
        )
        async for _ in clean_stream:
            pass
        self.assertEqual(provider_calls, 0)
        self.assertFalse(clean_state.continued)
        self.assertEqual(clean_state.native_response_id, "resp_clean")

        history_item = {"role": "user", "content": "resolved local history"}
        provider_calls = 0

        def history_provider() -> list[Any]:
            nonlocal provider_calls
            provider_calls += 1
            return [history_item]

        client = middleware_tests.FakeClient([
            middleware_tests.FakeResp(
                named_round("resp_1034", 1034, encrypted_content="ENC_1034")
            ),
            middleware_tests.FakeResp(
                named_round("resp_final", 17, encrypted_content="ENC_FINAL")
            ),
        ])
        state = FoldState()
        stream = fold_stream(
            client,
            cfg,
            {"model": "gpt-5.5", "input": [{"role": "user", "content": "delta"}]},
            {},
            middleware_tests.FakeResp(
                named_round("resp_516", 516, encrypted_content="ENC_516")
            ),
            continuation_input=history_provider,
            state=state,
        )
        async for _ in stream:
            pass
        self.assertEqual(provider_calls, 1)
        self.assertEqual(len(client.payloads), 2)
        self.assertTrue(
            all(history_item in (payload.get("input") or []) for payload in client.payloads)
        )
        self.assertTrue(state.continued)
        self.assertEqual(state.native_response_id, "resp_final")

    async def test_created_is_forwarded_before_terminal(self) -> None:
        """确认 created 在终态 gate 开启前送出，且不触发历史 provider。"""
        events = _sse_events(named_round("resp_created", 999))
        gate = asyncio.Event()
        response = GatedFakeResponse(
            gate,
            middleware_tests.make_sse(events[:1]),
            middleware_tests.make_sse(events[1:]),
        )
        provider_calls = 0

        def provider() -> list[Any]:
            nonlocal provider_calls
            provider_calls += 1
            return []

        stream = fold_stream(
            middleware_tests.FakeClient([]),
            Config(),
            {"model": "gpt-5.5", "input": [{"role": "user", "content": "delta"}]},
            {},
            response,
            continuation_input=provider,
            state=FoldState(),
        )
        first_chunk = await asyncio.wait_for(stream.__anext__(), timeout=1)
        first_events = await middleware_tests.parse_events(first_chunk)
        self.assertEqual(first_events[0].get("type"), "response.created")
        self.assertFalse(gate.is_set())
        self.assertEqual(provider_calls, 0)

        gate.set()
        try:
            while True:
                await asyncio.wait_for(stream.__anext__(), timeout=1)
        except StopAsyncIteration:
            pass
        self.assertTrue(response.closed)

