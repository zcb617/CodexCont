"""验证 fold 重建终态时保留上游 response.error。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


# 测试仓库根目录，供直接运行本文件时导入 middleware 包。
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from middleware.config import load_config
from middleware.proxy import _reconstruct_terminal


class TerminalErrorTests(unittest.TestCase):
    """验证终态重建对 response.error 的字段保留业务。"""

    def test_failed_terminal_preserves_error(self):
        """验证 response.failed 的完整上游错误字典传递到下游。"""
        base = {"id": "resp_first", "status": "in_progress", "error": None}
        error = {
            "type": "server_error",
            "code": "server_error",
            "message": "request failed",
            "param": None,
            "request_id": "upstream_diag",
        }
        terminal = {
            "type": "response.failed",
            "response": {
                "id": "resp_first",
                "status": "failed",
                "error": error,
            },
        }

        result = _reconstruct_terminal(terminal, base, [], {}, 1, [], None, {})

        self.assertEqual(result["response"]["error"], error)
        self.assertEqual(result["type"], "response.failed")
        self.assertEqual(result["response"]["status"], "failed")

    def test_continuation_failure_preserves_first_response_identity(self):
        """验证续接失败仍使用首轮响应身份并保留终态错误。"""
        base = {
            "id": "resp_first",
            "status": "in_progress",
            "created_at": 100,
            "error": None,
        }
        error = {
            "type": "server_error",
            "code": "server_error",
            "message": "request failed",
            "param": None,
            "request_id": "upstream_diag",
        }
        terminal = {
            "type": "response.failed",
            "response": {
                "id": "resp_second",
                "created_at": 200,
                "status": "failed",
                "error": error,
            },
        }

        result = _reconstruct_terminal(terminal, base, [], {}, 1, [], None, {})

        self.assertEqual(result["response"]["id"], "resp_first")
        self.assertEqual(result["response"]["error"], error)

    def test_explicit_null_error_overrides_base_error(self):
        """验证终态显式 error:null 会覆盖首轮残留错误。"""
        base = {"id": "resp_first", "status": "in_progress", "error": {"code": "stale"}}
        terminal = {
            "type": "response.completed",
            "response": {
                "id": "resp_first",
                "status": "completed",
                "error": None,
            },
        }

        result = _reconstruct_terminal(terminal, base, [], {}, 1, [], None, {})

        self.assertIn("error", result["response"])
        self.assertIsNone(result["response"]["error"])
        self.assertEqual(result["response"]["status"], "completed")

    def test_terminal_without_error_keeps_existing_behavior(self):
        """验证终态缺少 error 键时继续保留首轮响应错误。"""
        base = {"id": "resp_first", "status": "in_progress", "error": {"code": "existing"}}
        terminal = {
            "type": "response.completed",
            "response": {"id": "resp_first", "status": "completed"},
        }

        result = _reconstruct_terminal(terminal, base, [], {}, 1, [], None, {})

        self.assertEqual(result["response"]["error"], {"code": "existing"})


class FoldTerminalErrorTests(unittest.IsolatedAsyncioTestCase):
    """验证离线 fold 流程保留上游终态错误业务。"""

    async def test_fold_failed_stream_preserves_error(self):
        """验证 response.failed SSE 终态错误完整输出且只产生一个终态。"""
        # 复用既有离线测试夹具，保持 fold 流程与生产调用路径一致。
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_middleware import ROOT, FakeResp, make_sse, run_fold

        cfg = load_config(ROOT / "config.toml")
        error = {
            "type": "server_error",
            "code": "server_error",
            "message": "request failed",
            "param": None,
            "request_id": "upstream_diag",
        }
        response = FakeResp(make_sse([
            {
                "type": "response.created",
                "response": {
                    "id": "resp_failed_diag",
                    "status": "in_progress",
                    "error": None,
                },
            },
            {
                "type": "response.failed",
                "response": {
                    "id": "resp_failed_diag",
                    "status": "failed",
                    "error": error,
                },
            },
        ]))

        events = await run_fold(cfg, {"model": "gpt-5.5", "input": []}, response, [])
        terminal_events = [event for event in events
                           if isinstance(event, dict) and event.get("type") == "response.failed"]

        self.assertEqual(len(terminal_events), 1)
        self.assertEqual(terminal_events[0]["response"]["error"], error)


if __name__ == "__main__":
    unittest.main()
