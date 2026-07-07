"""Optional SQLite logging for outbound upstream payloads."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger("middleware.payload_log")


class SQLitePayloadLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._schema_ready = False

    @classmethod
    def from_config(cls, cfg: Config) -> "SQLitePayloadLogger | None":
        raw = cfg.log.payload_sqlite_path.strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = cfg.root / path
        return cls(path)

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        try:
            db.execute(
                """
                create table if not exists outbound_payloads (
                    id integer primary key autoincrement,
                    created_at text not null,
                    transport text not null,
                    phase text not null,
                    url text not null,
                    status_code integer,
                    model text,
                    payload_json text not null
                )
                """
            )
            db.execute(
                "create index if not exists idx_outbound_payloads_created_at "
                "on outbound_payloads(created_at)"
            )
            db.commit()
        finally:
            db.close()
        self._schema_ready = True

    def initialize(self) -> None:
        try:
            self._ensure_schema()
            log.info("outbound payload sqlite logging enabled: %s", self.path)
        except Exception as exc:
            log.warning("failed to initialize outbound payload sqlite log: %r", exc)

    def record(
        self,
        *,
        transport: str,
        phase: str,
        url: str,
        payload: Any,
        status_code: int | None,
    ) -> None:
        try:
            self._ensure_schema()
            model = payload.get("model") if isinstance(payload, dict) else None
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            created_at = datetime.now(timezone.utc).isoformat()
            db = sqlite3.connect(self.path)
            try:
                db.execute(
                    """
                    insert into outbound_payloads (
                        created_at, transport, phase, url, status_code, model, payload_json
                    ) values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (created_at, transport, phase, url, status_code, model, payload_json),
                )
                db.commit()
                log.info(
                    "recorded outbound payload: transport=%s phase=%s status=%s model=%s",
                    transport,
                    phase,
                    status_code,
                    model,
                )
            finally:
                db.close()
        except Exception as exc:
            log.warning("failed to write outbound payload sqlite log: %r", exc)
