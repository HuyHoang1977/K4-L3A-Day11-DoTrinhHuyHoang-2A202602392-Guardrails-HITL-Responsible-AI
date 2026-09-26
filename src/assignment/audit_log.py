"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    #: Cắt bớt nội dung lưu để nhật ký không phình vô hạn theo thời gian.
    preview_limit = 400

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    @staticmethod
    def _preview(text) -> str:
        text = "" if text is None else str(text)
        return text if len(text) <= AuditLogPlugin.preview_limit else (
            text[: AuditLogPlugin.preview_limit] + "…[truncated]"
        )

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Mở một entry nhật ký và ghi nhận thời điểm bắt đầu.

        Args:
            user_id: ai gửi câu hỏi.
            text: nội dung hỏi (chỉ lưu bản rút gọn).
            request_id: khoá để ``record_output`` ghi tiếp. Bỏ trống sẽ tự sinh.

        Returns:
            ``request_id`` để truyền lại cho :meth:`record_output`.
        """
        rid = request_id or f"req-{len(self.logs) + 1:04d}"
        self._open[rid] = time.perf_counter()
        # Ghi đè nếu request_id bị dùng lại (retries) — tránh nhân bản entry.
        self.logs = [e for e in self.logs if e.get("request_id") != rid]
        self.logs.append(
            {
                "request_id": rid,
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input": self._preview(text),
                "input_length": len(text or ""),
                "response": "",
                "response_preview": "",
                "blocked": False,
                "layer": None,
                "reason": None,
                "latency_ms": None,
                "status": "pending",
            }
        )
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        reason: str | None = None,
    ):
        """Đóng entry tương ứng: lưu phản hồi, quyết định của lớp chặn, latency.

        Args:
            user_id: ai nhận phản hồi.
            text: nội dung trả lời đã qua guardrail (KHÔNG phải bản gốc).
            blocked: lớp bảo vệ có chặn không.
            layer: tên lớp đã quyết định (``input_guardrail`` / ``rate_limiter`` …).
            request_id: khoá lấy từ :meth:`record_input`.
            reason: mã lý do để điều tra nhanh hơn đọc message.

        Returns:
            Entry vừa cập nhật, hoặc ``None`` nếu không tìm thấy ``request_id``.
        """
        rid = request_id or (self.logs[-1]["request_id"] if self.logs else None)
        entry = next(
            (e for e in reversed(self.logs) if e.get("request_id") == rid), None
        )
        if entry is None:
            # Không có record_input đi kèm (ví dụ lỗi trước khi mở entry):
            # vẫn ghi lại để không mất dấu vết của sự cố.
            entry = {
                "request_id": rid or f"req-{len(self.logs) + 1:04d}",
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input": "",
                "input_length": 0,
            }
            self.logs.append(entry)

        started = self._open.pop(entry["request_id"], None)
        entry["response"] = self._preview(text)
        entry["response_preview"] = self._preview(text)
        entry["blocked"] = bool(blocked)
        entry["layer"] = layer
        entry["reason"] = reason
        entry["status"] = "blocked" if blocked else "ok"
        if started is not None:
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return entry

    def export_json(self, filepath: str | None = None):
        """Ghi toàn bộ nhật ký ra đĩa (JSON array).

        Args:
            filepath: đường dẫn đích. Bỏ trống dùng ``outputs/audit_log.json``.

        Returns:
            ``path`` dạng str sau khi ghi.
        """
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
