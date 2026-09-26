"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        #: Lý do chặn của lần gần nhất — CP3 đưa vào audit log.
        self.last_block: dict | None = None

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]
        self.last_block = None

        # 1. Sliding window: bỏ các mốc thời gian đã trôi ra ngoài cửa sổ.
        #    deque giữ thứ tự tăng dần nên chỉ cần pop ở đầu (O(1)).
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        # 2. Đã đủ lượt trong cửa sổ → chặn, KHÔNG ghi thêm mốc thời gian.
        #    (Ghi thêm sẽ làm cửa sổ phình vô hạn khi bị spam.)
        if len(window) >= self.max_requests:
            wait = max(1.0, self.window_seconds - (now - window[0]))
            self.blocked_count += 1
            self.last_block = {
                "layer": "rate_limiter",
                "reason": "rate_limit_exceeded",
                "user_id": user_id,
                "window_used": len(window),
                "max_requests": self.max_requests,
                "retry_after_seconds": round(wait, 2),
            }
            return self._block_response(
                f"Rate limit exceeded. Try again in {wait:.0f}s."
            )

        # 3. Còn trong hạn → ghi mốc thời gian và cho qua.
        window.append(now)
        return None

    def remaining(self, user_id: str = "anonymous") -> int:
        """Số lượt còn lại cho ``user_id`` (dùng cho metrics / debug)."""
        window = self.user_windows.get(user_id)
        if not window:
            return self.max_requests
        cutoff = time.time() - self.window_seconds
        return max(0, self.max_requests - sum(1 for ts in window if ts > cutoff))

    def reset(self, user_id: str | None = None) -> None:
        """Xoá cửa sổ của một user (hoặc tất cả) — dùng cho test."""
        if user_id is None:
            self.user_windows.clear()
        else:
            self.user_windows.pop(user_id, None)
