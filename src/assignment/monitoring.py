"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def observe(
        self,
        *,
        total: int = 0,
        blocked: int = 0,
        rate_limit: int = 0,
        judge_check: int = 0,
        judge_fail: int = 0,
    ) -> None:
        """Cộng dồn bộ đếm sau mỗi request (nhiều tham số cùng lúc được phép)."""
        self.total_requests += total
        self.blocked_requests += blocked
        self.rate_limit_hits += rate_limit
        self.judge_checks += judge_check
        self.judge_fails += judge_fail

    def _upsert_alert(
        self, metric: str, value: float, threshold: float, message: str
    ) -> Alert:
        """Thêm alert, hoặc cập nhật alert cùng ``metric`` đã có.

        ``check_metrics()`` có thể được gọi nhiều lần; nếu cứ ``append`` thì
        danh sách alert phình vô hạn và báo cáo trở nên vô nghĩa. Giữ 1 alert
        cho mỗi chỉ số, cập nhật giá trị mới nhất.
        """
        existing = next((a for a in self.alerts if a.metric == metric), None)
        if existing is not None:
            existing.value = value
            existing.threshold = threshold
            existing.message = message
            return existing
        alert = Alert(
            metric=metric, value=value, threshold=threshold, message=message
        )
        self.alerts.append(alert)
        return alert

    def check_metrics(self) -> list[Alert]:
        """Tính tỉ lệ và tạo ``Alert`` khi vượt ngưỡng.

        Trả về danh sách alert **vừa được cập nhật** trong lần gọi này
        (không phải toàn bộ lịch sử) để pipeline chỉ ghi cái mới.
        """
        snap = self.snapshot()
        fired: list[Alert] = []

        # 1. Tỉ lệ bị chặn — cao bất thường nghĩa là đang có tấn công
        #    (hoặc chính sách đang chặn nhầm — cũng cần người xem).
        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            fired.append(
                self._upsert_alert(
                    "block_rate",
                    snap["block_rate"],
                    self.block_rate_threshold,
                    f"{snap['blocked_requests']}/{self.total_requests} request bị chặn "
                    f"({snap['block_rate']:.0%}) > ngưỡng {self.block_rate_threshold:.0%}",
                )
            )

        # 2. Số lần dính rate limit — dấu hiệu spam / flood.
        if self.rate_limit_hits > self.rate_limit_hit_threshold:
            fired.append(
                self._upsert_alert(
                    "rate_limit_hits",
                    float(self.rate_limit_hits),
                    float(self.rate_limit_hit_threshold),
                    f"{self.rate_limit_hits} lần bị rate limit "
                    f"> ngưỡng {self.rate_limit_hit_threshold} "
                    "— nghi vấn spam hoặc flood",
                )
            )

        # 3. Tỉ lệ judge bác — chỉ có ý nghĩa khi judge thực sự được bật.
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            fired.append(
                self._upsert_alert(
                    "judge_fail_rate",
                    snap["judge_fail_rate"],
                    self.judge_fail_rate_threshold,
                    f"{self.judge_fails}/{self.judge_checks} phản hồi bị judge "
                    f"đánh dấu UNSAFE ({snap['judge_fail_rate']:.0%}) "
                    f"> ngưỡng {self.judge_fail_rate_threshold:.0%}",
                )
            )

        return fired

    def export_json(self, filepath: str | None = None):
        """Ghi metrics + alerts ra JSON.

        Args:
            filepath: đường dẫn đích. Bỏ trống dùng ``outputs/metrics.json``.

        Returns:
            ``path`` dạng str sau khi ghi.
        """
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        payload["exported_at"] = datetime.now(timezone.utc).isoformat()
        payload["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
