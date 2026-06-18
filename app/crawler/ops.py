from dataclasses import dataclass
from typing import Any

import httpx


class IngestionOpsUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class JobTerminalState:
    status: str
    root_status: str | None
    child_summary: dict[str, Any]
    error: str | None = None

    @property
    def terminal(self) -> bool:
        if self.status in {"completed", "failed"}:
            return True
        return False


class IngestionOpsClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def job_state(self, job_id: str) -> JobTerminalState:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                job_response = await client.get(f"{self.base_url}/ops/jobs/{job_id}")
                if job_response.status_code == 404:
                    return JobTerminalState(status="pending_delivery", root_status=None, child_summary={})
                job_response.raise_for_status()
                job = job_response.json()
                root_status = str(job.get("status") or "")

                child_response = await client.get(f"{self.base_url}/ops/jobs/{job_id}/children-summary")
                child_summary = {}
                if child_response.status_code != 404:
                    child_response.raise_for_status()
                    child_summary = child_response.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise IngestionOpsUnavailable(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {502, 503, 504}:
                raise IngestionOpsUnavailable(str(exc)) from exc
            raise

        active_children = int(child_summary.get("active", 0) or 0)
        failed_children = int(child_summary.get("failed", 0) or 0)
        total_children = int(child_summary.get("total", 0) or 0)
        if root_status == "failed" and total_children == 0:
            return JobTerminalState(
                status="failed",
                root_status=root_status,
                child_summary=child_summary,
                error=job.get("error"),
            )
        if root_status in {"completed", "failed"} and total_children > 0 and active_children == 0:
            return JobTerminalState(
                status="failed" if failed_children else "completed",
                root_status=root_status,
                child_summary=child_summary,
                error="one or more child jobs failed" if failed_children else None,
            )
        if root_status == "completed" and active_children == 0:
            return JobTerminalState(
                status="completed",
                root_status=root_status,
                child_summary=child_summary,
            )
        return JobTerminalState(status="running", root_status=root_status, child_summary=child_summary)
