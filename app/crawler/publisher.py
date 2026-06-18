from typing import Any

import httpx


class LocalIngestionPublisher:
    """Submit crawler jobs to the VM-local ingestion API.

    The crawler remains a separate producer process, but it does not write the
    queue database directly. The FastAPI service validates and persists jobs.
    """

    def __init__(self, *, ingestion_url: str, timeout_seconds: float = 30.0):
        if not ingestion_url:
            raise RuntimeError("CRAWLER_INGESTION_URL must be set")
        self.ingestion_url = ingestion_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def publish(self, payload: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.ingestion_url}/jobs", json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("job_id") or payload.get("job_id") or "")
