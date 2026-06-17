from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.legacy.db import DBClient
from app.legacy.utils import GCSClient
from app.models import ModelRuntimeBundle
from app.queue import JobRecord, JobStore


@dataclass
class JobContext:
    settings: Settings
    store: JobStore
    models: ModelRuntimeBundle
    gcs: GCSClient
    db: DBClient

    def job_work_dir(self, job: JobRecord) -> Path:
        target = self.settings.work_dir / "jobs" / job.id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def merged_payload(self, job: JobRecord) -> dict[str, Any]:
        return {**job.payload, "_artifacts": job.artifacts}

