from abc import ABC, abstractmethod
from typing import Any

from app.jobs.context import JobContext
from app.queue import JobRecord, JobStage


class StageResult(dict):
    @property
    def next_stage(self) -> JobStage | str | None:
        return self.get("next_stage")

    @property
    def artifacts(self) -> dict[str, Any]:
        return self.get("artifacts", {})


class JobAdapter(ABC):
    job_types: set[str] = set()

    @abstractmethod
    async def download(self, job: JobRecord, context: JobContext) -> StageResult:
        raise NotImplementedError

    @abstractmethod
    async def analyze(self, job: JobRecord, context: JobContext) -> StageResult:
        raise NotImplementedError

    @abstractmethod
    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        raise NotImplementedError

    async def run_stage(self, job: JobRecord, context: JobContext) -> StageResult:
        handler = getattr(self, job.stage.value, None)
        if handler is None:
            raise RuntimeError(f"adapter {type(self).__name__} does not support stage {job.stage.value}")
        return await handler(job, context)
