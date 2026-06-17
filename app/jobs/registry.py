from app.jobs.adapters import BulkDissectAdapter, QuickDissectAdapter
from app.jobs.base import JobAdapter
from app.queue import JobType


class UnsupportedJobType(ValueError):
    """Raised when a message names a job type this service does not handle."""


class JobRegistry:
    def __init__(self, adapters: list[JobAdapter]):
        self.adapters = adapters
        self._by_type: dict[str, JobAdapter] = {}
        for adapter in adapters:
            for job_type in adapter.job_types:
                self._by_type[job_type] = adapter

    def get(self, job_type: JobType | str) -> JobAdapter:
        try:
            normalized = JobType(str(job_type)).value
        except ValueError as exc:
            raise UnsupportedJobType(f"unsupported job_type: {job_type}") from exc
        adapter = self._by_type.get(normalized)
        if adapter is None:
            raise UnsupportedJobType(f"unsupported job_type: {job_type}")
        return adapter

    def supports(self, job_type: JobType | str) -> bool:
        try:
            return JobType(str(job_type)).value in self._by_type
        except ValueError:
            return False


def build_default_registry() -> JobRegistry:
    return JobRegistry(
        [
            QuickDissectAdapter(),
            BulkDissectAdapter(),
        ]
    )
