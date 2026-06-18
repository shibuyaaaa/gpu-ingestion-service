import asyncio
import logging
import socket
import uuid
from dataclasses import dataclass
from typing import Any

from app.jobs import JobRegistry
from app.jobs.context import JobContext
from app.queue import JobRecord, JobStage

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    worker_id: str
    stages: list[JobStage]
    batch_size: int
    active_job_ids: list[str] = None
    processed: int = 0
    failures: int = 0
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.active_job_ids is None:
            self.active_job_ids = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "stages": [stage.value for stage in self.stages],
            "batch_size": self.batch_size,
            "active_job_ids": self.active_job_ids,
            "processed": self.processed,
            "failures": self.failures,
            "last_error": self.last_error,
        }


class WorkerManager:
    def __init__(self, *, context: JobContext, registry: JobRegistry):
        self.context = context
        self.registry = registry
        self.accepting = True
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._states: dict[str, WorkerState] = {}

    async def start(self) -> None:
        if self._tasks:
            return
        recovered = self.context.store.recover_stale_processing(
            lease_timeout_seconds=self.context.settings.job_lease_timeout_seconds
        )
        if recovered:
            logger.warning("requeued %d stale processing jobs on startup", recovered)
        await self.context.models.warmup()
        for idx in range(self.context.settings.download_workers):
            self._start_worker(f"download-{idx}", [JobStage.DOWNLOAD], self.context.settings.download_batch_size)
        self._start_worker("analyze-0", [JobStage.ANALYZE], self.context.settings.analyze_batch_size)
        for idx in range(self.context.settings.process_workers):
            self._start_worker(
                f"process-{idx}",
                [JobStage.PROCESS],
                self.context.settings.process_batch_size,
            )
        logger.info("started %d local workers", len(self._tasks))

    async def stop(self) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def drain(self) -> None:
        self.accepting = False

    def _start_worker(self, name: str, stages: list[JobStage], batch_size: int) -> None:
        worker_id = f"{socket.gethostname()}:{name}:{uuid.uuid4().hex[:8]}"
        state = WorkerState(worker_id=worker_id, stages=stages, batch_size=max(1, batch_size))
        self._states[worker_id] = state
        self._tasks.append(asyncio.create_task(self._worker_loop(state), name=name))

    async def _worker_loop(self, state: WorkerState) -> None:
        while not self._stop_event.is_set():
            jobs = self.context.store.claim_batch(
                state.stages,
                state.worker_id,
                limit=state.batch_size,
            )
            if not jobs:
                await asyncio.sleep(self.context.settings.worker_poll_seconds)
                continue
            for job in jobs:
                await self._process_job_stage(job, state)

    async def _process_job_stage(self, job: JobRecord, state: WorkerState) -> None:
        state.active_job_ids.append(job.id)
        adapter = self.registry.get(job.job_type)
        try:
            result = await adapter.run_stage(job, self.context)
            self.context.store.complete_stage(
                job.id,
                next_stage=result.next_stage,
                artifacts=result.artifacts,
            )
            if job.stage == JobStage.PROCESS:
                root_job_id = str(job.payload.get("root_job_id") or "")
                if root_job_id and root_job_id != job.id:
                    self.context.store.reconcile_failed_fanout_parent(root_job_id)
            state.processed += 1
            state.last_error = None
        except Exception as exc:
            logger.exception("job %s stage %s failed", job.id, job.stage)
            self.context.store.fail_stage(job.id, str(exc))
            state.failures += 1
            state.last_error = str(exc)
        finally:
            state.active_job_ids = [job_id for job_id in state.active_job_ids if job_id != job.id]

    def state(self) -> dict[str, Any]:
        return {
            "accepting": self.accepting,
            "workers": [state.to_dict() for state in self._states.values()],
            "running_tasks": len([task for task in self._tasks if not task.done()]),
            "scheduling": {
                "download_batch_size": self.context.settings.download_batch_size,
                "analyze_batch_size": self.context.settings.analyze_batch_size,
                "process_batch_size": self.context.settings.process_batch_size,
                "claim_order": ["priority_desc", "created_at_asc"],
                "stages": [stage.value for stage in JobStage],
            },
        }
