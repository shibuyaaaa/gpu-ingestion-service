import asyncio
import logging
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.jobs import JobRegistry
from app.jobs.context import JobContext
from app.legacy.audio import AudioOps
from app.legacy.utils.source import DownloadError, youtube_auth_status
from app.queue import JobRecord, JobStage

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    worker_id: str
    stages: list[JobStage]
    batch_size: int
    claim_mode: str = "stages"
    active_job_ids: list[str] = None
    active_job_started_at: dict[str, float] = None
    processed: int = 0
    failures: int = 0
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.active_job_ids is None:
            self.active_job_ids = []
        if self.active_job_started_at is None:
            self.active_job_started_at = {}

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {
            "worker_id": self.worker_id,
            "stages": [stage.value for stage in self.stages],
            "batch_size": self.batch_size,
            "claim_mode": self.claim_mode,
            "active_job_ids": self.active_job_ids,
            "active_job_ages_seconds": {
                job_id: round(max(0.0, now - started_at), 3)
                for job_id, started_at in self.active_job_started_at.items()
            },
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
        self._cleanup_state: dict[str, Any] = {
            "enabled": context.settings.work_dir_cleanup_enabled,
            "runs": 0,
            "dirs_removed": 0,
            "bytes_removed": 0,
            "last_run_at": None,
            "last_error": None,
        }
        self._gpu_health_state: dict[str, Any] = {
            "enabled": context.settings.gpu_health_restart_enabled and not context.settings.dry_run_mode,
            "checks": 0,
            "consecutive_failures": 0,
            "last_check_at": None,
            "last_error": None,
            "restart_threshold": context.settings.gpu_health_restart_failures,
        }
        self._job_watchdog_state: dict[str, Any] = {
            "enabled": context.settings.job_lease_timeout_seconds > 0,
            "checks": 0,
            "last_check_at": None,
            "last_error": None,
            "timeout_seconds": context.settings.job_lease_timeout_seconds,
        }
        self._youtube_auth_recovery_state: dict[str, Any] = {
            "enabled": context.settings.youtube_auth_recovery_enabled,
            "checks": 0,
            "last_check_at": None,
            "last_error": None,
            "last_refreshed_epoch": None,
            "last_requeued": 0,
            "total_requeued": 0,
        }

    async def start(self) -> None:
        if self._tasks:
            return
        recovered = self.context.store.recover_processing_after_restart(
            error="recovered processing job after service startup"
        )
        if recovered:
            logger.warning("recovered %d processing jobs on startup", recovered)
        await self.context.models.warmup()
        for idx in range(self.context.settings.download_workers):
            self._start_worker(f"download-{idx}", [JobStage.DOWNLOAD], self.context.settings.download_batch_size)
        self._start_worker("gpu-0", [JobStage.ANALYZE, JobStage.PROCESS], 1, claim_mode="gpu")
        for idx in range(self.context.settings.process_workers):
            self._start_worker(
                f"process-cpu-{idx}",
                [JobStage.PROCESS],
                self.context.settings.process_batch_size,
                claim_mode="cpu_process",
            )
        if self.context.settings.work_dir_cleanup_enabled:
            self._tasks.append(asyncio.create_task(self._cleanup_loop(), name="work-dir-cleanup"))
        if self._gpu_health_state["enabled"]:
            self._tasks.append(asyncio.create_task(self._gpu_health_loop(), name="gpu-health"))
        if self._job_watchdog_state["enabled"]:
            self._tasks.append(asyncio.create_task(self._job_watchdog_loop(), name="job-watchdog"))
        if self._youtube_auth_recovery_state["enabled"]:
            self._tasks.append(asyncio.create_task(self._youtube_auth_recovery_loop(), name="youtube-auth-recovery"))
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

    def _start_worker(
        self,
        name: str,
        stages: list[JobStage],
        batch_size: int,
        *,
        claim_mode: str = "stages",
    ) -> None:
        worker_id = f"{socket.gethostname()}:{name}:{uuid.uuid4().hex[:8]}"
        state = WorkerState(
            worker_id=worker_id,
            stages=stages,
            batch_size=max(1, batch_size),
            claim_mode=claim_mode,
        )
        self._states[worker_id] = state
        self._tasks.append(asyncio.create_task(self._worker_loop(state), name=name))

    async def _worker_loop(self, state: WorkerState) -> None:
        while not self._stop_event.is_set():
            if state.claim_mode == "gpu":
                jobs = self.context.store.claim_gpu_batch(state.worker_id, limit=state.batch_size)
            elif state.claim_mode == "cpu_process":
                jobs = self.context.store.claim_cpu_process_batch(state.worker_id, limit=state.batch_size)
            else:
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
        state.active_job_started_at[job.id] = time.time()
        adapter = self.registry.get(job.job_type)
        try:
            result = await adapter.run_stage(job, self.context)
            self.context.store.complete_stage(
                job.id,
                next_stage=result.next_stage,
                artifacts=result.artifacts,
            )
            if job.stage == JobStage.PROCESS and result.artifacts.get("fanout_maybe_complete"):
                root_job_id = str(job.payload.get("root_job_id") or "")
                if root_job_id and root_job_id != job.id:
                    self.context.store.reconcile_failed_fanout_parent(root_job_id)
            state.processed += 1
            state.last_error = None
        except Exception as exc:
            logger.exception("job %s stage %s failed", job.id, job.stage)
            self.context.store.fail_stage(
                job.id,
                _job_error_text(exc),
                retry_delay_seconds=self._retry_delay_seconds(job),
            )
            state.failures += 1
            state.last_error = _job_error_text(exc)
        finally:
            state.active_job_ids = [job_id for job_id in state.active_job_ids if job_id != job.id]
            state.active_job_started_at.pop(job.id, None)

    def _retry_delay_seconds(self, job: JobRecord) -> int:
        if job.stage == JobStage.DOWNLOAD:
            return max(0, self.context.settings.download_retry_delay_seconds)
        return max(0, self.context.settings.default_retry_delay_seconds)

    async def _cleanup_loop(self) -> None:
        while not self._stop_event.is_set():
            await self._cleanup_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(1.0, self.context.settings.work_dir_cleanup_interval_seconds),
                )
            except asyncio.TimeoutError:
                pass

    async def _cleanup_once(self) -> None:
        self._cleanup_state["runs"] += 1
        self._cleanup_state["last_run_at"] = time.time()
        try:
            work_dirs = self.context.store.inactive_work_dirs(
                older_than_seconds=self.context.settings.work_dir_cleanup_min_age_seconds,
                limit=self.context.settings.work_dir_cleanup_max_dirs_per_run,
            )
            removed_count = 0
            removed_bytes = 0
            for work_dir in work_dirs:
                path = Path(work_dir)
                if not path.exists():
                    continue
                if not self._cleanup_path_allowed(path):
                    logger.warning("skipping cleanup outside work dir: %s", path)
                    continue
                size = await asyncio.to_thread(_directory_size, path)
                await asyncio.to_thread(shutil.rmtree, path, True)
                removed_count += 1
                removed_bytes += size
            self._cleanup_state["dirs_removed"] += removed_count
            self._cleanup_state["bytes_removed"] += removed_bytes
            self._cleanup_state["last_error"] = None
            if removed_count:
                logger.info("cleaned %d inactive work dirs freeing %.2f GiB", removed_count, removed_bytes / (1024**3))
        except Exception as exc:
            logger.exception("work dir cleanup failed")
            self._cleanup_state["last_error"] = str(exc)

    async def _gpu_health_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(5.0, self.context.settings.gpu_health_check_interval_seconds),
                )
                continue
            except asyncio.TimeoutError:
                pass
            status = await asyncio.to_thread(self.context.models.status)
            gpu = status.get("gpu", {})
            self._gpu_health_state["checks"] += 1
            self._gpu_health_state["last_check_at"] = time.time()
            if gpu.get("available"):
                self._gpu_health_state["consecutive_failures"] = 0
                self._gpu_health_state["last_error"] = None
                continue
            self._gpu_health_state["consecutive_failures"] += 1
            self._gpu_health_state["last_error"] = gpu.get("error") or "GPU unavailable"
            logger.error(
                "GPU health check failed %s/%s: %s",
                self._gpu_health_state["consecutive_failures"],
                self._gpu_health_state["restart_threshold"],
                self._gpu_health_state["last_error"],
            )
            if self._gpu_health_state["consecutive_failures"] >= self._gpu_health_state["restart_threshold"]:
                logger.critical("exiting process so systemd can refresh container GPU runtime")
                os._exit(75)

    async def _job_watchdog_loop(self) -> None:
        timeout = max(1.0, float(self.context.settings.job_lease_timeout_seconds))
        interval = min(60.0, max(5.0, timeout / 6.0))
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                continue
            except asyncio.TimeoutError:
                pass
            self._job_watchdog_state["checks"] += 1
            self._job_watchdog_state["last_check_at"] = time.time()
            overdue = self._overdue_active_jobs(now=self._job_watchdog_state["last_check_at"])
            if not overdue:
                self._job_watchdog_state["last_error"] = None
                continue
            self._job_watchdog_state["last_error"] = f"{len(overdue)} active job(s) exceeded lease timeout"
            logger.critical(
                "exiting process because active job exceeded lease timeout: %s",
                overdue,
            )
            os._exit(76)

    async def _youtube_auth_recovery_loop(self) -> None:
        interval = max(30.0, float(self.context.settings.youtube_auth_recovery_interval_seconds))
        while not self._stop_event.is_set():
            await self._youtube_auth_recovery_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _youtube_auth_recovery_once(self) -> None:
        self._youtube_auth_recovery_state["checks"] += 1
        self._youtube_auth_recovery_state["last_check_at"] = time.time()
        try:
            status = await asyncio.to_thread(youtube_auth_status)
            refreshed_epoch = status.get("extracted_epoch")
            self._youtube_auth_recovery_state["last_refreshed_epoch"] = refreshed_epoch
            if not refreshed_epoch or not status.get("has_cookies"):
                self._youtube_auth_recovery_state["last_requeued"] = 0
                self._youtube_auth_recovery_state["last_error"] = "YouTube auth payload has no refreshed cookies"
                return
            requeued = await asyncio.to_thread(
                self.context.store.recover_failed_download_auth_jobs,
                refreshed_after=float(refreshed_epoch),
                limit=max(1, int(self.context.settings.youtube_auth_recovery_batch_size)),
            )
            self._youtube_auth_recovery_state["last_requeued"] = requeued
            self._youtube_auth_recovery_state["total_requeued"] += requeued
            self._youtube_auth_recovery_state["last_error"] = None
            if requeued:
                logger.warning("requeued %d failed YouTube auth download job(s) after cookie refresh", requeued)
        except Exception as exc:
            logger.exception("YouTube auth recovery failed")
            self._youtube_auth_recovery_state["last_error"] = str(exc)

    def _overdue_active_jobs(self, *, now: float | None = None) -> list[dict[str, Any]]:
        current_time = time.time() if now is None else now
        timeout = max(1.0, float(self.context.settings.job_lease_timeout_seconds))
        overdue: list[dict[str, Any]] = []
        for state in self._states.values():
            for job_id, started_at in state.active_job_started_at.items():
                age = max(0.0, current_time - started_at)
                if age <= timeout:
                    continue
                overdue.append(
                    {
                        "worker_id": state.worker_id,
                        "job_id": job_id,
                        "age_seconds": round(age, 3),
                        "timeout_seconds": timeout,
                    }
                )
        return overdue

    def _cleanup_path_allowed(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            work_root = self.context.settings.work_dir.resolve()
            return resolved.is_dir() and resolved.is_relative_to(work_root)
        except Exception:
            return False

    def state(self) -> dict[str, Any]:
        return {
            "accepting": self.accepting,
            "workers": [state.to_dict() for state in self._states.values()],
            "running_tasks": len([task for task in self._tasks if not task.done()]),
            "work_dir_cleanup": {
                **self._cleanup_state,
                "bytes_removed_gib": round(float(self._cleanup_state["bytes_removed"]) / (1024**3), 3),
                "interval_seconds": self.context.settings.work_dir_cleanup_interval_seconds,
                "min_age_seconds": self.context.settings.work_dir_cleanup_min_age_seconds,
            },
            "gpu_health": self._gpu_health_state,
            "job_watchdog": {
                **self._job_watchdog_state,
                "overdue_active_jobs": self._overdue_active_jobs(),
            },
            "youtube_auth_recovery": self._youtube_auth_recovery_state,
            "scheduling": {
                "download_workers": self.context.settings.download_workers,
                "download_batch_size": self.context.settings.download_batch_size,
                "gpu_workers": 1,
                "gpu_batch_size": 1,
                "gpu_stages": [JobStage.PROCESS.value, JobStage.ANALYZE.value],
                "process_cpu_workers": self.context.settings.process_workers,
                "process_cpu_batch_size": self.context.settings.process_batch_size,
                "ffmpeg_concurrency": self.context.settings.ffmpeg_concurrency,
                "ffmpeg_runtime": AudioOps.runtime_status(),
                "worker_poll_seconds": self.context.settings.worker_poll_seconds,
                "retry_delay_seconds": {
                    "download": self.context.settings.download_retry_delay_seconds,
                    "default": self.context.settings.default_retry_delay_seconds,
                },
                "claim_order": ["priority_desc", "created_at_asc"],
                "stages": [stage.value for stage in JobStage],
            },
        }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _job_error_text(exc: Exception) -> str:
    if isinstance(exc, DownloadError):
        return f"{exc.category}: {exc}"
    return str(exc)
