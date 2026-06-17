import asyncio
import tempfile
from pathlib import Path

from app.config import Settings
from app.jobs import build_default_registry
from app.jobs.context import JobContext
from app.legacy.db import DBClient
from app.legacy.utils import GCSClient
from app.models import ModelRuntimeBundle
from app.queue import JobStatus, JobStore, JobType
from app.workers import WorkerManager


async def test_worker_completes_local_dry_run_job():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            queue_db_path=root / "queue.sqlite3",
            work_dir=root / "work",
            max_total_queue_depth=10,
            download_workers=1,
            download_batch_size=1,
            process_workers=1,
            process_batch_size=1,
            analyze_batch_size=1,
            dry_run_mode=True,
            start_workers=False,
        )
        store = JobStore(settings.queue_db_path, max_depth=settings.max_total_queue_depth)
        models = ModelRuntimeBundle.from_settings(settings)
        context = JobContext(
            settings=settings,
            store=store,
            models=models,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
        )
        manager = WorkerManager(context=context, registry=build_default_registry())
        store.enqueue(
            {
                "job_id": "dry-run-1",
                "job_type": "bulk_dissect",
                "source": "Daft Punk One More Time",
            }
        )

        await manager.start()
        try:
            for _ in range(80):
                job = store.get("dry-run-1")
                if job and job.status == "completed":
                    break
                await asyncio.sleep(0.1)
        finally:
            await manager.stop()

        job = store.get("dry-run-1")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.artifacts["final_outputs"]["dry_run"] is True
        assert sorted(job.artifacts["processed_segments"]) == ["seg-0", "seg-1", "seg-2"]
        assert "analysis" in job.artifacts
        assert job.artifacts["final_outputs"]["processed_segment_ids"] == ["seg-0", "seg-1", "seg-2"]


async def test_quick_dissect_stems_only_chorus_segment():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            queue_db_path=root / "queue.sqlite3",
            work_dir=root / "work",
            max_total_queue_depth=10,
            download_workers=1,
            download_batch_size=1,
            process_workers=1,
            process_batch_size=1,
            analyze_batch_size=1,
            dry_run_mode=True,
            start_workers=False,
        )
        store = JobStore(settings.queue_db_path, max_depth=settings.max_total_queue_depth)
        models = ModelRuntimeBundle.from_settings(settings)
        context = JobContext(
            settings=settings,
            store=store,
            models=models,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
        )
        manager = WorkerManager(context=context, registry=build_default_registry())
        store.enqueue(
            {
                "job_id": "quick-1",
                "job_type": "quick_dissect",
                "source": "The Weeknd Blinding Lights",
            }
        )

        await manager.start()
        try:
            for _ in range(80):
                job = store.get("quick-1")
                if job and job.status == "completed":
                    break
                await asyncio.sleep(0.1)
        finally:
            await manager.stop()

        job = store.get("quick-1")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert "analysis" in job.artifacts
        assert "quick_chorus_result" in job.artifacts
        assert job.artifacts["final_outputs"]["quick_dissect"] is True
        continuation_id = job.artifacts["bulk_continuation_job_id"]
        continuation = store.get(continuation_id)
        assert continuation is not None
        assert continuation.job_type == JobType.BULK_DISSECT
        assert continuation.payload["parent_job_id"] == "quick-1"
        assert continuation.artifacts["skip_segment_ids"] == [job.artifacts["chorus_segment"]["id"]]
