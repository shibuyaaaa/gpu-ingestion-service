import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.jobs import build_default_registry
from app.jobs.adapters import BulkDissectAdapter, _should_skip_segment_processing
from app.jobs.base import StageResult
from app.jobs.context import JobContext
from app.library_membership import LibraryMembershipChecker
from app.library_writer import LibraryWriter
from app.legacy.db import DBClient
from app.legacy.utils.source import _yt_dlp_js_args
from app.legacy.utils import GCSClient
from app.models import ModelRuntimeBundle
from app.queue import JobStage, JobStatus, JobStore, JobType
from app.workers import WorkerManager


def test_segment_normalization_skips_short_boundary_segments():
    segments = BulkDissectAdapter._normalize_segments(
        {
            "duration": 60.0,
            "segments": [
                {"label": "start", "start": 0.0, "end": 0.83},
                {"label": "verse", "start": 0.83, "end": 18.0},
                {"label": "end", "start": 58.7, "end": 60.0},
            ],
        }
    )

    assert segments == [{"id": "seg-1", "start": 0.83, "end": 18.0, "label": "verse"}]


def test_existing_tiny_fanout_segments_are_skipped():
    assert _should_skip_segment_processing({"label": "start", "start": 0.0, "end": 0.83}) is True
    assert _should_skip_segment_processing({"label": "verse", "start": 0.83, "end": 18.0}) is False


def test_yt_dlp_js_runtime_enables_ejs_remote_component(monkeypatch):
    def fake_which(binary: str) -> str | None:
        return "/usr/local/bin/deno" if binary == "deno" else None

    monkeypatch.setattr("app.legacy.utils.source.shutil.which", fake_which)

    assert _yt_dlp_js_args() == [
        "--js-runtimes",
        "deno:/usr/local/bin/deno",
        "--remote-components",
        "ejs:github",
    ]


def test_worker_uses_short_retry_delay_for_download_failures():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            queue_db_path=root / "queue.sqlite3",
            work_dir=root / "work",
            default_retry_delay_seconds=31,
            download_retry_delay_seconds=4,
            start_workers=False,
        )
        store = JobStore(settings.queue_db_path, max_depth=10)
        job, _ = store.enqueue({"job_id": "retry-policy", "job_type": "quick_dissect", "source": "song"})
        manager = WorkerManager(context=SimpleNamespace(settings=settings), registry=None)

        assert manager._retry_delay_seconds(job) == 4
        assert manager._retry_delay_seconds(replace(job, stage=JobStage.ANALYZE)) == 31
        assert manager.state()["scheduling"]["retry_delay_seconds"] == {"download": 4, "default": 31}
        assert manager.state()["scheduling"]["ffmpeg_concurrency"] == settings.ffmpeg_concurrency
        assert manager.state()["scheduling"]["worker_poll_seconds"] == settings.worker_poll_seconds


async def test_worker_reconciles_parent_only_when_fanout_maybe_complete():
    class FakeAdapter:
        def __init__(self, *, fanout_maybe_complete: bool):
            self.fanout_maybe_complete = fanout_maybe_complete

        async def run_stage(self, job, context):
            return StageResult(
                next_stage=None,
                artifacts={"fanout_maybe_complete": self.fanout_maybe_complete},
            )

    class FakeRegistry:
        def __init__(self, adapter):
            self.adapter = adapter

        def get(self, job_type):
            return self.adapter

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            queue_db_path=root / "queue.sqlite3",
            work_dir=root / "work",
            dry_run_mode=True,
            start_workers=False,
        )
        store = JobStore(settings.queue_db_path, max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        child = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-1:other",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": parent.id},
            artifacts={"process_mode": "segment_other", "segment_id": "seg-1"},
            priority=100,
        )
        claimed = store.claim_next([JobStage.PROCESS], "process-worker")
        calls = []

        def fake_reconcile(root_job_id):
            calls.append(root_job_id)
            return None

        store.reconcile_failed_fanout_parent = fake_reconcile
        context = SimpleNamespace(settings=settings, store=store)
        state = SimpleNamespace(active_job_ids=[], processed=0, failures=0, last_error=None)

        manager = WorkerManager(context=context, registry=FakeRegistry(FakeAdapter(fanout_maybe_complete=False)))
        await manager._process_job_stage(claimed, state)
        assert calls == []

        store.retry(child.id)
        claimed = store.claim_next([JobStage.PROCESS], "process-worker")
        manager = WorkerManager(context=context, registry=FakeRegistry(FakeAdapter(fanout_maybe_complete=True)))
        await manager._process_job_stage(claimed, state)

        assert calls == ["root"]


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
        db = DBClient(database_url="")
        library = LibraryMembershipChecker(db=db, settings=settings)
        context = JobContext(
            settings=settings,
            store=store,
            models=models,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=db,
            library=library,
            library_writer=LibraryWriter(db=db, settings=settings, membership=library),
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
                if store.stats()["active_depth"] == 0:
                    break
                await asyncio.sleep(0.1)
        finally:
            await manager.stop()

        job = store.get("dry-run-1")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.artifacts["final_outputs"]["dry_run"] is True
        assert "analysis" in job.artifacts
        assert job.artifacts["fanout"]["child_count"] == 3
        assert store.child_summary("dry-run-1") == {"total": 6, "active": 0, "completed": 6, "failed": 0}
        for segment_id in ("seg-0", "seg-1", "seg-2"):
            chord = store.get(f"dry-run-1:bulk:{segment_id}:chord")
            other = store.get(f"dry-run-1:bulk:{segment_id}:other")
            assert chord is not None
            assert other is not None
            assert chord.status == JobStatus.COMPLETED
            assert other.status == JobStatus.COMPLETED
            assert chord.artifacts["final_outputs"]["chord_outputs"]
            assert other.artifacts["final_outputs"]["stem_outputs"]
            assert chord.artifacts["analysis"]["duration"] > 0
            assert "segments" not in chord.artifacts
            assert "chorus_segment" not in chord.artifacts
            assert "full_stem_urls" not in chord.artifacts
            assert other.artifacts["analysis"]["duration"] > 0
            assert "segments" not in other.artifacts
            assert "chorus_segment" not in other.artifacts
            assert "full_stem_urls" not in other.artifacts


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
        db = DBClient(database_url="")
        library = LibraryMembershipChecker(db=db, settings=settings)
        context = JobContext(
            settings=settings,
            store=store,
            models=models,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=db,
            library=library,
            library_writer=LibraryWriter(db=db, settings=settings, membership=library),
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
                if store.stats()["active_depth"] == 0:
                    break
                await asyncio.sleep(0.1)
        finally:
            await manager.stop()

        job = store.get("quick-1")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert "analysis" in job.artifacts
        assert job.artifacts["final_outputs"]["status"] == "process_fanout_enqueued"

        chorus_id = job.artifacts["chorus_segment"]["id"]
        quick_chord = store.get(f"quick-1:quick:{chorus_id}:chord")
        quick_other = store.get(f"quick-1:quick:{chorus_id}:other")
        assert quick_chord is not None
        assert quick_other is not None
        assert quick_chord.job_type == JobType.QUICK_DISSECT
        assert quick_chord.priority == 400
        assert quick_chord.artifacts["final_outputs"]["quick_dissect_confirmation"] is True
        assert quick_other.priority == 200

        for segment_id in ("seg-0", "seg-1"):
            bulk_chord = store.get(f"quick-1:bulk:{segment_id}:chord")
            bulk_other = store.get(f"quick-1:bulk:{segment_id}:other")
            assert bulk_chord is not None
            assert bulk_other is not None
            assert bulk_chord.job_type == JobType.BULK_DISSECT
            assert bulk_chord.priority == 300
            assert bulk_other.priority == 100
        assert store.child_summary("quick-1") == {"total": 6, "active": 0, "completed": 6, "failed": 0}
