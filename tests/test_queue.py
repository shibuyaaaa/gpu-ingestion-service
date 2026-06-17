import tempfile
import time
from pathlib import Path

import pytest

from app.queue import JobStage, JobStatus, JobStore, JobType, QueueFull


def test_enqueue_is_idempotent_by_job_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        first, created_first = store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})
        second, created_second = store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})

        assert created_first is True
        assert created_second is False
        assert first.id == second.id == "a"
        assert first.job_type == JobType.BULK_DISSECT
        assert first.stage == JobStage.DOWNLOAD
        assert first.status == JobStatus.QUEUED
        assert store.stats()["active_depth"] == 1


def test_queue_depth_rejects_new_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=1)
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})

        with pytest.raises(QueueFull):
            store.enqueue({"job_id": "b", "job_type": "bulk_dissect", "source": "song"})


def test_claim_complete_and_retry_flow():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})

        claimed = store.claim_next([JobStage.DOWNLOAD], "worker-1")
        assert claimed is not None
        assert claimed.id == "a"
        assert claimed.status == JobStatus.PROCESSING
        assert claimed.attempts == 0

        store.complete_stage("a", next_stage=JobStage.ANALYZE, artifacts={"audio_path": "/tmp/a.wav"})
        claimed_analysis = store.claim_next([JobStage.ANALYZE], "gpu-1")
        assert claimed_analysis is not None
        assert claimed_analysis.artifacts["audio_path"] == "/tmp/a.wav"
        assert claimed_analysis.attempts == 0

        failed = store.fail_stage("a", "temporary")
        assert failed.status == JobStatus.QUEUED
        assert failed.error == "temporary"
        assert failed.attempts == 1


def test_fail_stage_respects_max_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"}, max_attempts=2)

        store.claim_next([JobStage.DOWNLOAD], "worker-1")
        first = store.fail_stage("a", "temporary", retry_delay_seconds=0)
        assert first.status == JobStatus.QUEUED
        assert first.attempts == 1

        store.claim_next([JobStage.DOWNLOAD], "worker-1")
        second = store.fail_stage("a", "still broken", retry_delay_seconds=0)
        assert second.status == JobStatus.FAILED
        assert second.attempts == 2


def test_claim_batch_respects_limit():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        for idx in range(3):
            store.enqueue({"job_id": f"job-{idx}", "job_type": "bulk_dissect", "source": "song"})

        claimed = store.claim_batch([JobStage.DOWNLOAD], "worker-1", limit=2)

        assert [job.id for job in claimed] == ["job-0", "job-1"]
        assert store.get("job-2").status == JobStatus.QUEUED


def test_claim_batch_prioritizes_higher_priority_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song", "priority": 0})
        store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song", "priority": 100})

        claimed = store.claim_batch([JobStage.DOWNLOAD], "worker-1", limit=2)

        assert [job.id for job in claimed] == ["quick", "bulk"]


def test_quick_process_job_preempts_bulk_process_backlog_after_current_claim():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=20)
        for idx in range(5):
            store.enqueue({"job_id": f"bulk-{idx}", "job_type": "bulk_dissect", "source": "song"})
            store.complete_stage(f"bulk-{idx}", next_stage=JobStage.PROCESS)

        in_flight_bulk = store.claim_next([JobStage.PROCESS], "process-worker-1")
        assert in_flight_bulk is not None
        assert in_flight_bulk.id == "bulk-0"

        store.enqueue({"job_id": "quick-late", "job_type": "quick_dissect", "source": "song"})
        store.complete_stage("quick-late", next_stage=JobStage.PROCESS)

        next_claim = store.claim_next([JobStage.PROCESS], "process-worker-2")

        assert next_claim is not None
        assert next_claim.id == "quick-late"
        assert next_claim.priority == 100
        assert store.get("bulk-1").status == JobStatus.QUEUED


def test_claim_batch_is_fifo_for_same_priority_across_stages():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "analysis-job", "job_type": "bulk_dissect", "source": "song", "priority": 50})
        time.sleep(0.001)
        store.enqueue({"job_id": "process-job", "job_type": "bulk_dissect", "source": "song", "priority": 50})
        store.complete_stage("analysis-job", next_stage=JobStage.ANALYZE)
        store.complete_stage("process-job", next_stage=JobStage.PROCESS)

        claimed = store.claim_batch([JobStage.PROCESS, JobStage.ANALYZE], "gpu-1", limit=2)

        assert [job.id for job in claimed] == ["analysis-job", "process-job"]


def test_job_type_default_priorities_are_applied():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        quick, _ = store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song"})
        bulk, _ = store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song"})

        assert quick.priority == 100
        assert bulk.priority == 10


def test_recover_stale_processing_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})
        claimed = store.claim_next([JobStage.DOWNLOAD], "worker-1")
        assert claimed is not None

        recovered = store.recover_stale_processing(lease_timeout_seconds=0)

        assert recovered == 1
        job = store.get("a")
        assert job.status == JobStatus.QUEUED
        assert job.worker_id is None


def test_continuation_is_inserted_directly_at_process_stage():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song"})

        continuation = store.enqueue_continuation(
            parent_job=parent,
            payload={"source": "song"},
            artifacts={"audio_path": "/tmp/a.wav", "skip_segment_ids": ["seg-1"]},
        )

        assert continuation.job_type == JobType.BULK_DISSECT
        assert continuation.stage == JobStage.PROCESS
        assert continuation.priority == 10
        assert continuation.payload["parent_job_id"] == "quick"
        assert continuation.artifacts["skip_segment_ids"] == ["seg-1"]


def test_invalid_stage_in_sqlite_row_fails_clearly():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        path = Path(tmp) / "queue.sqlite3"
        with store._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, job_type, stage, status, payload_json, artifacts_json,
                    attempts, max_attempts, priority, created_at, updated_at, available_at
                )
                VALUES ('bad', 'bulk_dissect', 'not_a_stage', 'queued', '{}', '{}', 0, 3, 0, 1, 1, 1)
                """
            )

        with pytest.raises(ValueError):
            store.get("bad")
