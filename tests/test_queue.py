import tempfile
import time
import uuid
from pathlib import Path

import pytest

from app.jobs.adapters import (
    PROCESS_PRIORITY_BULK_CHORD,
    PROCESS_PRIORITY_BULK_OTHER,
    PROCESS_PRIORITY_QUICK_CHORD,
    PROCESS_PRIORITY_QUICK_OTHER,
)
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


def test_complete_stage_serializes_uuid_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        song_id = uuid.uuid4()
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})
        store.claim_next([JobStage.DOWNLOAD], "worker-1")

        completed = store.complete_stage("a", next_stage=None, artifacts={"library_song_id": song_id})

        assert completed.status == JobStatus.COMPLETED
        assert completed.artifacts["library_song_id"] == str(song_id)


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


def test_process_claim_order_prioritizes_chords_before_other_stems():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "quick_dissect", "source": "song"})
        children = [
            ("bulk-other", JobType.BULK_DISSECT, PROCESS_PRIORITY_BULK_OTHER),
            ("quick-other", JobType.QUICK_DISSECT, PROCESS_PRIORITY_QUICK_OTHER),
            ("bulk-chord", JobType.BULK_DISSECT, PROCESS_PRIORITY_BULK_CHORD),
            ("quick-chord", JobType.QUICK_DISSECT, PROCESS_PRIORITY_QUICK_CHORD),
        ]
        for job_id, job_type, priority in children:
            store.enqueue_process_child(
                parent_job=parent,
                child_id=job_id,
                job_type=job_type,
                payload={"source": "song", "root_job_id": "root"},
                artifacts={"process_mode": "segment_chord", "segment_id": job_id},
                priority=priority,
            )

        claimed = store.claim_batch([JobStage.PROCESS], "process-worker", limit=4)

        assert [job.id for job in claimed] == ["quick-chord", "bulk-chord", "quick-other", "bulk-other"]


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
        assert next_claim.priority == 350
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


def test_gpu_claim_order_prioritizes_quick_work_over_bulk_analysis():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        bulk, _ = store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song"})
        quick, _ = store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song"})
        store.complete_stage(bulk.id, next_stage=JobStage.ANALYZE)
        store.complete_stage(quick.id, next_stage=JobStage.ANALYZE)

        claimed = store.claim_batch([JobStage.PROCESS, JobStage.ANALYZE], "gpu-worker", limit=2)

        assert [job.id for job in claimed] == ["quick", "bulk"]
        assert claimed[0].priority > PROCESS_PRIORITY_BULK_CHORD


def test_gpu_claim_order_prioritizes_quick_chord_over_bulk_analysis():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        bulk, _ = store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song"})
        quick, _ = store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song"})
        store.complete_stage(bulk.id, next_stage=JobStage.ANALYZE)
        store.enqueue_process_child(
            parent_job=quick,
            child_id="quick:quick:seg-1:chord",
            job_type=JobType.QUICK_DISSECT,
            payload={"source": "song", "root_job_id": "quick"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-1"},
            priority=PROCESS_PRIORITY_QUICK_CHORD,
        )

        claimed = store.claim_batch([JobStage.PROCESS, JobStage.ANALYZE], "gpu-worker", limit=2)

        assert [job.id for job in claimed] == ["quick:quick:seg-1:chord", "bulk"]


def test_job_type_default_priorities_are_applied():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        quick, _ = store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song"})
        bulk, _ = store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song"})

        assert quick.priority == 350
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


def test_recover_processing_after_restart_counts_as_attempt():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"})
        claimed = store.claim_next([JobStage.DOWNLOAD], "worker-1")
        assert claimed is not None

        recovered = store.recover_processing_after_restart(error="crashed")

        assert recovered == 1
        job = store.get("a")
        assert job.status == JobStatus.QUEUED
        assert job.worker_id is None
        assert job.attempts == 1
        assert job.error == "crashed"


def test_recover_processing_after_restart_fails_at_max_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "a", "job_type": "bulk_dissect", "source": "song"}, max_attempts=1)
        claimed = store.claim_next([JobStage.DOWNLOAD], "worker-1")
        assert claimed is not None

        recovered = store.recover_processing_after_restart(error="crashed")

        assert recovered == 1
        job = store.get("a")
        assert job.status == JobStatus.FAILED
        assert job.attempts == 1
        assert job.error == "crashed"


def test_inactive_work_dirs_returns_only_terminal_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        old_dir = str(Path(tmp) / "old")
        active_dir = str(Path(tmp) / "active")
        old, _ = store.enqueue(
            {"job_id": "old", "job_type": "bulk_dissect", "source": "song"},
            initial_artifacts={"work_dir": old_dir},
        )
        active, _ = store.enqueue(
            {"job_id": "active", "job_type": "bulk_dissect", "source": "song"},
            initial_artifacts={"work_dir": active_dir},
        )
        store.claim_next([JobStage.DOWNLOAD], "worker-1")
        store.complete_stage(old.id, next_stage=None)

        dirs = store.inactive_work_dirs(older_than_seconds=0, limit=10)

        assert old_dir in dirs
        assert active_dir not in dirs


def test_inactive_work_dirs_keeps_terminal_parent_with_active_child():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        work_dir = str(Path(tmp) / "shared")
        parent, _ = store.enqueue(
            {"job_id": "root", "job_type": "bulk_dissect", "source": "song"},
            initial_artifacts={"work_dir": work_dir},
        )
        store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"work_dir": work_dir, "process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.claim_next([JobStage.DOWNLOAD], "worker-1")
        store.complete_stage(parent.id, next_stage=None)

        dirs = store.inactive_work_dirs(older_than_seconds=0, limit=10)

        assert work_dir not in dirs


def test_reconcile_failed_fanout_parent_when_children_completed():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        child = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.claim_next([JobStage.PROCESS], "process-worker")
        store.complete_stage(child.id, next_stage=None)
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        store.fail_stage("root", "old fanout failure", retry_delay_seconds=0)
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        store.fail_stage("root", "old fanout failure", retry_delay_seconds=0)
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        failed = store.fail_stage("root", "old fanout failure", retry_delay_seconds=0)
        assert failed.status == JobStatus.FAILED

        reconciled = store.reconcile_failed_fanout_parent("root")

        assert reconciled is not None
        assert reconciled.status == JobStatus.COMPLETED
        assert reconciled.error is None
        assert reconciled.artifacts["fanout_reconciled"]["completed_children"] == 1


def test_reconcile_failed_fanout_parent_waits_for_active_children():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        store.fail_stage("root", "old fanout failure", retry_delay_seconds=0)
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        store.fail_stage("root", "old fanout failure", retry_delay_seconds=0)
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        failed = store.fail_stage("root", "old fanout failure", retry_delay_seconds=0)

        reconciled = store.reconcile_failed_fanout_parent("root")

        assert failed.status == JobStatus.FAILED
        assert reconciled is not None
        assert reconciled.status == JobStatus.FAILED


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
