import sqlite3
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


def test_claim_batch_returns_processing_records_and_writes_events():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "job-0", "job_type": "bulk_dissect", "source": "song"})
        store.enqueue({"job_id": "job-1", "job_type": "bulk_dissect", "source": "song"})

        claimed = store.claim_batch([JobStage.DOWNLOAD], "worker-1", limit=2)

        assert [job.id for job in claimed] == ["job-0", "job-1"]
        assert all(job.status == JobStatus.PROCESSING for job in claimed)
        assert all(job.worker_id == "worker-1" for job in claimed)
        for job in claimed:
            events = store.recent_events(job.id, limit=5)
            claimed_events = [event for event in events if event["event_type"] == "claimed"]
            assert claimed_events
            assert claimed_events[0]["data"] == {"stage": "download", "worker_id": "worker-1"}


def test_claim_batch_prioritizes_higher_priority_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song", "priority": 0})
        store.enqueue({"job_id": "quick", "job_type": "quick_dissect", "source": "song", "priority": 100})

        claimed = store.claim_batch([JobStage.DOWNLOAD], "worker-1", limit=2)

        assert [job.id for job in claimed] == ["quick", "bulk"]


def test_claim_query_uses_scheduling_order_index_without_temp_sort():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=20)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        for idx in range(5):
            store.enqueue_process_child(
                parent_job=parent,
                child_id=f"root:bulk:seg-{idx}:chord",
                job_type=JobType.BULK_DISSECT,
                payload={"source": "song", "root_job_id": "root"},
                artifacts={"process_mode": "segment_chord", "segment_id": f"seg-{idx}", "requires_gpu": False},
                priority=PROCESS_PRIORITY_BULK_CHORD,
            )

        with sqlite3.connect(store.db_path) as conn:
            plan = conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM jobs
                WHERE status = ?
                  AND available_at <= ?
                  AND stage = ?
                  AND COALESCE(json_extract(artifacts_json, '$.requires_gpu'), 1) = 0
                ORDER BY priority DESC,
                         created_at ASC,
                         id ASC
                LIMIT ?
                """,
                (JobStatus.QUEUED.value, time.time(), JobStage.PROCESS.value, 1),
            ).fetchall()

        detail = " ".join(str(row[-1]) for row in plan)
        assert "idx_jobs_claim" in detail
        assert "USE TEMP B-TREE" not in detail


def test_active_depth_query_uses_partial_active_index():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=20)
        active, _ = store.enqueue({"job_id": "active", "job_type": "bulk_dissect", "source": "song"})
        done, _ = store.enqueue({"job_id": "done", "job_type": "bulk_dissect", "source": "song"})
        store.claim_next([JobStage.DOWNLOAD], "worker-1")
        store.complete_stage(active.id, next_stage=None)
        store.claim_next([JobStage.DOWNLOAD], "worker-1")
        store.complete_stage(done.id, next_stage=None)
        store.enqueue({"job_id": "queued", "job_type": "bulk_dissect", "source": "song"})

        with sqlite3.connect(store.db_path) as conn:
            plan = conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT COUNT(*) FROM jobs
                WHERE status NOT IN ('completed', 'failed')
                """
            ).fetchall()

        detail = " ".join(str(row[-1]) for row in plan)
        assert "idx_jobs_active_depth" in detail


def test_recent_timing_summary_aggregates_analyze_and_process_timings():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        download, _ = store.enqueue(
            {"job_id": "download", "job_type": "bulk_dissect", "source": "song"},
            initial_stage=JobStage.DOWNLOAD,
        )
        analyze, _ = store.enqueue(
            {"job_id": "analyze", "job_type": "bulk_dissect", "source": "song"},
            initial_stage=JobStage.ANALYZE,
        )
        process, _ = store.enqueue(
            {"job_id": "process", "job_type": "bulk_dissect", "source": "song"},
            initial_stage=JobStage.PROCESS,
            initial_artifacts={"process_mode": "segment_other", "segment_id": "seg-1"},
        )
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        store.complete_stage(
            download.id,
            next_stage=None,
            artifacts={
                "download_timings": {
                    "source_metadata_seconds": 0.5,
                    "library_precheck_seconds": 0.25,
                    "youtube_match_seconds": 1.0,
                    "youtube_download_seconds": 2.0,
                    "wav_copy_seconds": 0.1,
                    "download_total_seconds": 3.85,
                }
            },
        )
        store.claim_next([JobStage.ANALYZE], "gpu")
        store.complete_stage(
            analyze.id,
            next_stage=None,
            artifacts={
                "analysis_timings": {
                    "allin1_analyze_seconds": 10.0,
                    "demix_total_seconds": 7.0,
                    "demix_apply_seconds": 5.0,
                    "demix_save_seconds": 2.0,
                    "demix_segment_seconds": 7.5,
                    "demix_segment_configured_seconds": 15.0,
                    "demix_segment_max_seconds": 7.5,
                    "gpu_sample_count": 4,
                    "gpu_utilization_avg_pct": 61.5,
                    "gpu_memory_used_max_mb": 12000,
                    "stem_count": 4,
                }
            },
        )
        store.claim_next([JobStage.PROCESS], "cpu")
        store.complete_stage(
            process.id,
            next_stage=None,
            artifacts={
                "final_outputs": {"process_mode": "segment_other", "segment_id": "seg-1"},
                "segment_result": {
                    "timings": {
                        "stem_segment_extract_seconds": 1.25,
                        "upload_and_publish_seconds": 0.75,
                        "gpu_sample_count": 2,
                        "gpu_utilization_avg_pct": 42.0,
                        "gpu_memory_used_max_mb": 8000,
                    }
                },
            },
        )

        summary = store.recent_timing_summary(limit=10)

        assert summary["completed_jobs_sampled"] == 3
        assert summary["download"]["count"] == 1
        assert summary["download"]["avg_seconds"]["source_metadata_seconds"] == 0.5
        assert summary["download"]["avg_seconds"]["library_precheck_seconds"] == 0.25
        assert summary["download"]["avg_seconds"]["youtube_match_seconds"] == 1.0
        assert summary["download"]["avg_seconds"]["youtube_download_seconds"] == 2.0
        assert summary["download"]["avg_seconds"]["download_total_seconds"] == 3.85
        assert summary["download"]["avg_seconds"]["queue_wait_seconds"] is not None
        assert summary["download"]["avg_seconds"]["processing_seconds"] is not None
        assert summary["analyze"]["count"] == 1
        assert summary["analyze"]["avg_seconds"]["demix_save_seconds"] == 2.0
        assert summary["analyze"]["avg_seconds"]["demix_segment_seconds"] == 7.5
        assert summary["analyze"]["avg_seconds"]["demix_segment_configured_seconds"] == 15.0
        assert summary["analyze"]["avg_seconds"]["demix_segment_max_seconds"] == 7.5
        assert summary["analyze"]["avg_seconds"]["gpu_utilization_avg_pct"] == 61.5
        assert summary["analyze"]["avg_seconds"]["gpu_memory_used_max_mb"] == 12000.0
        assert summary["analyze"]["avg_seconds"]["processing_seconds"] is not None
        assert summary["process_by_mode"]["segment_other"]["count"] == 1
        assert summary["process_by_mode"]["segment_other"]["avg_seconds"]["stem_segment_extract_seconds"] == 1.25
        assert summary["process_by_mode"]["segment_other"]["avg_seconds"]["gpu_utilization_avg_pct"] == 42.0
        assert summary["process_by_mode"]["segment_other"]["avg_seconds"]["gpu_memory_used_max_mb"] == 8000.0
        assert summary["process_by_mode"]["segment_other"]["avg_seconds"]["queue_wait_seconds"] is not None
        assert summary["process_by_mode"]["segment_other"]["avg_seconds"]["processing_seconds"] is not None
        assert [item["job_id"] for item in summary["latest"]] == ["process", "analyze", "download"]
        assert summary["latest"][0]["queue_wait_seconds"] is not None
        assert summary["latest"][0]["processing_seconds"] is not None
        assert summary["latest"][0]["duration_seconds"] >= summary["latest"][0]["processing_seconds"]
        assert summary["latest"][1]["queue_wait_seconds"] is not None
        assert summary["latest"][1]["processing_seconds"] is not None
        assert summary["latest"][2]["timings"]["youtube_download_seconds"] == 2.0


def test_recent_timing_summary_caps_limit():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)

        summary = store.recent_timing_summary(limit=5000)

        assert summary["limit"] == 1000
        assert summary["completed_jobs_sampled"] == 0


def test_recent_timing_summary_keeps_download_timings_after_stage_progression():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        job, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        store.claim_next([JobStage.DOWNLOAD], "download-worker")
        store.complete_stage(
            job.id,
            next_stage=JobStage.ANALYZE,
            artifacts={
                "download_timings": {
                    "source_metadata_seconds": 0.2,
                    "youtube_download_seconds": 1.5,
                    "download_total_seconds": 1.9,
                }
            },
        )
        store.claim_next([JobStage.ANALYZE], "gpu")
        store.complete_stage(
            job.id,
            next_stage=None,
            artifacts={"analysis_timings": {"allin1_analyze_seconds": 10.0}},
        )

        summary = store.recent_timing_summary(limit=10)

        assert summary["completed_jobs_sampled"] == 1
        assert summary["download"]["count"] == 1
        assert summary["download"]["avg_seconds"]["source_metadata_seconds"] == 0.2
        assert summary["download"]["avg_seconds"]["youtube_download_seconds"] == 1.5
        assert summary["download"]["avg_seconds"]["download_total_seconds"] == 1.9
        assert summary["analyze"]["count"] == 1
        assert summary["latest"][0]["stage"] == JobStage.ANALYZE.value


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


def test_enqueue_process_children_inserts_batch_and_events():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})

        children = store.enqueue_process_children(
            parent_job=parent,
            children=[
                {
                    "child_id": "root:bulk:seg-0:chord",
                    "job_type": JobType.BULK_DISSECT,
                    "payload": {"source": "song", "root_job_id": "root"},
                    "artifacts": {"process_mode": "segment_chord", "segment_id": "seg-0", "requires_gpu": False},
                    "priority": PROCESS_PRIORITY_BULK_CHORD,
                },
                {
                    "child_id": "root:bulk:seg-1:chord",
                    "job_type": JobType.BULK_DISSECT,
                    "payload": {"source": "song", "root_job_id": "root"},
                    "artifacts": {"process_mode": "segment_chord", "segment_id": "seg-1", "requires_gpu": False},
                    "priority": PROCESS_PRIORITY_BULK_CHORD,
                },
            ],
        )

        assert [child.id for child in children] == ["root:bulk:seg-0:chord", "root:bulk:seg-1:chord"]
        assert all(child.stage == JobStage.PROCESS for child in children)
        assert store.child_summary("root") == {"total": 2, "active": 2, "completed": 0, "failed": 0}
        parent_events = store.recent_events("root", limit=10)
        child_events = store.recent_events("root:bulk:seg-0:chord", limit=10)
        assert sum(event["event_type"] == "process_child_enqueued" for event in parent_events) == 2
        assert child_events[0]["event_type"] == "enqueued"
        assert child_events[0]["data"]["created"] is True


def test_enqueue_process_children_is_idempotent_for_existing_child_ids():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        child_spec = {
            "child_id": "root:bulk:seg-0:chord",
            "job_type": JobType.BULK_DISSECT,
            "payload": {"source": "song", "root_job_id": "root"},
            "artifacts": {"process_mode": "segment_chord", "segment_id": "seg-0", "requires_gpu": False},
            "priority": PROCESS_PRIORITY_BULK_CHORD,
        }

        first = store.enqueue_process_children(parent_job=parent, children=[child_spec])
        second = store.enqueue_process_children(parent_job=parent, children=[child_spec])

        assert first[0].id == second[0].id == "root:bulk:seg-0:chord"
        assert store.stats()["active_depth"] == 2
        child_events = store.recent_events("root:bulk:seg-0:chord", limit=10)
        assert [event["data"]["created"] for event in child_events if event["event_type"] == "enqueued"] == [
            False,
            True,
        ]


def test_enqueue_process_child_uses_batch_semantics_for_existing_child_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})

        first = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-0:other",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_other", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_OTHER,
        )
        second = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-0:other",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_other", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_OTHER,
        )

        assert first.id == second.id == "root:bulk:seg-0:other"
        assert first.stage == second.stage == JobStage.PROCESS
        assert store.stats()["active_depth"] == 2
        parent_events = store.recent_events("root", limit=10)
        child_events = store.recent_events("root:bulk:seg-0:other", limit=10)
        assert [event["data"]["created"] for event in child_events if event["event_type"] == "enqueued"] == [
            False,
            True,
        ]
        process_events = [event for event in parent_events if event["event_type"] == "process_child_enqueued"]
        assert [event["data"]["created"] for event in process_events] == [False, True]
        assert process_events[0]["data"]["process_mode"] == "segment_other"
        assert process_events[0]["data"]["segment_id"] == "seg-0"


def test_enqueue_process_children_checks_queue_depth_once_for_new_batch():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=2)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})

        with pytest.raises(QueueFull):
            store.enqueue_process_children(
                parent_job=parent,
                children=[
                    {
                        "child_id": "root:bulk:seg-0:chord",
                        "job_type": JobType.BULK_DISSECT,
                        "payload": {"source": "song", "root_job_id": "root"},
                        "artifacts": {"process_mode": "segment_chord", "segment_id": "seg-0"},
                        "priority": PROCESS_PRIORITY_BULK_CHORD,
                    },
                    {
                        "child_id": "root:bulk:seg-1:chord",
                        "job_type": JobType.BULK_DISSECT,
                        "payload": {"source": "song", "root_job_id": "root"},
                        "artifacts": {"process_mode": "segment_chord", "segment_id": "seg-1"},
                        "priority": PROCESS_PRIORITY_BULK_CHORD,
                    },
                ],
            )

        assert store.child_summary("root") == {"total": 0, "active": 0, "completed": 0, "failed": 0}


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


def test_cpu_process_claim_only_takes_jobs_that_do_not_require_gpu():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "quick_dissect", "source": "song"})
        store.enqueue_process_child(
            parent_job=parent,
            child_id="root:quick:seg-1:chord",
            job_type=JobType.QUICK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-1", "requires_gpu": False},
            priority=PROCESS_PRIORITY_QUICK_CHORD,
        )
        store.enqueue_process_child(
            parent_job=parent,
            child_id="root:quick:seg-2:chord",
            job_type=JobType.QUICK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-2", "requires_gpu": True},
            priority=PROCESS_PRIORITY_QUICK_CHORD,
        )

        claimed = store.claim_cpu_process_batch("cpu-process", limit=10)

        assert [job.id for job in claimed] == ["root:quick:seg-1:chord"]
        assert store.get("root:quick:seg-2:chord").status == JobStatus.QUEUED


def test_gpu_claim_takes_analyze_and_gpu_required_process_only():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=10)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "quick_dissect", "source": "song"})
        bulk, _ = store.enqueue({"job_id": "bulk", "job_type": "bulk_dissect", "source": "song"})
        store.complete_stage(bulk.id, next_stage=JobStage.ANALYZE)
        store.enqueue_process_child(
            parent_job=parent,
            child_id="root:quick:seg-1:chord",
            job_type=JobType.QUICK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-1", "requires_gpu": False},
            priority=PROCESS_PRIORITY_QUICK_CHORD,
        )
        store.enqueue_process_child(
            parent_job=parent,
            child_id="root:quick:seg-2:chord",
            job_type=JobType.QUICK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-2", "requires_gpu": True},
            priority=PROCESS_PRIORITY_QUICK_CHORD,
        )

        claimed = store.claim_gpu_batch("gpu", limit=10)

        assert [job.id for job in claimed] == ["root:quick:seg-2:chord", "bulk"]
        assert store.get("root:quick:seg-1:chord").status == JobStatus.QUEUED


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


def test_child_summary_counts_only_matching_root_jobs_and_can_exclude_current_child():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=20)
        parent, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
        other_parent, _ = store.enqueue({"job_id": "other-root", "job_type": "bulk_dissect", "source": "other"})
        completed_child = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        active_child = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-1:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-1"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        failed_child = store.enqueue_process_child(
            parent_job=parent,
            child_id="root:bulk:seg-2:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song", "root_job_id": "root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-2"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.enqueue_process_child(
            parent_job=other_parent,
            child_id="other-root:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "other", "root_job_id": "other-root"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.claim_next([JobStage.PROCESS], "process-worker")
        store.complete_stage(completed_child.id, next_stage=None)
        store.fail_stage(failed_child.id, "broken", retry_delay_seconds=0)
        store.fail_stage(failed_child.id, "broken", retry_delay_seconds=0)
        store.fail_stage(failed_child.id, "broken", retry_delay_seconds=0)

        assert store.child_summary("root") == {"total": 3, "active": 1, "completed": 1, "failed": 1}
        assert store.child_summary("root", exclude_job_id=active_child.id) == {
            "total": 2,
            "active": 0,
            "completed": 1,
            "failed": 1,
        }


def test_child_summaries_counts_multiple_roots_in_one_call():
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "queue.sqlite3", max_depth=20)
        root_a, _ = store.enqueue({"job_id": "root-a", "job_type": "bulk_dissect", "source": "song-a"})
        root_b, _ = store.enqueue({"job_id": "root-b", "job_type": "bulk_dissect", "source": "song-b"})
        child_a = store.enqueue_process_child(
            parent_job=root_a,
            child_id="root-a:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song-a", "root_job_id": "root-a"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.enqueue_process_child(
            parent_job=root_a,
            child_id="root-a:bulk:seg-1:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song-a", "root_job_id": "root-a"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-1"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        child_b = store.enqueue_process_child(
            parent_job=root_b,
            child_id="root-b:bulk:seg-0:chord",
            job_type=JobType.BULK_DISSECT,
            payload={"source": "song-b", "root_job_id": "root-b"},
            artifacts={"process_mode": "segment_chord", "segment_id": "seg-0"},
            priority=PROCESS_PRIORITY_BULK_CHORD,
        )
        store.claim_next([JobStage.PROCESS], "process-worker")
        store.complete_stage(child_a.id, next_stage=None)
        store.fail_stage(child_b.id, "broken", retry_delay_seconds=0)
        store.fail_stage(child_b.id, "broken", retry_delay_seconds=0)
        store.fail_stage(child_b.id, "broken", retry_delay_seconds=0)

        assert store.child_summaries(["root-a", "root-b", "missing-root"]) == {
            "root-a": {"total": 2, "active": 1, "completed": 1, "failed": 0},
            "root-b": {"total": 1, "active": 0, "completed": 0, "failed": 1},
            "missing-root": {"total": 0, "active": 0, "completed": 0, "failed": 0},
        }


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
