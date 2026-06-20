import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.queue.types import JobEventType, JobStage, JobStatus, JobType


DEFAULT_JOB_PRIORITIES = {
    JobType.QUICK_DISSECT: 350,
    JobType.BULK_DISSECT: 10,
}


class QueueFull(RuntimeError):
    """Raised when the durable local queue is at capacity."""


@dataclass(frozen=True)
class JobRecord:
    id: str
    job_type: JobType
    stage: JobStage
    status: JobStatus
    payload: dict[str, Any]
    artifacts: dict[str, Any]
    attempts: int
    max_attempts: int
    priority: int
    worker_id: str | None
    error: str | None
    created_at: float
    updated_at: float
    available_at: float


class JobStore:
    """Small durable queue backed by SQLite WAL.

    Every public method opens its own connection and uses a process-local lock.
    This keeps the implementation simple for one VM process while still allowing
    safe recovery after process restart.
    """

    def __init__(self, db_path: Path | str, max_depth: int = 200):
        self.db_path = Path(db_path)
        self.max_depth = max_depth
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_message_id TEXT,
                    job_type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    priority INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    available_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_source_message
                    ON jobs(source_message_id)
                    WHERE source_message_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_jobs_claim
                    ON jobs(status, stage, available_at, priority, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_payload_root_status
                    ON jobs(json_extract(payload_json, '$.root_job_id'), status, id);
                CREATE INDEX IF NOT EXISTS idx_jobs_payload_parent
                    ON jobs(json_extract(payload_json, '$.parent_job_id'));
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    data_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                """
            )

    def enqueue(
        self,
        payload: dict[str, Any],
        *,
        source_message_id: str | None = None,
        priority: int | None = None,
        max_attempts: int = 3,
        initial_stage: JobStage = JobStage.DOWNLOAD,
        initial_artifacts: dict[str, Any] | None = None,
    ) -> tuple[JobRecord, bool]:
        job_type = JobType(str(payload.get("job_type") or "unknown"))
        job_id = str(payload.get("job_id") or payload.get("id") or uuid.uuid4())
        job_priority = self._resolve_priority(payload, priority, job_type)
        artifacts_json = _json_dumps(initial_artifacts or {})
        now = time.time()
        with self._lock, self._connect() as conn:
            current_depth = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status NOT IN (?, ?)",
                (JobStatus.COMPLETED.value, JobStatus.FAILED.value),
            ).fetchone()[0]
            existing_by_source = self.get_by_source_message(source_message_id)
            if existing_by_source:
                return existing_by_source, False
            existing = self.get(job_id)
            if existing:
                return existing, False
            if current_depth >= self.max_depth:
                raise QueueFull(f"queue depth {current_depth} >= max {self.max_depth}")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO jobs (
                    id, source_message_id, job_type, stage, status, payload_json,
                    artifacts_json, attempts, max_attempts, priority, created_at,
                    updated_at, available_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    source_message_id,
                    job_type.value,
                    initial_stage.value,
                    JobStatus.QUEUED.value,
                    _json_dumps(payload),
                    artifacts_json,
                    max_attempts,
                    job_priority,
                    now,
                    now,
                    now,
                ),
            )
            conn.execute("COMMIT")
        record = self.get_by_source_message(source_message_id) if source_message_id else self.get(job_id)
        if record is None:
            raise RuntimeError("job enqueue failed")
        created = record.id == job_id
        self.add_event(
            record.id,
            JobEventType.ENQUEUED,
            data={"job_type": job_type.value, "created": created, "priority": record.priority},
        )
        return record, created

    def enqueue_continuation(
        self,
        *,
        parent_job: JobRecord,
        payload: dict[str, Any],
        artifacts: dict[str, Any],
        priority: int | None = None,
    ) -> JobRecord:
        continuation_id = str(payload.get("job_id") or f"{parent_job.id}:bulk")
        continuation_payload = {
            **payload,
            "job_id": continuation_id,
            "job_type": JobType.BULK_DISSECT.value,
            "parent_job_id": parent_job.id,
        }
        job, created = self.enqueue(
            continuation_payload,
            priority=priority if priority is not None else DEFAULT_JOB_PRIORITIES[JobType.BULK_DISSECT],
            initial_stage=JobStage.PROCESS,
            initial_artifacts=artifacts,
        )
        self.add_event(
            parent_job.id,
            JobEventType.CONTINUATION_ENQUEUED,
            data={"continuation_job_id": job.id, "created": created},
        )
        return job

    def enqueue_process_child(
        self,
        *,
        parent_job: JobRecord,
        child_id: str,
        job_type: JobType,
        payload: dict[str, Any],
        artifacts: dict[str, Any],
        priority: int,
    ) -> JobRecord:
        child_payload = {
            **payload,
            "job_id": child_id,
            "job_type": job_type.value,
            "parent_job_id": parent_job.id,
            "root_job_id": payload.get("root_job_id") or parent_job.payload.get("root_job_id") or parent_job.id,
        }
        job, created = self.enqueue(
            child_payload,
            priority=priority,
            initial_stage=JobStage.PROCESS,
            initial_artifacts=artifacts,
        )
        self.add_event(
            parent_job.id,
            JobEventType.PROCESS_CHILD_ENQUEUED,
            data={
                "child_job_id": job.id,
                "created": created,
                "job_type": job.job_type.value,
                "priority": job.priority,
                "process_mode": artifacts.get("process_mode"),
                "segment_id": artifacts.get("segment_id"),
            },
        )
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_source_message(self, source_message_id: str | None) -> JobRecord | None:
        if not source_message_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE source_message_id = ?", (source_message_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def claim_next(self, stages: Iterable[JobStage | str], worker_id: str) -> JobRecord | None:
        records = self.claim_batch(stages, worker_id, limit=1)
        return records[0] if records else None

    def claim_batch(self, stages: Iterable[JobStage | str], worker_id: str, *, limit: int) -> list[JobRecord]:
        stage_list = [_stage_value(stage) for stage in stages]
        if not stage_list or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in stage_list)
        return self._claim_where(
            f"stage IN ({placeholders})",
            tuple(stage_list),
            worker_id=worker_id,
            limit=limit,
        )

    def claim_cpu_process_batch(self, worker_id: str, *, limit: int) -> list[JobRecord]:
        if limit <= 0:
            return []
        return self._claim_where(
            "stage = ? AND COALESCE(json_extract(artifacts_json, '$.requires_gpu'), 1) = 0",
            (JobStage.PROCESS.value,),
            worker_id=worker_id,
            limit=limit,
        )

    def claim_gpu_batch(self, worker_id: str, *, limit: int) -> list[JobRecord]:
        if limit <= 0:
            return []
        return self._claim_where(
            """
            (
                stage = ?
                OR (
                    stage = ?
                    AND COALESCE(json_extract(artifacts_json, '$.requires_gpu'), 1) != 0
                )
            )
            """,
            (JobStage.ANALYZE.value, JobStage.PROCESS.value),
            worker_id=worker_id,
            limit=limit,
        )

    def _claim_where(
        self,
        extra_where: str,
        extra_params: tuple[Any, ...],
        *,
        worker_id: str,
        limit: int,
    ) -> list[JobRecord]:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE status = ?
                  AND available_at <= ?
                  AND {extra_where}
                ORDER BY priority DESC,
                         created_at ASC,
                         id ASC
                LIMIT ?
                """,
                (JobStatus.QUEUED.value, now, *extra_params, limit),
            ).fetchall()
            if not rows:
                conn.execute("COMMIT")
                return []
            job_ids = [row["id"] for row in rows]
            id_placeholders = ",".join("?" for _ in job_ids)
            conn.execute(
                f"""
                UPDATE jobs
                SET status = ?,
                    worker_id = ?,
                    updated_at = ?
                WHERE id IN ({id_placeholders})
                """,
                (JobStatus.PROCESSING.value, worker_id, now, *job_ids),
            )
            conn.execute("COMMIT")
        records = [record for job_id in job_ids if (record := self.get(job_id)) is not None]
        for record in records:
            self.add_event(record.id, JobEventType.CLAIMED, data={"stage": record.stage.value, "worker_id": worker_id})
        return records

    def complete_stage(
        self,
        job_id: str,
        *,
        next_stage: JobStage | str | None,
        artifacts: dict[str, Any] | None = None,
    ) -> JobRecord:
        now = time.time()
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        merged_artifacts = {**record.artifacts, **(artifacts or {})}
        status = JobStatus.COMPLETED if next_stage is None else JobStatus.QUEUED
        stage = record.stage if next_stage is None else JobStage(str(next_stage))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET stage = ?,
                    status = ?,
                    artifacts_json = ?,
                    worker_id = NULL,
                    error = NULL,
                    updated_at = ?,
                    available_at = ?
                WHERE id = ?
                """,
                (stage.value, status.value, _json_dumps(merged_artifacts), now, now, job_id),
            )
        self.add_event(
            job_id,
            JobEventType.STAGE_COMPLETED,
            data={"next_stage": next_stage.value if isinstance(next_stage, JobStage) else next_stage},
        )
        updated = self.get(job_id)
        if updated is None:
            raise KeyError(job_id)
        return updated

    def fail_stage(self, job_id: str, error: str, *, retry_delay_seconds: int = 30) -> JobRecord:
        now = time.time()
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        next_attempts = record.attempts + 1
        retry = next_attempts < record.max_attempts
        status = JobStatus.QUEUED if retry else JobStatus.FAILED
        available_at = now + retry_delay_seconds if retry else now
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    worker_id = NULL,
                    attempts = ?,
                    error = ?,
                    updated_at = ?,
                    available_at = ?
                WHERE id = ?
                """,
                (status.value, next_attempts, error[:4000], now, available_at, job_id),
            )
        self.add_event(job_id, JobEventType.FAILED if not retry else JobEventType.RETRY_SCHEDULED, message=error)
        updated = self.get(job_id)
        if updated is None:
            raise KeyError(job_id)
        return updated

    def retry(self, job_id: str) -> JobRecord:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                    SET status = ?,
                    worker_id = NULL,
                    error = NULL,
                    updated_at = ?,
                    available_at = ?
                WHERE id = ?
                """,
                (JobStatus.QUEUED.value, now, now, job_id),
            )
        self.add_event(job_id, JobEventType.MANUAL_RETRY)
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def reconcile_failed_fanout_parent(self, root_job_id: str) -> JobRecord | None:
        record = self.get(root_job_id)
        if record is None or record.status != JobStatus.FAILED:
            return record
        summary = self.child_summary(root_job_id)
        if summary["total"] <= 0 or summary["active"] > 0 or summary["failed"] > 0:
            return record

        now = time.time()
        artifacts = {
            **record.artifacts,
            "fanout_reconciled": {
                "completed_children": summary["completed"],
                "reconciled_at": now,
            },
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    worker_id = NULL,
                    error = NULL,
                    artifacts_json = ?,
                    updated_at = ?,
                    available_at = ?
                WHERE id = ?
                  AND status = ?
                """,
                (
                    JobStatus.COMPLETED.value,
                    _json_dumps(artifacts),
                    now,
                    now,
                    root_job_id,
                    JobStatus.FAILED.value,
                ),
            )
        self.add_event(
            root_job_id,
            JobEventType.FANOUT_PARENT_RECONCILED,
            data={"child_summary": summary},
        )
        return self.get(root_job_id)

    def recover_stale_processing(self, *, lease_timeout_seconds: int) -> int:
        cutoff = time.time() - lease_timeout_seconds
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id FROM jobs
                WHERE status = ?
                  AND updated_at <= ?
                """,
                (JobStatus.PROCESSING.value, cutoff),
            ).fetchall()
            job_ids = [row["id"] for row in rows]
            if not job_ids:
                conn.execute("COMMIT")
                return 0
            placeholders = ",".join("?" for _ in job_ids)
            conn.execute(
                f"""
                UPDATE jobs
                SET status = ?,
                    worker_id = NULL,
                    error = 'recovered stale processing lease',
                    updated_at = ?,
                    available_at = ?
                WHERE id IN ({placeholders})
                """,
                (JobStatus.QUEUED.value, now, now, *job_ids),
            )
            conn.execute("COMMIT")
        for job_id in job_ids:
            self.add_event(job_id, JobEventType.LEASE_RECOVERED, message="requeued stale processing job")
        return len(job_ids)

    def recover_processing_after_restart(self, *, error: str = "recovered processing job after service restart") -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id, attempts, max_attempts FROM jobs
                WHERE status = ?
                """,
                (JobStatus.PROCESSING.value,),
            ).fetchall()
            if not rows:
                conn.execute("COMMIT")
                return 0
            for row in rows:
                attempts = int(row["attempts"]) + 1
                terminal = attempts >= int(row["max_attempts"])
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?,
                        worker_id = NULL,
                        attempts = ?,
                        error = ?,
                        updated_at = ?,
                        available_at = ?
                    WHERE id = ?
                    """,
                    (
                        JobStatus.FAILED.value if terminal else JobStatus.QUEUED.value,
                        attempts,
                        error,
                        now,
                        now,
                        row["id"],
                    ),
                )
            conn.execute("COMMIT")
        for row in rows:
            attempts = int(row["attempts"]) + 1
            terminal = attempts >= int(row["max_attempts"])
            self.add_event(
                row["id"],
                JobEventType.FAILED if terminal else JobEventType.LEASE_RECOVERED,
                message=error,
                data={"attempts": attempts, "terminal": terminal},
            )
        return len(rows)

    def inactive_work_dirs(self, *, older_than_seconds: float, limit: int = 100) -> list[str]:
        cutoff = time.time() - older_than_seconds
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    json_extract(artifacts_json, '$.work_dir') AS work_dir,
                    MAX(updated_at) AS last_updated_at,
                    SUM(
                        CASE
                            WHEN status NOT IN (?, ?) THEN 1
                            ELSE 0
                        END
                    ) AS active_count
                FROM jobs
                WHERE json_extract(artifacts_json, '$.work_dir') IS NOT NULL
                  AND json_extract(artifacts_json, '$.work_dir') != ''
                GROUP BY work_dir
                HAVING active_count = 0
                   AND last_updated_at <= ?
                ORDER BY last_updated_at ASC
                LIMIT ?
                """,
                (JobStatus.COMPLETED.value, JobStatus.FAILED.value, cutoff, max(0, limit)),
            ).fetchall()
        return [str(row["work_dir"]) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            by_stage = {
                row["stage"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT stage, COUNT(*) AS count
                    FROM jobs
                    WHERE status NOT IN (?, ?)
                    GROUP BY stage
                    """,
                    (JobStatus.COMPLETED.value, JobStatus.FAILED.value),
                )
            }
            by_status = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
                )
            }
            by_priority = {
                str(row["priority"]): row["count"]
                for row in conn.execute(
                    """
                    SELECT priority, COUNT(*) AS count
                    FROM jobs
                    WHERE status NOT IN (?, ?)
                    GROUP BY priority
                    ORDER BY priority DESC
                    """,
                    (JobStatus.COMPLETED.value, JobStatus.FAILED.value),
                )
            }
            oldest = conn.execute(
                """
                SELECT MIN(created_at) AS created_at
                FROM jobs
                WHERE status NOT IN (?, ?)
                """,
                (JobStatus.COMPLETED.value, JobStatus.FAILED.value),
            ).fetchone()["created_at"]
        now = time.time()
        return {
            "by_stage": by_stage,
            "by_status": by_status,
            "by_priority": by_priority,
            "active_depth": sum(by_stage.values()),
            "max_depth": self.max_depth,
            "oldest_job_age_seconds": round(now - oldest, 3) if oldest else None,
            "backpressure": sum(by_stage.values()) >= self.max_depth,
        }

    def recent_events(self, job_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_type, message, data_json, created_at
                FROM job_events
                WHERE job_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (job_id, limit),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "message": row["message"],
                "data": json.loads(row["data_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def child_summary(self, root_job_id: str, *, exclude_job_id: str | None = None) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM jobs
                WHERE json_extract(payload_json, '$.root_job_id') = ?
                  AND (? IS NULL OR id != ?)
                GROUP BY status
                """,
                (root_job_id, exclude_job_id, exclude_job_id),
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        completed = counts.get(JobStatus.COMPLETED.value, 0)
        failed = counts.get(JobStatus.FAILED.value, 0)
        total = sum(counts.values())
        active = total - completed - failed
        return {
            "total": total,
            "active": active,
            "completed": completed,
            "failed": failed,
        }

    def add_event(
        self,
        job_id: str,
        event_type: str,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_events (job_id, event_type, message, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, str(event_type), message, _json_dumps(data or {}), time.time()),
            )

    @staticmethod
    def _resolve_priority(payload: dict[str, Any], priority: int | None, job_type: JobType) -> int:
        if priority is not None:
            return int(priority)
        if "priority" in payload and payload["priority"] is not None:
            return int(payload["priority"])
        return DEFAULT_JOB_PRIORITIES.get(job_type, 0)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            job_type=JobType(row["job_type"]),
            stage=JobStage(row["stage"]),
            status=JobStatus(row["status"]),
            payload=json.loads(row["payload_json"]),
            artifacts=json.loads(row["artifacts_json"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            priority=row["priority"],
            worker_id=row["worker_id"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            available_at=row["available_at"],
        )


def _stage_value(stage: JobStage | str) -> str:
    return stage.value if isinstance(stage, JobStage) else JobStage(str(stage)).value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)
