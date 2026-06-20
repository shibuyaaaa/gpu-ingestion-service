import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from app.queue.types import JobEventType, JobStage, JobStatus, JobType


DEFAULT_JOB_PRIORITIES = {
    JobType.QUICK_DISSECT: 350,
    JobType.BULK_DISSECT: 10,
}
ACTIVE_JOBS_WHERE_SQL = "status NOT IN ('completed', 'failed')"


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
                DROP INDEX IF EXISTS idx_jobs_claim;
                CREATE INDEX IF NOT EXISTS idx_jobs_claim
                    ON jobs(status, priority DESC, created_at ASC, id ASC, available_at, stage);
                CREATE INDEX IF NOT EXISTS idx_jobs_active_depth
                    ON jobs(status)
                    WHERE status NOT IN ('completed', 'failed');
                DROP INDEX IF EXISTS idx_jobs_payload_root;
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
                f"SELECT COUNT(*) FROM jobs WHERE {ACTIVE_JOBS_WHERE_SQL}",
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
        children = self.enqueue_process_children(
            parent_job=parent_job,
            children=[
                {
                    "child_id": child_id,
                    "job_type": job_type,
                    "payload": payload,
                    "artifacts": artifacts,
                    "priority": priority,
                }
            ],
        )
        if not children:
            raise RuntimeError("process child enqueue failed")
        return children[0]

    def enqueue_process_children(
        self,
        *,
        parent_job: JobRecord,
        children: list[dict[str, Any]],
    ) -> list[JobRecord]:
        if not children:
            return []
        now = time.time()
        specs = []
        for child in children:
            job_type = JobType(str(child["job_type"]))
            child_id = str(child["child_id"])
            payload = dict(child.get("payload") or {})
            child_payload = {
                **payload,
                "job_id": child_id,
                "job_type": job_type.value,
                "parent_job_id": parent_job.id,
                "root_job_id": payload.get("root_job_id") or parent_job.payload.get("root_job_id") or parent_job.id,
            }
            priority = int(child["priority"])
            artifacts = dict(child.get("artifacts") or {})
            specs.append(
                {
                    "child_id": child_id,
                    "job_type": job_type,
                    "payload": child_payload,
                    "artifacts": artifacts,
                    "priority": priority,
                }
            )

        job_ids = [spec["child_id"] for spec in specs]
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in job_ids)
                existing_rows = conn.execute(
                    f"SELECT id FROM jobs WHERE id IN ({placeholders})",
                    tuple(job_ids),
                ).fetchall()
                existing_ids = {row["id"] for row in existing_rows}
                new_specs = [spec for spec in specs if spec["child_id"] not in existing_ids]
                current_depth = conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE {ACTIVE_JOBS_WHERE_SQL}",
                ).fetchone()[0]
                if current_depth + len(new_specs) > self.max_depth:
                    raise QueueFull(f"queue depth {current_depth} + {len(new_specs)} new jobs > max {self.max_depth}")

                for spec in new_specs:
                    conn.execute(
                        """
                        INSERT INTO jobs (
                            id, source_message_id, job_type, stage, status, payload_json,
                            artifacts_json, attempts, max_attempts, priority, created_at,
                            updated_at, available_at
                        )
                        VALUES (?, NULL, ?, ?, ?, ?, ?, 0, 3, ?, ?, ?, ?)
                        """,
                        (
                            spec["child_id"],
                            spec["job_type"].value,
                            JobStage.PROCESS.value,
                            JobStatus.QUEUED.value,
                            _json_dumps(spec["payload"]),
                            _json_dumps(spec["artifacts"]),
                            spec["priority"],
                            now,
                            now,
                            now,
                        ),
                    )

                for spec in specs:
                    created = spec["child_id"] not in existing_ids
                    _add_event_conn(
                        conn,
                        spec["child_id"],
                        JobEventType.ENQUEUED,
                        created_at=now,
                        data={
                            "job_type": spec["job_type"].value,
                            "created": created,
                            "priority": spec["priority"],
                        },
                    )
                    _add_event_conn(
                        conn,
                        parent_job.id,
                        JobEventType.PROCESS_CHILD_ENQUEUED,
                        created_at=now,
                        data={
                            "child_job_id": spec["child_id"],
                            "created": created,
                            "job_type": spec["job_type"].value,
                            "priority": spec["priority"],
                            "process_mode": spec["artifacts"].get("process_mode"),
                            "segment_id": spec["artifacts"].get("segment_id"),
                        },
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        records = self.get_many(job_ids)
        return [records[job_id] for job_id in job_ids if job_id in records]

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def get_many(self, job_ids: Iterable[str]) -> dict[str, JobRecord]:
        normalized = [str(job_id) for job_id in job_ids if str(job_id)]
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE id IN ({placeholders})",
                tuple(normalized),
            ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        return {record.id: record for record in records}

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
            try:
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
                for row in rows:
                    _add_event_conn(
                        conn,
                        row["id"],
                        JobEventType.CLAIMED,
                        created_at=now,
                        data={"stage": row["stage"], "worker_id": worker_id},
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        records = [
            self._row_to_record(
                row,
                status=JobStatus.PROCESSING,
                worker_id=worker_id,
                updated_at=now,
            )
            for row in rows
        ]
        return records

    def complete_stage(
        self,
        job_id: str,
        *,
        next_stage: JobStage | str | None,
        artifacts: dict[str, Any] | None = None,
    ) -> JobRecord:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise KeyError(job_id)
                record = self._row_to_record(row)
                merged_artifacts = {**record.artifacts, **(artifacts or {})}
                artifacts_json = _json_dumps(merged_artifacts)
                normalized_artifacts = json.loads(artifacts_json)
                status = JobStatus.COMPLETED if next_stage is None else JobStatus.QUEUED
                stage = record.stage if next_stage is None else JobStage(str(next_stage))
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
                    (stage.value, status.value, artifacts_json, now, now, job_id),
                )
                _add_event_conn(
                    conn,
                    job_id,
                    JobEventType.STAGE_COMPLETED,
                    created_at=now,
                    data={"next_stage": next_stage.value if isinstance(next_stage, JobStage) else next_stage},
                )
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return replace(
            record,
            stage=stage,
            status=status,
            artifacts=normalized_artifacts,
            worker_id=None,
            error=None,
            updated_at=now,
            available_at=now,
        )

    def fail_stage(self, job_id: str, error: str, *, retry_delay_seconds: int = 30) -> JobRecord:
        now = time.time()
        stored_error = error[:4000]
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise KeyError(job_id)
                record = self._row_to_record(row)
                next_attempts = record.attempts + 1
                retry = next_attempts < record.max_attempts
                status = JobStatus.QUEUED if retry else JobStatus.FAILED
                event_type = JobEventType.RETRY_SCHEDULED if retry else JobEventType.FAILED
                available_at = now + retry_delay_seconds if retry else now
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
                    (status.value, next_attempts, stored_error, now, available_at, job_id),
                )
                _add_event_conn(conn, job_id, event_type, message=error, created_at=now)
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return replace(
            record,
            status=status,
            attempts=next_attempts,
            worker_id=None,
            error=stored_error,
            updated_at=now,
            available_at=available_at,
        )

    def retry(self, job_id: str) -> JobRecord:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise KeyError(job_id)
                record = self._row_to_record(row)
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
                _add_event_conn(conn, job_id, JobEventType.MANUAL_RETRY, created_at=now)
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return replace(
            record,
            status=JobStatus.QUEUED,
            worker_id=None,
            error=None,
            updated_at=now,
            available_at=now,
        )

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
                    WHERE status NOT IN ('completed', 'failed')
                    GROUP BY stage
                    """
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
                    WHERE status NOT IN ('completed', 'failed')
                    GROUP BY priority
                    ORDER BY priority DESC
                    """
                )
            }
            oldest = conn.execute(
                """
                SELECT MIN(created_at) AS created_at
                FROM jobs
                WHERE status NOT IN ('completed', 'failed')
                """
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

    def child_summaries(self, root_job_ids: Iterable[str]) -> dict[str, dict[str, int]]:
        normalized = [str(root_job_id) for root_job_id in dict.fromkeys(root_job_ids) if str(root_job_id)]
        summaries = {
            root_job_id: {"total": 0, "active": 0, "completed": 0, "failed": 0}
            for root_job_id in normalized
        }
        if not normalized:
            return summaries

        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    json_extract(payload_json, '$.root_job_id') AS root_job_id,
                    status,
                    COUNT(*) AS count
                FROM jobs
                WHERE json_extract(payload_json, '$.root_job_id') IN ({placeholders})
                GROUP BY root_job_id, status
                """,
                tuple(normalized),
            ).fetchall()

        for row in rows:
            root_job_id = str(row["root_job_id"])
            status = str(row["status"])
            count = int(row["count"])
            summary = summaries[root_job_id]
            summary["total"] += count
            if status == JobStatus.COMPLETED.value:
                summary["completed"] += count
            elif status == JobStatus.FAILED.value:
                summary["failed"] += count
            else:
                summary["active"] += count
        return summaries

    def add_event(
        self,
        job_id: str,
        event_type: str,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            _add_event_conn(conn, job_id, event_type, message=message, data=data, created_at=time.time())

    @staticmethod
    def _resolve_priority(payload: dict[str, Any], priority: int | None, job_type: JobType) -> int:
        if priority is not None:
            return int(priority)
        if "priority" in payload and payload["priority"] is not None:
            return int(payload["priority"])
        return DEFAULT_JOB_PRIORITIES.get(job_type, 0)

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
        *,
        status: JobStatus | None = None,
        worker_id: str | None = None,
        updated_at: float | None = None,
    ) -> JobRecord:
        return JobRecord(
            id=row["id"],
            job_type=JobType(row["job_type"]),
            stage=JobStage(row["stage"]),
            status=status or JobStatus(row["status"]),
            payload=json.loads(row["payload_json"]),
            artifacts=json.loads(row["artifacts_json"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            priority=row["priority"],
            worker_id=worker_id if worker_id is not None else row["worker_id"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=updated_at if updated_at is not None else row["updated_at"],
            available_at=row["available_at"],
        )


def _stage_value(stage: JobStage | str) -> str:
    return stage.value if isinstance(stage, JobStage) else JobStage(str(stage)).value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _add_event_conn(
    conn: sqlite3.Connection,
    job_id: str,
    event_type: str,
    *,
    message: str | None = None,
    data: dict[str, Any] | None = None,
    created_at: float,
) -> None:
    conn.execute(
        """
        INSERT INTO job_events (job_id, event_type, message, data_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job_id, str(event_type), message, _json_dumps(data or {}), created_at),
    )
