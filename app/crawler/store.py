import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.crawler.types import ChartCandidate

CONSUMED_CANDIDATE_STATUSES = {"submitted", "skipped_library"}
TERMINAL_SUBMISSION_STATUSES = {"completed", "failed"}


class CrawlerStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS crawler_sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    batch_size INTEGER NOT NULL,
                    submitted_count INTEGER NOT NULL DEFAULT 0,
                    completed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    next_poll_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS crawler_candidates (
                    spotify_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    artists_json TEXT NOT NULL,
                    popularity INTEGER NOT NULL,
                    playlist_source TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    last_session_id TEXT,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS crawler_submissions (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    spotify_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    root_status TEXT,
                    child_summary_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_crawler_sessions_active
                    ON crawler_sessions(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_crawler_submissions_session
                    ON crawler_submissions(session_id, status);
                """
            )

    def get_active_session(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM crawler_sessions
                WHERE status IN ('submitting', 'waiting')
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
        return _row_dict(row) if row else None

    def create_session(self, *, batch_size: int) -> dict[str, Any]:
        now = time.time()
        session_id = f"{int(now)}-{uuid.uuid4().hex[:8]}"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crawler_sessions (
                    id, status, batch_size, created_at, updated_at, next_poll_at
                )
                VALUES (?, 'submitting', ?, ?, ?, ?)
                """,
                (session_id, batch_size, now, now, now),
            )
        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError("crawler session create failed")
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM crawler_sessions WHERE id = ?", (session_id,)).fetchone()
        return _row_dict(row) if row else None

    def session_detail(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM crawler_submissions
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return {**session, "submissions": [_submission_dict(row) for row in rows]}

    def candidate_consumed(self, spotify_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM crawler_candidates WHERE spotify_id = ?",
                (spotify_id,),
            ).fetchone()
        return bool(row and row["status"] in CONSUMED_CANDIDATE_STATUSES)

    def record_candidate(self, candidate: ChartCandidate, *, status: str, session_id: str, error: str | None = None) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crawler_candidates (
                    spotify_id, title, artist, artists_json, popularity, playlist_source,
                    rank, status, first_seen_at, last_seen_at, last_session_id, last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(spotify_id) DO UPDATE SET
                    title = excluded.title,
                    artist = excluded.artist,
                    artists_json = excluded.artists_json,
                    popularity = excluded.popularity,
                    playlist_source = excluded.playlist_source,
                    rank = excluded.rank,
                    status = excluded.status,
                    last_seen_at = excluded.last_seen_at,
                    last_session_id = excluded.last_session_id,
                    last_error = excluded.last_error
                """,
                (
                    candidate.spotify_id,
                    candidate.title,
                    candidate.artist,
                    json.dumps(candidate.artists),
                    candidate.popularity,
                    candidate.playlist_source,
                    candidate.rank,
                    status,
                    now,
                    now,
                    session_id,
                    error,
                ),
            )

    def record_submission(
        self,
        *,
        session_id: str,
        candidate: ChartCandidate,
        job_id: str,
        payload: dict[str, Any],
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crawler_submissions (
                    job_id, session_id, spotify_id, payload_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'submitted', ?, ?)
                ON CONFLICT(job_id) DO NOTHING
                """,
                (job_id, session_id, candidate.spotify_id, json.dumps(payload), now, now),
            )
            self._refresh_session_counts(conn, session_id)

    def mark_session_waiting(self, session_id: str, *, next_poll_at: float | None = None, error: str | None = None) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE crawler_sessions
                SET status = 'waiting',
                    updated_at = ?,
                    next_poll_at = ?,
                    last_error = COALESCE(?, last_error)
                WHERE id = ?
                """,
                (now, next_poll_at if next_poll_at is not None else now, error, session_id),
            )
            self._refresh_session_counts(conn, session_id)

    def mark_session_error(self, session_id: str, *, next_poll_at: float | None = None, error: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE crawler_sessions
                SET updated_at = ?,
                    next_poll_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (now, next_poll_at if next_poll_at is not None else now, error, session_id),
            )
            self._refresh_session_counts(conn, session_id)

    def mark_session_completed(self, session_id: str) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE crawler_sessions
                SET status = 'completed',
                    updated_at = ?,
                    completed_at = ?,
                    next_poll_at = NULL
                WHERE id = ?
                """,
                (now, now, session_id),
            )
            self._refresh_session_counts(conn, session_id)

    def submissions_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM crawler_submissions
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [_submission_dict(row) for row in rows]

    def update_submission_status(
        self,
        *,
        job_id: str,
        status: str,
        root_status: str | None,
        child_summary: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        now = time.time()
        completed_at = now if status in TERMINAL_SUBMISSION_STATUSES else None
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT session_id FROM crawler_submissions WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return
            conn.execute(
                """
                UPDATE crawler_submissions
                SET status = ?,
                    root_status = ?,
                    child_summary_json = ?,
                    error = ?,
                    updated_at = ?,
                    completed_at = COALESCE(completed_at, ?)
                WHERE job_id = ?
                """,
                (
                    status,
                    root_status,
                    json.dumps(child_summary or {}),
                    error,
                    now,
                    completed_at,
                    job_id,
                ),
            )
            self._refresh_session_counts(conn, row["session_id"])

    def all_submissions_terminal(self, session_id: str) -> bool:
        submissions = self.submissions_for_session(session_id)
        return bool(submissions) and all(row["status"] in TERMINAL_SUBMISSION_STATUSES for row in submissions)

    def status(self) -> dict[str, Any]:
        active = self.get_active_session()
        with self._connect() as conn:
            by_status = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM crawler_sessions GROUP BY status"
                )
            }
            candidate_by_status = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM crawler_candidates GROUP BY status"
                )
            }
        return {
            "session_db_path": str(self.db_path),
            "active_session_id": active["id"] if active else None,
            "active_session": active,
            "sessions_by_status": by_status,
            "candidates_by_status": candidate_by_status,
        }

    @staticmethod
    def _refresh_session_counts(conn: sqlite3.Connection, session_id: str) -> None:
        counts = {
            row["status"]: row["count"]
            for row in conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM crawler_submissions
                WHERE session_id = ?
                GROUP BY status
                """,
                (session_id,),
            )
        }
        submitted = sum(counts.values())
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        conn.execute(
            """
            UPDATE crawler_sessions
            SET submitted_count = ?,
                completed_count = ?,
                failed_count = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (submitted, completed, failed, time.time(), session_id),
        )


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _submission_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["payload"] = json.loads(data.pop("payload_json"))
    data["child_summary"] = json.loads(data.pop("child_summary_json") or "{}")
    return data
