import argparse
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

import asyncpg


YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})"
)


DUPLICATE_SQL = """
WITH grouped AS (
    SELECT
        song_id,
        LOWER(TRIM(stem_type)) AS stem_type,
        COALESCE(segment, '') AS segment,
        ROUND(COALESCE(start_time, -1)::numeric, 3) AS start_time,
        ROUND(COALESCE(end_time, -1)::numeric, 3) AS end_time,
        COUNT(*) AS count
    FROM stems
    WHERE COALESCE(audio_url, '') <> ''
      AND (
          COALESCE(model, '') LIKE 'gpu-ingestion%'
          OR COALESCE(audio_url, '') LIKE '%/gpu-ingestion/%'
          OR COALESCE(audio_url, '') LIKE '%gpu-ingestion/cache/%'
      )
    GROUP BY 1, 2, 3, 4, 5
    HAVING COUNT(*) > 1
)
SELECT COALESCE(SUM(count - 1), 0)::int AS duplicate_rows,
       COUNT(*)::int AS duplicate_groups
FROM grouped
"""


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed {exc.code}: {body}") from exc


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def submit_job(
    base_url: str,
    *,
    job_id: str,
    job_type: str,
    source: str,
    force_processing: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "source": source,
    }
    if force_processing:
        payload["skip_library_precheck"] = True
    return post_json(f"{base_url}/jobs", payload)


def terminal(job: dict[str, Any]) -> bool:
    if job["status"] not in {"completed", "failed"}:
        return False
    summary = job.get("children_summary") or {}
    total = int(summary.get("total") or 0)
    if total == 0:
        return True
    active = int(summary.get("active") or 0)
    completed = int(summary.get("completed") or 0)
    failed = int(summary.get("failed") or 0)
    return active == 0 and completed + failed == total


def assert_not_failed(job: dict[str, Any]) -> None:
    failed_children = []
    for child in job.get("children") or []:
        if child.get("status") == "failed":
            failed_children.append({"id": child.get("id"), "error": child.get("error")})
    if job.get("status") == "failed" or failed_children:
        raise RuntimeError(
            json.dumps(
                {
                    "job_id": job.get("id"),
                    "status": job.get("status"),
                    "error": job.get("error"),
                    "failed_children": failed_children,
                },
                sort_keys=True,
            )
        )


def wait_job(base_url: str, job_id: str, *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        last = get_json(f"{base_url}/ops/jobs/{job_id}")
        if terminal(last):
            assert_not_failed(last)
            return last
        time.sleep(5)
    raise TimeoutError(json.dumps({"job_id": job_id, "last": summarize_job(last)}, sort_keys=True))


def summarize_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "id": job.get("id"),
        "job_type": job.get("job_type"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "children_summary": job.get("children_summary"),
        "error": job.get("error"),
    }


def youtube_id(source: str) -> str | None:
    match = YOUTUBE_ID_RE.search(source)
    return match.group(1) if match else None


async def verify_db(source: str) -> dict[str, Any]:
    video_id = youtube_id(source)
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        duplicate_summary = dict(await conn.fetchrow(DUPLICATE_SQL))
        if not video_id:
            return {"video_id": None, "duplicate_summary": duplicate_summary, "songs": []}
        rows = await conn.fetch(
            """
            SELECT
                s.id,
                s.title,
                s.audio_url,
                s.all_in_one_bpm,
                s.beat_analysis_bpm,
                s.key,
                s.analysis_json::text AS analysis_json_text,
                COUNT(st.id)::int AS stem_count,
                COUNT(st.id) FILTER (WHERE COALESCE(st.audio_url, '') <> '')::int AS audio_stem_count,
                COUNT(st.id) FILTER (WHERE LOWER(TRIM(st.stem_type)) = 'chord' AND COALESCE(st.audio_url, '') <> '')::int AS chord_count,
                COUNT(st.id) FILTER (WHERE LOWER(TRIM(st.stem_type)) = 'voice' AND COALESCE(st.audio_url, '') <> '')::int AS voice_count,
                COUNT(st.id) FILTER (WHERE LOWER(TRIM(st.stem_type)) = 'beat' AND COALESCE(st.audio_url, '') <> '')::int AS beat_count,
                COUNT(st.id) FILTER (WHERE LOWER(TRIM(st.stem_type)) = 'bass' AND COALESCE(st.audio_url, '') <> '')::int AS bass_count
            FROM songs s
            LEFT JOIN stems st ON st.song_id = s.id
            WHERE COALESCE(s.audio_url, '') LIKE $1
               OR COALESCE(s.analysis_json::text, '') LIKE $2
               OR EXISTS (
                    SELECT 1
                    FROM stems st_match
                    WHERE st_match.song_id = s.id
                      AND COALESCE(st_match.audio_url, '') LIKE $3
               )
            GROUP BY s.id
            ORDER BY s.created_at DESC
            LIMIT 10
            """,
            f"%source-audio/youtube-{video_id}/full.mp3%",
            f"%youtube:{video_id}%",
            f"%youtube-{video_id}%",
        )
        songs = []
        for row in rows:
            item = dict(row)
            analysis_text = item.pop("analysis_json_text") or ""
            item["has_gpu_ingestion_json"] = "gpu_ingestion" in analysis_text
            item["gpu_ingestion_complete"] = '"status": "complete"' in analysis_text or '"status":"complete"' in analysis_text
            songs.append(item)
        return {"video_id": video_id, "duplicate_summary": duplicate_summary, "songs": songs}
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--quick-source", required=True)
    parser.add_argument("--bulk-source", required=True)
    parser.add_argument("--job-prefix", default=f"live-smoke-{int(time.time())}")
    parser.add_argument("--force-processing", action="store_true")
    args = parser.parse_args()

    quick_id = f"{args.job_prefix}-quick"
    bulk_id = f"{args.job_prefix}-bulk"
    submitted = [
        submit_job(
            args.base_url,
            job_id=quick_id,
            job_type="quick_dissect",
            source=args.quick_source,
            force_processing=args.force_processing,
        ),
        submit_job(
            args.base_url,
            job_id=bulk_id,
            job_type="bulk_dissect",
            source=args.bulk_source,
            force_processing=args.force_processing,
        ),
    ]
    print(json.dumps({"submitted": submitted}, sort_keys=True), flush=True)
    quick_job = wait_job(args.base_url, quick_id, timeout_seconds=args.timeout_seconds)
    bulk_job = wait_job(args.base_url, bulk_id, timeout_seconds=args.timeout_seconds)
    db_checks = {
        "quick": asyncio.run(verify_db(args.quick_source)),
        "bulk": asyncio.run(verify_db(args.bulk_source)),
    }
    print(
        json.dumps(
            {
                "quick_job": summarize_job(quick_job),
                "bulk_job": summarize_job(bulk_job),
                "db_checks": db_checks,
            },
            default=str,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
