import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg


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


async def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_duplicate_stem_migration.py <migration.sql>")
    migration_path = Path(sys.argv[1])
    sql = migration_path.read_text(encoding="utf-8")
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        before = dict(await conn.fetchrow(DUPLICATE_SQL))
        print(json.dumps({"before": before}, sort_keys=True), flush=True)
        await conn.execute(sql)
        after = dict(await conn.fetchrow(DUPLICATE_SQL))
        index_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = current_schema()
                  AND indexname = 'idx_stems_gpu_ingestion_unique_segment'
            )
            """
        )
        print(
            json.dumps(
                {"after": after, "index_exists": bool(index_exists)},
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
