import os
from typing import Any


class DBClient:
    """Small asyncpg pool wrapper.

    The new service keeps DB updates isolated behind this wrapper so job
    adapters can gradually port legacy status mutations without importing the
    legacy worker.
    """

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self._pool: Any | None = None

    async def pool(self) -> Any:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        return self._pool

    async def health(self) -> bool:
        try:
            pool = await self.pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def execute(self, query: str, *args: Any) -> str:
        pool = await self.pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)
