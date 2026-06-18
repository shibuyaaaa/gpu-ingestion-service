import os
from typing import Any


class DBClient:
    """Small asyncpg pool wrapper.

    The new service keeps DB updates isolated behind this wrapper so job
    adapters can gradually port legacy status mutations without importing the
    legacy worker.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        min_size: int | None = None,
        max_size: int | None = None,
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self.min_size = min_size if min_size is not None else int(os.getenv("DB_POOL_MIN_SIZE", "1"))
        self.max_size = max_size if max_size is not None else int(os.getenv("DB_POOL_MAX_SIZE", "5"))
        self._pool: Any | None = None

    async def pool(self) -> Any:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        if self._pool is None:
            import asyncpg

            max_size = max(1, self.max_size)
            min_size = min(max(0, self.min_size), max_size)
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=min_size,
                max_size=max_size,
            )
        return self._pool

    async def warmup(self) -> bool:
        pool = await self.pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True

    async def health(self) -> bool:
        try:
            await self.warmup()
            return True
        except Exception:
            return False

    async def execute(self, query: str, *args: Any) -> str:
        pool = await self.pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        pool = await self.pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        pool = await self.pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def status(self) -> dict[str, Any]:
        return {
            "configured": bool(self.database_url),
            "pool_created": self._pool is not None,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }
