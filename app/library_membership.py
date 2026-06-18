import asyncio
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings
from app.legacy.db import DBClient

logger = logging.getLogger(__name__)

DISCOVERY_MATCH_THRESHOLD = 700

PLAYABLE_FULL_SONG_EXISTS_CLAUSE = """
    EXISTS (
        SELECT 1
        FROM stems st_catalog
        WHERE st_catalog.song_id = s.id
          AND COALESCE(st_catalog.audio_url, '') <> ''
          AND st_catalog.segment = 'full_song'
        GROUP BY st_catalog.song_id
        HAVING COUNT(
            DISTINCT CASE
                WHEN LOWER(TRIM(st_catalog.stem_type)) IN ('voice', 'vocal') THEN 'voice'
                WHEN LOWER(TRIM(st_catalog.stem_type)) IN ('beat', 'drum', 'drums') THEN 'beat'
                WHEN LOWER(TRIM(st_catalog.stem_type)) = 'bass' THEN 'bass'
                WHEN LOWER(TRIM(st_catalog.stem_type)) IN ('chord', 'chords', 'instrumental', 'melody') THEN 'chord'
                ELSE NULL
            END
        ) = 4
    )
"""


@dataclass(frozen=True)
class LibrarySong:
    id: str
    title: str
    artists: list[str]
    artist: str
    match_score: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "artists": self.artists,
            "match_score": self.match_score,
            **(self.metadata or {}),
        }


@dataclass(frozen=True)
class LibraryLookupResult:
    checked: bool
    exists: bool
    song: LibrarySong | None = None
    source: str = "disabled"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "exists": self.exists,
            "source": self.source,
            "error": self.error,
            "song": self.song.to_dict() if self.song else None,
        }


class LibraryMembershipChecker:
    def __init__(
        self,
        *,
        db: DBClient,
        settings: Settings,
        now: Callable[[], float] | None = None,
    ):
        self.db = db
        self.enabled = settings.library_precheck_enabled
        self.idle_ttl_seconds = settings.library_cache_idle_ttl_seconds
        self.max_age_seconds = settings.library_cache_max_age_seconds
        self._now = now or time.time
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._eviction_task: asyncio.Task[None] | None = None
        self._catalog: list[LibrarySong] | None = None
        self._catalog_by_title: dict[str, list[LibrarySong]] = {}
        self._loaded_at: float | None = None
        self._last_used_at: float | None = None
        self._last_refresh_duration_seconds: float | None = None
        self._last_error: str | None = None
        self._hits = 0
        self._misses = 0
        self._db_fallbacks = 0
        self._refreshes = 0
        self._evictions = 0

    async def lookup(self, metadata: dict[str, Any] | None) -> LibraryLookupResult:
        if not self.enabled:
            return LibraryLookupResult(checked=False, exists=False, source="disabled")
        title = _metadata_title(metadata)
        artist = _metadata_artist(metadata)
        if not title or not artist:
            return LibraryLookupResult(checked=False, exists=False, source="missing_metadata")

        self._touch()
        try:
            await self._ensure_catalog()
            cache_match = self._lookup_cache(title, artist)
            if cache_match:
                self._hits += 1
                return LibraryLookupResult(checked=True, exists=True, song=cache_match, source="cache")
            self._misses += 1
        except Exception as exc:
            self._record_error(exc)

        db_match = await self._lookup_db_safely(title, artist)
        if db_match:
            self._add_to_cache(db_match)
            return LibraryLookupResult(checked=True, exists=True, song=db_match, source="db")
        if self._last_error:
            return LibraryLookupResult(checked=True, exists=False, source="error", error=self._last_error)
        return LibraryLookupResult(checked=True, exists=False, source="miss")

    async def warmup(self) -> None:
        if not self.enabled:
            return
        try:
            await self.db.warmup()
        except Exception as exc:
            self._record_error(exc)

    def status(self) -> dict[str, Any]:
        now = self._now()
        age = None if self._loaded_at is None else max(0.0, now - self._loaded_at)
        idle_for = None if self._last_used_at is None else max(0.0, now - self._last_used_at)
        return {
            "enabled": self.enabled,
            "cache_loaded": self._catalog is not None,
            "cache_song_count": len(self._catalog or []),
            "cache_age_seconds": age,
            "idle_for_seconds": idle_for,
            "idle_ttl_seconds": self.idle_ttl_seconds,
            "max_age_seconds": self.max_age_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "db_fallbacks": self._db_fallbacks,
            "refreshes": self._refreshes,
            "evictions": self._evictions,
            "last_refresh_duration_seconds": self._last_refresh_duration_seconds,
            "last_error": self._last_error,
            "db_pool": self.db.status(),
        }

    def record_library_song(self, song: LibrarySong) -> None:
        self._touch()
        self._add_to_cache(song)

    async def _ensure_catalog(self) -> None:
        if self._catalog is not None and not self._is_stale():
            return
        async with self._lock:
            if self._catalog is not None and not self._is_stale():
                return
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = asyncio.create_task(self._refresh_catalog())
            task = self._refresh_task
        await task

    async def _refresh_catalog(self) -> None:
        started = self._now()
        rows = await self.db.fetch(
            f"""
            SELECT
                s.id,
                s.title,
                s.key,
                s.genre,
                s.cover_art_url,
                COALESCE(
                    array_agg(DISTINCT a.name) FILTER (WHERE a.name IS NOT NULL),
                    ARRAY[]::text[]
                ) AS artists
            FROM songs s
            LEFT JOIN song_artists sa ON s.id = sa.song_id
            LEFT JOIN artists a ON sa.artist_id = a.id
            WHERE {PLAYABLE_FULL_SONG_EXISTS_CLAUSE}
            GROUP BY s.id
            ORDER BY s.created_at DESC
            """
        )
        songs = [_song_from_row(row) for row in rows]
        by_title: dict[str, list[LibrarySong]] = {}
        for song in songs:
            by_title.setdefault(normalize_search_text(song.title), []).append(song)
        self._catalog = songs
        self._catalog_by_title = by_title
        self._loaded_at = self._now()
        self._last_refresh_duration_seconds = self._loaded_at - started
        self._last_error = None
        self._refreshes += 1
        logger.info("loaded library membership cache songs=%d", len(songs))

    async def _lookup_db_safely(self, title: str, artist: str) -> LibrarySong | None:
        self._db_fallbacks += 1
        try:
            return await self._lookup_db(title, artist)
        except Exception as exc:
            self._record_error(exc)
            return None

    async def _lookup_db(self, title: str, artist: str) -> LibrarySong | None:
        normalized_title = normalize_search_text(title)
        normalized_artist = normalize_search_text(artist)
        rows = await self.db.fetch(
            f"""
            SELECT
                s.id,
                s.title,
                s.key,
                s.genre,
                s.cover_art_url,
                COALESCE(
                    array_agg(DISTINCT a.name) FILTER (WHERE a.name IS NOT NULL),
                    ARRAY[]::text[]
                ) AS artists
            FROM songs s
            LEFT JOIN song_artists sa ON s.id = sa.song_id
            LEFT JOIN artists a ON sa.artist_id = a.id
            WHERE {PLAYABLE_FULL_SONG_EXISTS_CLAUSE}
              AND (
                LOWER(s.title) LIKE $1
                OR EXISTS (
                    SELECT 1
                    FROM song_artists sa_search
                    JOIN artists a_search ON sa_search.artist_id = a_search.id
                    WHERE sa_search.song_id = s.id
                      AND LOWER(a_search.name) LIKE $2
                )
              )
            GROUP BY s.id
            LIMIT 100
            """,
            f"%{normalized_title}%",
            f"%{normalized_artist}%",
        )
        self._last_error = None
        candidates = [_score_song(_song_from_row(row), title, artist) for row in rows]
        candidates = [song for song in candidates if (song.match_score or 0) >= DISCOVERY_MATCH_THRESHOLD]
        return max(candidates, key=lambda song: song.match_score or 0, default=None)

    def _lookup_cache(self, title: str, artist: str) -> LibrarySong | None:
        normalized_title = normalize_search_text(title)
        candidates = list(self._catalog_by_title.get(normalized_title, []))
        if not candidates and self._catalog is not None:
            candidates = [
                song
                for song in self._catalog
                if normalized_title in normalize_search_text(song.title)
                or normalize_search_text(song.title) in normalized_title
            ]
        scored = [_score_song(song, title, artist) for song in candidates]
        scored = [song for song in scored if (song.match_score or 0) >= DISCOVERY_MATCH_THRESHOLD]
        return max(scored, key=lambda song: song.match_score or 0, default=None)

    def _add_to_cache(self, song: LibrarySong) -> None:
        if self._catalog is None:
            return
        normalized_title = normalize_search_text(song.title)
        if all(existing.id != song.id for existing in self._catalog):
            self._catalog.append(song)
        bucket = self._catalog_by_title.setdefault(normalized_title, [])
        if all(existing.id != song.id for existing in bucket):
            bucket.append(song)

    def _is_stale(self) -> bool:
        return self._loaded_at is None or self._now() - self._loaded_at >= self.max_age_seconds

    def _touch(self) -> None:
        self._last_used_at = self._now()
        self._schedule_idle_eviction()

    def _schedule_idle_eviction(self) -> None:
        if self.idle_ttl_seconds <= 0:
            self._evict_cache()
            return
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
        try:
            self._eviction_task = asyncio.create_task(self._evict_after_idle())
        except RuntimeError:
            self._eviction_task = None

    async def _evict_after_idle(self) -> None:
        try:
            await asyncio.sleep(self.idle_ttl_seconds)
            if self._last_used_at and self._now() - self._last_used_at >= self.idle_ttl_seconds:
                self._evict_cache()
        except asyncio.CancelledError:
            return

    def _evict_cache(self) -> None:
        if self._catalog is not None:
            self._evictions += 1
        self._catalog = None
        self._catalog_by_title = {}
        self._loaded_at = None

    def _record_error(self, exc: Exception) -> None:
        self._last_error = str(exc)[:1000]
        logger.warning("library membership check failed: %s", self._last_error)


def normalize_search_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value).lower()
    normalized = re.sub(r"[\u0300-\u036f]", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _metadata_title(metadata: dict[str, Any] | None) -> str | None:
    value = (metadata or {}).get("title")
    return str(value).strip() if value else None


def _metadata_artist(metadata: dict[str, Any] | None) -> str | None:
    metadata = metadata or {}
    value = metadata.get("artist")
    if value:
        return str(value).strip()
    artists = metadata.get("artists")
    if isinstance(artists, list):
        for artist in artists:
            if isinstance(artist, dict):
                name = artist.get("name")
                if str(name or "").strip():
                    return str(name).strip()
            elif str(artist).strip():
                return str(artist).strip()
    return None


def _song_from_row(row: Any) -> LibrarySong:
    artists = [str(artist) for artist in (row["artists"] or []) if str(artist).strip()]
    return LibrarySong(
        id=str(row["id"]),
        title=str(row["title"] or ""),
        artists=artists,
        artist=artists[0] if artists else "",
        metadata={
            "key": row.get("key") if hasattr(row, "get") else row["key"],
            "genre": row.get("genre") if hasattr(row, "get") else row["genre"],
            "cover_art_url": row.get("cover_art_url") if hasattr(row, "get") else row["cover_art_url"],
        },
    )


def _score_song(song: LibrarySong, title: str, artist: str) -> LibrarySong:
    normalized_title = normalize_search_text(title)
    normalized_artist = normalize_search_text(artist)
    song_title = normalize_search_text(song.title)
    primary_artist = normalize_search_text(song.artist)
    all_artists = [normalize_search_text(value) for value in song.artists]
    score = 0
    if song_title == normalized_title:
        score += 700
    if primary_artist == normalized_artist:
        score += 620
    if normalized_artist in all_artists:
        score += 480
    if song_title.startswith(normalized_title) or normalized_title.startswith(song_title):
        score += 240
    if primary_artist and normalized_artist and (primary_artist in normalized_artist or normalized_artist in primary_artist):
        score += 180
    if song_title and normalized_title and (song_title in normalized_title or normalized_title in song_title):
        score += 140
    return LibrarySong(
        id=song.id,
        title=song.title,
        artists=song.artists,
        artist=song.artist,
        match_score=score,
        metadata=song.metadata,
    )
