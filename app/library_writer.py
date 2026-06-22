import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.identity import identity_key_from_artifacts
from app.library_membership import LibraryMembershipChecker, LibrarySong
from app.legacy.db import DBClient
from app.queue import JobRecord


@dataclass(frozen=True)
class LibraryPublishResult:
    enabled: bool
    song_id: str | None = None
    status: str | None = None
    inserted_stems: list[dict[str, Any]] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "song_id": str(self.song_id) if self.song_id is not None else None,
            "status": self.status,
            "inserted_stems": _json_safe(self.inserted_stems or []),
            "error": self.error,
        }


class LibraryWriter:
    def __init__(self, *, db: DBClient, settings: Settings, membership: LibraryMembershipChecker):
        self.db = db
        self.enabled = settings.library_precheck_enabled
        self.membership = membership

    async def publish_segment(
        self,
        *,
        job: JobRecord,
        segment: dict[str, Any],
        segment_result: dict[str, Any],
        status: str,
    ) -> LibraryPublishResult:
        if not self.enabled:
            return LibraryPublishResult(enabled=False)
        if _skip_library_write(job):
            return LibraryPublishResult(enabled=False, status="skipped_by_job")
        outputs = segment_result.get("outputs") or {}
        publishable = {
            _canonical_stem_type(stem): url
            for stem, url in outputs.items()
            if isinstance(url, str) and url and _canonical_stem_type(stem)
        }
        if not publishable:
            return LibraryPublishResult(enabled=True, status="no_publishable_stems")

        try:
            pool = await self.db.pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    identity_key = identity_key_from_artifacts(job.artifacts, job.payload)
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtext($1))",
                        f"gpu_ingestion_song:{identity_key}",
                    )
                    song_id = await self._ensure_song(conn, job)
                    artist_id = await self._main_artist_id(conn, song_id)
                    stem_rows = []
                    for stem_type in _ordered_stems(publishable):
                        stem_rows.append(
                            await self._upsert_stem(
                                conn,
                                song_id=song_id,
                                artist_id=artist_id,
                                stem_type=stem_type,
                                audio_url=publishable[stem_type],
                                segment=segment,
                                model_name="gpu-ingestion-htdemucs",
                            )
                        )
                    await self._update_ingestion_status(conn, song_id=song_id, job=job, status=status)
                    song = await self._song_for_cache(conn, song_id, status=status)
            if song:
                self.membership.record_library_song(song)
            return LibraryPublishResult(
                enabled=True,
                song_id=song_id,
                status=status,
                inserted_stems=[_json_safe(dict(row)) for row in stem_rows],
            )
        except Exception as exc:
            return LibraryPublishResult(enabled=True, status=status, error=str(exc)[:1000])

    async def mark_complete(self, *, job: JobRecord) -> LibraryPublishResult:
        if not self.enabled:
            return LibraryPublishResult(enabled=False)
        if _skip_library_write(job):
            return LibraryPublishResult(enabled=False, status="skipped_by_job")
        try:
            pool = await self.db.pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    identity_key = identity_key_from_artifacts(job.artifacts, job.payload)
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtext($1))",
                        f"gpu_ingestion_song:{identity_key}",
                    )
                    song_id = await self._ensure_song(conn, job)
                    await self._update_ingestion_status(conn, song_id=song_id, job=job, status="complete")
                    song = await self._song_for_cache(conn, song_id, status="complete")
            if song:
                self.membership.record_library_song(song)
            return LibraryPublishResult(enabled=True, song_id=song_id, status="complete")
        except Exception as exc:
            return LibraryPublishResult(enabled=True, status="complete", error=str(exc)[:1000])

    async def _ensure_song(self, conn: Any, job: JobRecord) -> str:
        existing_library_song_id = str(
            job.artifacts.get("existing_library_song_id")
            or job.payload.get("existing_library_song_id")
            or ""
        ).strip()
        if existing_library_song_id:
            existing = await conn.fetchrow("SELECT id FROM songs WHERE id = $1 LIMIT 1", existing_library_song_id)
            if existing:
                await self._refresh_song_metadata(conn, existing["id"], job)
                return str(existing["id"])

        identity_key = identity_key_from_artifacts(job.artifacts, job.payload)
        if identity_key and identity_key != "unknown":
            existing = await conn.fetchrow(
                """
                SELECT id
                FROM songs
                WHERE POSITION($1 IN COALESCE(analysis_json::text, '')) > 0
                ORDER BY created_at ASC
                LIMIT 1
                """,
                identity_key,
            )
            if existing:
                await self._refresh_song_metadata(conn, existing["id"], job)
                return str(existing["id"])

        youtube_url = job.artifacts.get("youtube_url")
        if youtube_url:
            existing = await conn.fetchrow("SELECT id FROM songs WHERE youtube_url = $1 LIMIT 1", youtube_url)
            if existing:
                await self._refresh_song_metadata(conn, existing["id"], job)
                return str(existing["id"])

        metadata = job.artifacts.get("spotify_metadata") or {}
        title = _title_from_job(job)
        artists = _artists_from_metadata(metadata)
        primary_artist = artists[0] if artists else "Unknown Artist"
        existing = await conn.fetchrow(
            """
            SELECT s.id
            FROM songs s
            JOIN song_artists sa ON s.id = sa.song_id
            JOIN artists a ON sa.artist_id = a.id
            WHERE LOWER(TRIM(s.title)) = LOWER(TRIM($1))
              AND LOWER(TRIM(a.name)) = LOWER(TRIM($2))
            LIMIT 1
            """,
            title,
            primary_artist,
        )
        if existing:
            await self._refresh_song_metadata(conn, existing["id"], job)
            return str(existing["id"])

        analysis = job.artifacts.get("analysis") or {}
        all_in_one_bpm, beat_analysis_bpm = _analysis_bpms(analysis)
        mix_id = await self._next_mix_id(conn)
        row = await conn.fetchrow(
            """
            INSERT INTO songs (
                title, cover_art_url, all_in_one_bpm, beat_analysis_bpm, key,
                audio_url, genre, youtube_url, analysis_json,
                cover_art_square_lowres, cover_art_square_medres, cover_art_square_highres,
                mix_id
            )
            VALUES ($1, $2, $3, $4, $5, NULLIF($6, ''), $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """,
            title,
            metadata.get("album_art_url") or metadata.get("album_art_highres"),
            all_in_one_bpm,
            beat_analysis_bpm,
            _analysis_key(analysis, metadata),
            str(job.artifacts.get("source_audio_url") or ""),
            _analysis_genre(analysis, metadata),
            youtube_url,
            json.dumps(_analysis_payload(job, status="partial")),
            metadata.get("album_art_lowres") or metadata.get("album_art_url"),
            metadata.get("album_art_medres") or metadata.get("album_art_url"),
            metadata.get("album_art_highres") or metadata.get("album_art_url"),
            mix_id,
        )
        song_id = str(row["id"])
        for index, artist_name in enumerate(artists or [primary_artist]):
            artist_id = await self._ensure_artist(conn, artist_name)
            await conn.execute(
                """
                INSERT INTO song_artists (song_id, artist_id, artist_role)
                VALUES ($1, $2, $3)
                ON CONFLICT (song_id, artist_id, artist_role) DO NOTHING
                """,
                song_id,
                artist_id,
                "main" if index == 0 else "featured",
            )
        return song_id

    async def _refresh_song_metadata(self, conn: Any, song_id: str, job: JobRecord) -> None:
        metadata = job.artifacts.get("spotify_metadata") or {}
        analysis = job.artifacts.get("analysis") or {}
        all_in_one_bpm, beat_analysis_bpm = _analysis_bpms(analysis)
        await conn.execute(
            """
            UPDATE songs
            SET cover_art_url = COALESCE(NULLIF($2, ''), cover_art_url),
                cover_art_square_lowres = COALESCE(NULLIF($3, ''), cover_art_square_lowres),
                cover_art_square_medres = COALESCE(NULLIF($4, ''), cover_art_square_medres),
                cover_art_square_highres = COALESCE(NULLIF($5, ''), cover_art_square_highres),
                all_in_one_bpm = COALESCE($6, all_in_one_bpm),
                beat_analysis_bpm = COALESCE($7, beat_analysis_bpm),
                key = COALESCE(NULLIF($8, ''), key),
                youtube_url = COALESCE(NULLIF($9, ''), youtube_url),
                audio_url = COALESCE(NULLIF($10, ''), audio_url),
                genre = COALESCE(NULLIF($11, ''), genre)
            WHERE id = $1
            """,
            song_id,
            metadata.get("album_art_url") or metadata.get("album_art_highres") or "",
            metadata.get("album_art_lowres") or metadata.get("album_art_url") or "",
            metadata.get("album_art_medres") or metadata.get("album_art_url") or "",
            metadata.get("album_art_highres") or metadata.get("album_art_url") or "",
            all_in_one_bpm,
            beat_analysis_bpm,
            _analysis_key(analysis, metadata),
            str(job.artifacts.get("youtube_url") or ""),
            str(job.artifacts.get("source_audio_url") or ""),
            _analysis_genre(analysis, metadata),
        )

    async def _ensure_artist(self, conn: Any, name: str) -> str:
        row = await conn.fetchrow(
            """
            INSERT INTO artists (name)
            VALUES ($1)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            name,
        )
        return str(row["id"])

    async def _main_artist_id(self, conn: Any, song_id: str) -> str | None:
        row = await conn.fetchrow(
            """
            SELECT artist_id
            FROM song_artists
            WHERE song_id = $1 AND artist_role = 'main'
            LIMIT 1
            """,
            song_id,
        )
        return str(row["artist_id"]) if row else None

    async def _next_mix_id(self, conn: Any) -> int:
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_ingestion_mix_id'))")
        value = await conn.fetchval("SELECT COALESCE(MAX(mix_id), 99999) + 1 FROM songs")
        return int(value)

    async def _upsert_stem(
        self,
        conn: Any,
        *,
        song_id: str,
        artist_id: str | None,
        stem_type: str,
        audio_url: str,
        segment: dict[str, Any],
        model_name: str,
    ) -> Any:
        segment_label = _segment_label(segment)
        start_time = _number_or_none(segment.get("start"))
        end_time = _number_or_none(segment.get("end"))
        label = f"{segment_label.title()}_section_{segment.get('id', 'segment')} ({stem_type.title()})"
        existing = await conn.fetchrow(
            """
            SELECT id
            FROM stems
            WHERE song_id = $1
              AND stem_type = $2
              AND segment = $3
              AND ROUND(COALESCE(start_time, -1)::numeric, 3) = ROUND(COALESCE($4::numeric, -1), 3)
              AND ROUND(COALESCE(end_time, -1)::numeric, 3) = ROUND(COALESCE($5::numeric, -1), 3)
            LIMIT 1
            """,
            song_id,
            stem_type,
            segment_label,
            start_time,
            end_time,
        )
        if existing:
            return await conn.fetchrow(
                """
                UPDATE stems
                SET audio_url = $2,
                    label = $3,
                    model = $4,
                    is_original = true,
                    is_full_song = false,
                    has_audio = true,
                    is_usable = true,
                    artist_id = COALESCE($5, artist_id)
                WHERE id = $1
                RETURNING id, song_id, stem_type, audio_url, segment, start_time, end_time
                """,
                existing["id"],
                audio_url,
                label,
                model_name,
                artist_id,
            )
        return await conn.fetchrow(
            """
            INSERT INTO stems (
                song_id, stem_type, audio_url, is_original, label,
                start_time, end_time, model, segment, is_full_song, artist_id, has_audio, is_usable
            )
            VALUES ($1, $2, $3, true, $4, $5, $6, $7, $8, false, $9, true, true)
            RETURNING id, song_id, stem_type, audio_url, segment, start_time, end_time
            """,
            song_id,
            stem_type,
            audio_url,
            label,
            start_time,
            end_time,
            model_name,
            segment_label,
            artist_id,
        )

    async def _update_ingestion_status(self, conn: Any, *, song_id: str, job: JobRecord, status: str) -> None:
        existing = await conn.fetchval("SELECT analysis_json FROM songs WHERE id = $1", song_id)
        payload = _safe_json(existing)
        payload.update(_analysis_payload(job, status=status))
        available = await conn.fetch(
            """
            SELECT stem_type, segment, audio_url, start_time, end_time
            FROM stems
            WHERE song_id = $1 AND COALESCE(audio_url, '') <> ''
            ORDER BY created_at ASC
            """,
            song_id,
        )
        payload["gpu_ingestion"]["available_stems"] = [dict(row) for row in available]
        payload["gpu_ingestion"]["available_stem_types"] = sorted({row["stem_type"] for row in available})
        all_in_one_bpm, beat_analysis_bpm = _analysis_bpms(job.artifacts.get("analysis") or {})
        await conn.execute(
            """
            UPDATE songs
            SET analysis_json = $2,
                all_in_one_bpm = COALESCE($3, all_in_one_bpm),
                beat_analysis_bpm = COALESCE($4, beat_analysis_bpm),
                audio_url = COALESCE(NULLIF($5, ''), audio_url),
                key = COALESCE(NULLIF($6, ''), key),
                genre = COALESCE(NULLIF($7, ''), genre)
            WHERE id = $1
            """,
            song_id,
            json.dumps(payload),
            all_in_one_bpm,
            beat_analysis_bpm,
            str(job.artifacts.get("source_audio_url") or ""),
            _analysis_key(job.artifacts.get("analysis") or {}, job.artifacts.get("spotify_metadata") or {}),
            _analysis_genre(job.artifacts.get("analysis") or {}, job.artifacts.get("spotify_metadata") or {}),
        )

    async def _song_for_cache(self, conn: Any, song_id: str, *, status: str) -> LibrarySong | None:
        row = await conn.fetchrow(
            """
            SELECT
                s.id,
                s.title,
                COALESCE(
                    array_agg(DISTINCT a.name) FILTER (WHERE a.name IS NOT NULL),
                    ARRAY[]::text[]
                ) AS artists
            FROM songs s
            LEFT JOIN song_artists sa ON s.id = sa.song_id
            LEFT JOIN artists a ON sa.artist_id = a.id
            WHERE s.id = $1
            GROUP BY s.id
            """,
            song_id,
        )
        if not row:
            return None
        artists = [str(artist) for artist in row["artists"] if str(artist).strip()]
        return LibrarySong(
            id=str(row["id"]),
            title=str(row["title"]),
            artists=artists,
            artist=artists[0] if artists else "",
            ingestion_status=status,
            metadata={"partial": status != "complete"},
        )


def _canonical_stem_type(stem: str) -> str | None:
    normalized = stem.strip().lower()
    mapping = {
        "vocals": "voice",
        "vocal": "voice",
        "voice": "voice",
        "drums": "beat",
        "drum": "beat",
        "beat": "beat",
        "bass": "bass",
        "other": "chord",
        "chords": "chord",
        "chord": "chord",
        "instrumental": "chord",
    }
    return mapping.get(normalized)


def _ordered_stems(stems: dict[str, str]) -> list[str]:
    priority = {"chord": 0, "beat": 1, "bass": 2, "voice": 3}
    return sorted(stems, key=lambda stem: priority.get(stem, 99))


def _segment_label(segment: dict[str, Any]) -> str:
    raw = str(segment.get("label") or "segment").lower().strip()
    if "chorus" in raw:
        return "chorus"
    if "verse" in raw:
        return "verse"
    if "intro" in raw:
        return "intro"
    if "outro" in raw:
        return "outro"
    return raw.replace(" ", "_") or "segment"


def _title_from_job(job: JobRecord) -> str:
    metadata = job.artifacts.get("spotify_metadata") or {}
    return str(metadata.get("title") or job.artifacts.get("source") or job.payload.get("source") or "Unknown Title")


def _artists_from_metadata(metadata: dict[str, Any]) -> list[str]:
    raw_artists = metadata.get("artists")
    if isinstance(raw_artists, list):
        artists = []
        for artist in raw_artists:
            if isinstance(artist, dict):
                name = artist.get("name")
            else:
                name = artist
            if str(name or "").strip():
                artists.append(str(name).strip())
        if artists:
            return artists
    artist = metadata.get("artist")
    return [str(artist).strip()] if str(artist or "").strip() else []


def _analysis_bpms(analysis: dict[str, Any]) -> tuple[float | None, float | None]:
    bpm = _first_number(
        analysis.get("all_in_one_bpm"),
        analysis.get("tempo_bpm"),
        analysis.get("tempo"),
        analysis.get("bpm"),
        _nested_value(analysis.get("bpm"), "all_in_one"),
        _nested_value(analysis.get("bpm"), "value"),
        _nested_value(analysis.get("tempo"), "bpm"),
    )
    beat_bpm = _first_number(
        analysis.get("beat_analysis_bpm"),
        analysis.get("beat_bpm"),
        _nested_value(analysis.get("bpm"), "beat_analysis"),
        _nested_value(analysis.get("beats"), "bpm"),
        _estimate_bpm_from_beats(
            analysis.get("beats")
            or analysis.get("beat_times")
            or analysis.get("beat_timestamps")
        ),
    )
    return bpm or beat_bpm, beat_bpm or bpm


def _analysis_key(analysis: dict[str, Any], metadata: dict[str, Any]) -> str:
    for value in (
        analysis.get("key"),
        analysis.get("detected_key"),
        analysis.get("musical_key"),
        _nested_value(analysis.get("key"), "name"),
        _nested_value(analysis.get("key"), "value"),
        metadata.get("key"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _analysis_genre(analysis: dict[str, Any], metadata: dict[str, Any]) -> str:
    for value in (
        metadata.get("genre"),
        analysis.get("genre"),
        analysis.get("detected_genre"),
        _nested_value(analysis.get("genre"), "name"),
        _nested_value(analysis.get("genre"), "value"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    genres = metadata.get("genres") or analysis.get("genres")
    if isinstance(genres, list):
        for value in genres:
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _nested_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number_or_none(value)
        if number is not None and number > 0:
            return number
    return None


def _estimate_bpm_from_beats(value: Any) -> float | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    positions = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("time") or item.get("start") or item.get("position")
        number = _number_or_none(item)
        if number is not None:
            positions.append(number)
    if len(positions) < 3:
        return None
    intervals = [
        positions[index + 1] - positions[index]
        for index in range(len(positions) - 1)
        if positions[index + 1] > positions[index]
    ]
    if not intervals:
        return None
    average_interval = sum(intervals) / len(intervals)
    if average_interval <= 0:
        return None
    bpm = 60.0 / average_interval
    while bpm < 70:
        bpm *= 2
    while bpm > 210:
        bpm /= 2
    return round(bpm, 3)


def _analysis_payload(job: JobRecord, *, status: str) -> dict[str, Any]:
    analysis = job.artifacts.get("analysis") or {}
    all_in_one_bpm, beat_analysis_bpm = _analysis_bpms(analysis)
    return {
        "analysis": analysis,
        "gpu_ingestion": {
            "job_id": job.id,
            "job_type": job.job_type.value,
            "identity_key": identity_key_from_artifacts(job.artifacts, job.payload),
            "source": job.artifacts.get("source"),
            "youtube_url": job.artifacts.get("youtube_url"),
            "source_audio_url": job.artifacts.get("source_audio_url"),
            "source_audio_upload_error": job.artifacts.get("source_audio_upload_error"),
            "analyzer_result_url": job.artifacts.get("analyzer_result_url"),
            "existing_library_song_id": job.artifacts.get("existing_library_song_id"),
            "spotify_metadata": job.artifacts.get("spotify_metadata"),
            "youtube_match": job.artifacts.get("youtube_match"),
            "summary": {
                "all_in_one_bpm": all_in_one_bpm,
                "beat_analysis_bpm": beat_analysis_bpm,
                "bpm": all_in_one_bpm or beat_analysis_bpm,
                "key": _analysis_key(analysis, job.artifacts.get("spotify_metadata") or {}),
                "genre": _analysis_genre(analysis, job.artifacts.get("spotify_metadata") or {}),
                "segment_count": len(analysis.get("segments") or []),
                "duration": _number_or_none(analysis.get("duration")),
                "beat_count": len(analysis.get("beats") or []),
                "downbeat_count": len(analysis.get("downbeats") or []),
                "upbeat_count": len(analysis.get("upbeats") or []),
            },
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _skip_library_write(job: JobRecord) -> bool:
    value = job.payload.get("skip_library_write", job.artifacts.get("skip_library_write"))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _safe_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {"legacy_analysis_json": value}


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
