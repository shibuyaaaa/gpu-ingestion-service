from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.legacy.utils.source import _format_spotify_track, _get_spotify_track
from app.tools.gcs_inventory import GpuIngestionObject, parse_gpu_ingestion_uri


CANONICAL_STEMS = ("chord", "beat", "bass", "voice")


@dataclass
class SegmentBackfill:
    original_segment_id: str
    segment: str
    stems: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return set(CANONICAL_STEMS) <= set(self.stems)


@dataclass
class SongBackfill:
    spotify_id: str
    crawler_session_ids: set[str] = field(default_factory=set)
    job_ids: set[str] = field(default_factory=set)
    segments: list[SegmentBackfill] = field(default_factory=list)

    @property
    def complete_segments(self) -> list[SegmentBackfill]:
        return [segment for segment in self.segments if segment.complete]


@dataclass
class BackfillStats:
    songs_seen: int = 0
    songs_skipped: int = 0
    songs_inserted: int = 0
    songs_updated: int = 0
    stems_inserted: int = 0
    stems_updated: int = 0
    segments_seen: int = 0
    segments_backfilled: int = 0
    metadata_failures: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "songs_seen": self.songs_seen,
            "songs_skipped": self.songs_skipped,
            "songs_inserted": self.songs_inserted,
            "songs_updated": self.songs_updated,
            "stems_inserted": self.stems_inserted,
            "stems_updated": self.stems_updated,
            "segments_seen": self.segments_seen,
            "segments_backfilled": self.segments_backfilled,
            "metadata_failures": self.metadata_failures,
            "errors": self.errors[:25],
        }


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cdn_url_from_gcs_uri(uri: str, *, cdn_base_url: str) -> str:
    match = re.match(r"^gs://[^/]+/(?P<path>.+)$", uri)
    if not match:
        raise ValueError(f"not a gs:// URI: {uri}")
    return f"{cdn_base_url.rstrip('/')}/{match.group('path')}"


def load_inventory(path: Path, *, cdn_base_url: str) -> list[GpuIngestionObject]:
    rows: list[GpuIngestionObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if "uri" in parsed:
            row = parse_gpu_ingestion_uri(str(parsed["uri"]))
        else:
            row = parse_gpu_ingestion_uri(line)
        if row and row.spotify_id:
            rows.append(row)
    return rows


def group_inventory(rows: list[GpuIngestionObject], *, cdn_base_url: str) -> list[SongBackfill]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[str, SongBackfill] = {}
    for row in rows:
        if not row.spotify_id:
            continue
        song = meta.setdefault(row.spotify_id, SongBackfill(spotify_id=row.spotify_id))
        song.job_ids.add(row.job_id)
        if row.crawler_session_id:
            song.crawler_session_ids.add(row.crawler_session_id)
        grouped[row.spotify_id][row.segment_id][row.canonical_stem] = cdn_url_from_gcs_uri(
            row.uri,
            cdn_base_url=cdn_base_url,
        )

    songs: list[SongBackfill] = []
    for spotify_id, segment_map in grouped.items():
        song = meta[spotify_id]
        for index, (segment_id, stems) in enumerate(sorted(segment_map.items(), key=lambda item: _segment_sort_key(item[0]))):
            song.segments.append(
                SegmentBackfill(
                    original_segment_id=segment_id,
                    segment="chorus" if index == 0 else segment_id.replace("-", "_"),
                    stems=dict(stems),
                )
            )
        songs.append(song)
    return sorted(songs, key=lambda song: song.spotify_id)


def _segment_sort_key(segment_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", segment_id)
    return (int(match.group(1)) if match else 10**9, segment_id)


class SpotifyMetadataCache:
    def __init__(self, path: Path, *, source: str = "embed"):
        self.path = path
        self.source = source
        self.values: dict[str, dict[str, Any]] = {}
        if path.exists():
            self.values = json.loads(path.read_text(encoding="utf-8"))

    async def get(self, spotify_id: str) -> dict[str, Any]:
        if spotify_id not in self.values:
            self.values[spotify_id] = await self._fetch(spotify_id)
            self.flush()
        return self.values[spotify_id]

    async def _fetch(self, spotify_id: str) -> dict[str, Any]:
        if self.source == "spotify":
            return _format_spotify_track(await _get_spotify_track(spotify_id))
        if self.source == "embed":
            return await fetch_embed_metadata(spotify_id)
        try:
            return _format_spotify_track(await _get_spotify_track(spotify_id))
        except Exception:
            return await fetch_embed_metadata(spotify_id)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.values, indent=2, sort_keys=True), encoding="utf-8")


async def fetch_embed_metadata(spotify_id: str) -> dict[str, Any]:
    import httpx

    oembed_url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/track/{spotify_id}"
    embed_url = f"https://open.spotify.com/embed/track/{spotify_id}?utm_source=oembed"
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        oembed_response = await client.get(oembed_url)
        oembed_response.raise_for_status()
        oembed = oembed_response.json()
        embed_response = await client.get(embed_url)
        embed_response.raise_for_status()
        embed_html = embed_response.text

    artists = _artists_from_embed_html(embed_html)
    title = str(oembed.get("title") or "").strip()
    return {
        "spotify_id": spotify_id,
        "title": title or spotify_id,
        "artist": artists[0] if artists else "",
        "artists": artists,
        "album": "",
        "duration_ms": 0,
        "album_art_url": oembed.get("thumbnail_url"),
        "album_art_highres": oembed.get("thumbnail_url"),
        "album_art_medres": oembed.get("thumbnail_url"),
        "album_art_lowres": oembed.get("thumbnail_url"),
        "isrc": None,
        "popularity": 0,
        "metadata_source": "spotify_embed",
    }


def _artists_from_embed_html(html: str) -> list[str]:
    match = re.search(r'"artists":\[(?P<artists>.*?)\]', html)
    if not match:
        return []
    try:
        artists = json.loads(f"[{match.group('artists')}]")
    except json.JSONDecodeError:
        return []
    names = [str(artist.get("name") or "").strip() for artist in artists if isinstance(artist, dict)]
    return [name for name in names if name]


class LibraryBackfillWriter:
    def __init__(self, conn: Any):
        self.conn = conn

    async def backfill_song(self, song: SongBackfill, metadata: dict[str, Any]) -> dict[str, int | str]:
        async with self.conn.transaction():
            song_id, song_action = await self._ensure_song(song, metadata)
            artist_id = await self._main_artist_id(song_id)
            inserted, updated = await self._upsert_stems_for_song(
                song_id=song_id,
                artist_id=artist_id,
                segments=song.complete_segments,
            )
            await self._update_analysis_json(song_id, song, metadata)
        return {
            "song_id": song_id,
            "song_action": song_action,
            "stems_inserted": inserted,
            "stems_updated": updated,
        }

    async def _ensure_song(self, song: SongBackfill, metadata: dict[str, Any]) -> tuple[str, str]:
        title = str(metadata.get("title") or song.spotify_id)
        artists = _artists_from_metadata(metadata)
        primary_artist = artists[0] if artists else "Unknown Artist"
        existing = await self.conn.fetchrow(
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
            await self._refresh_song(str(existing["id"]), song, metadata)
            return str(existing["id"]), "updated"

        mix_id = await self._next_mix_id()
        row = await self.conn.fetchrow(
            """
            INSERT INTO songs (
                title, cover_art_url, all_in_one_bpm, beat_analysis_bpm, key,
                audio_url, genre, youtube_url, analysis_json,
                cover_art_square_lowres, cover_art_square_medres, cover_art_square_highres,
                mix_id
            )
            VALUES ($1, $2, NULL, NULL, $3, NULL, $4, NULL, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            title,
            metadata.get("album_art_url") or metadata.get("album_art_highres"),
            "C major",
            "",
            json.dumps(_analysis_payload(song, metadata)),
            metadata.get("album_art_lowres") or metadata.get("album_art_url"),
            metadata.get("album_art_medres") or metadata.get("album_art_url"),
            metadata.get("album_art_highres") or metadata.get("album_art_url"),
            mix_id,
        )
        song_id = str(row["id"])
        for index, artist_name in enumerate(artists or [primary_artist]):
            artist_id = await self._ensure_artist(artist_name)
            await self.conn.execute(
                """
                INSERT INTO song_artists (song_id, artist_id, artist_role)
                VALUES ($1, $2, $3)
                ON CONFLICT (song_id, artist_id, artist_role) DO NOTHING
                """,
                song_id,
                artist_id,
                "main" if index == 0 else "featured",
            )
        return song_id, "inserted"

    async def _refresh_song(self, song_id: str, song: SongBackfill, metadata: dict[str, Any]) -> None:
        await self.conn.execute(
            """
            UPDATE songs
            SET cover_art_url = COALESCE(NULLIF($2, ''), cover_art_url),
                cover_art_square_lowres = COALESCE(NULLIF($3, ''), cover_art_square_lowres),
                cover_art_square_medres = COALESCE(NULLIF($4, ''), cover_art_square_medres),
                cover_art_square_highres = COALESCE(NULLIF($5, ''), cover_art_square_highres)
            WHERE id = $1
            """,
            song_id,
            metadata.get("album_art_url") or metadata.get("album_art_highres") or "",
            metadata.get("album_art_lowres") or metadata.get("album_art_url") or "",
            metadata.get("album_art_medres") or metadata.get("album_art_url") or "",
            metadata.get("album_art_highres") or metadata.get("album_art_url") or "",
        )

    async def _ensure_artist(self, name: str) -> str:
        row = await self.conn.fetchrow(
            """
            INSERT INTO artists (name)
            VALUES ($1)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            name,
        )
        return str(row["id"])

    async def _main_artist_id(self, song_id: str) -> str | None:
        row = await self.conn.fetchrow(
            """
            SELECT artist_id
            FROM song_artists
            WHERE song_id = $1 AND artist_role = 'main'
            LIMIT 1
            """,
            song_id,
        )
        return str(row["artist_id"]) if row else None

    async def _next_mix_id(self) -> int:
        await self.conn.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_ingestion_mix_id'))")
        value = await self.conn.fetchval("SELECT COALESCE(MAX(mix_id), 99999) + 1 FROM songs")
        return int(value)

    async def _upsert_stems_for_song(
        self,
        *,
        song_id: str,
        artist_id: str | None,
        segments: list[SegmentBackfill],
    ) -> tuple[int, int]:
        rows: list[tuple[str, str, str, str, str, str | None]] = []
        for segment in segments:
            for stem_type in CANONICAL_STEMS:
                label = f"{segment.segment.title()}_section_{segment.original_segment_id} ({stem_type.title()})"
                rows.append(
                    (
                        stem_type,
                        segment.stems[stem_type],
                        label,
                        segment.original_segment_id,
                        segment.segment,
                        artist_id,
                    )
                )
        urls = [row[1] for row in rows]
        existing_urls = {
            str(row["audio_url"])
            for row in await self.conn.fetch(
                "SELECT audio_url FROM stems WHERE audio_url = ANY($1::text[])",
                urls,
            )
        }
        update_rows = [(song_id, *row) for row in rows if row[1] in existing_urls]
        insert_rows = [(song_id, *row) for row in rows if row[1] not in existing_urls]

        if update_rows:
            await self.conn.executemany(
                """
                UPDATE stems
                SET song_id = $1,
                    stem_type = $2,
                    audio_url = $3,
                    is_original = true,
                    label = $4,
                    label_alt = $5,
                    model = 'gpu-ingestion-gcs-backfill',
                    segment = $6,
                    is_full_song = false,
                    artist_id = COALESCE($7, artist_id),
                    is_usable = true,
                    has_audio = true
                WHERE audio_url = $3
                """,
                update_rows,
            )

        if insert_rows:
            await self.conn.executemany(
                """
                INSERT INTO stems (
                    song_id, stem_type, audio_url, is_original, label, label_alt,
                    model, segment, is_full_song, artist_id, is_usable, has_audio
                )
                VALUES ($1, $2, $3, true, $4, $5, 'gpu-ingestion-gcs-backfill', $6, false, $7, true, true)
                """,
                insert_rows,
            )

        return len(insert_rows), len(update_rows)

    async def _update_analysis_json(self, song_id: str, song: SongBackfill, metadata: dict[str, Any]) -> None:
        existing = await self.conn.fetchval("SELECT analysis_json FROM songs WHERE id = $1", song_id)
        rows = await self.conn.fetch(
            """
            SELECT stem_type, segment, audio_url, start_time, end_time, label_alt
            FROM stems
            WHERE song_id = $1 AND COALESCE(audio_url, '') <> ''
            ORDER BY created_at ASC
            """,
            song_id,
        )
        await self.conn.execute(
            "UPDATE songs SET analysis_json = $2 WHERE id = $1",
            song_id,
            json.dumps(
                _merge_analysis_payload(
                    existing,
                    _analysis_payload(song, metadata, available_stems=[dict(row) for row in rows]),
                )
            ),
        )


def _artists_from_metadata(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("artists")
    if isinstance(raw, list):
        names = [str(item.get("name") if isinstance(item, dict) else item).strip() for item in raw]
        names = [name for name in names if name]
        if names:
            return names
    artist = str(metadata.get("artist") or "").strip()
    return [artist] if artist else []


def _analysis_payload(
    song: SongBackfill,
    metadata: dict[str, Any],
    *,
    available_stems: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "spotify": {
            "id": song.spotify_id,
            "album": metadata.get("album"),
            "isrc": metadata.get("isrc"),
            "popularity": metadata.get("popularity"),
            "duration_ms": metadata.get("duration_ms"),
        },
        "gpu_ingestion": {
            "status": "backfilled",
            "source": "gcs_existing_artifacts",
            "spotify_id": song.spotify_id,
            "crawler_session_ids": sorted(song.crawler_session_ids),
            "job_ids": sorted(song.job_ids),
            "segments_seen": len(song.segments),
            "segments_backfilled": len(song.complete_segments),
            "original_segment_ids": [segment.original_segment_id for segment in song.complete_segments],
            "available_stem_types": list(CANONICAL_STEMS),
            "available_stems": available_stems or [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _merge_analysis_payload(existing: str | None, patch: dict[str, Any]) -> dict[str, Any]:
    try:
        merged = json.loads(existing) if existing else {}
        if not isinstance(merged, dict):
            merged = {"legacy_analysis_json": existing}
    except Exception:
        merged = {"legacy_analysis_json": existing}
    merged.update(patch)
    return merged


async def run_backfill(args: argparse.Namespace) -> BackfillStats:
    load_dotenv(args.env_file)
    database_url = args.database_url or os.getenv("PROD_DATABASE_URL") or os.getenv("DATABASE_URL")
    if args.apply and not database_url:
        raise RuntimeError("DATABASE_URL or PROD_DATABASE_URL is required for --apply")

    rows = load_inventory(args.inventory, cdn_base_url=args.cdn_base_url)
    songs = group_inventory(rows, cdn_base_url=args.cdn_base_url)
    if args.limit:
        songs = songs[: args.limit]
    stats = BackfillStats(songs_seen=len(songs), segments_seen=sum(len(song.segments) for song in songs))

    metadata_cache = SpotifyMetadataCache(args.metadata_cache, source=args.metadata_source)
    conn = None
    if args.apply:
        import asyncpg

        conn = await asyncpg.connect(database_url)
    try:
        writer = LibraryBackfillWriter(conn) if conn else None
        for song in songs:
            if not song.complete_segments:
                stats.songs_skipped += 1
                continue
            try:
                metadata = await metadata_cache.get(song.spotify_id)
            except Exception as exc:
                stats.metadata_failures += 1
                stats.errors.append(f"{song.spotify_id}: metadata: {str(exc)[:200]}")
                continue

            if not args.apply:
                stats.segments_backfilled += len(song.complete_segments)
                continue

            try:
                assert writer is not None
                result = await writer.backfill_song(song, metadata)
                if result["song_action"] == "inserted":
                    stats.songs_inserted += 1
                else:
                    stats.songs_updated += 1
                stats.stems_inserted += int(result["stems_inserted"])
                stats.stems_updated += int(result["stems_updated"])
                stats.segments_backfilled += len(song.complete_segments)
            except Exception as exc:
                stats.errors.append(f"{song.spotify_id}: write: {str(exc)[:500]}")
    finally:
        if conn:
            await conn.close()
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill old gpu-ingestion GCS artifacts into Aime Postgres.")
    parser.add_argument("--inventory", type=Path, default=Path("docs/reports/gpu_ingestion_gcs_inventory.jsonl"))
    parser.add_argument("--metadata-cache", type=Path, default=Path("docs/reports/gpu_ingestion_spotify_metadata.json"))
    parser.add_argument("--env-file", type=Path, default=Path("../ingestion-api/.env"))
    parser.add_argument("--database-url", default="")
    parser.add_argument("--cdn-base-url", default=os.getenv("CDN_BASE_URL", "https://cdn.shibuyaaa.com"))
    parser.add_argument("--metadata-source", choices=["embed", "spotify", "auto"], default="embed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = asyncio.run(run_backfill(args))
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    return 1 if stats.errors and args.apply else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
