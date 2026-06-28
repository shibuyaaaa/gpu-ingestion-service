from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.legacy.utils.source import _format_spotify_track, _get_spotify_track, _search_spotify_tracks, _spotify_get_json
from app.library_writer import upsert_song_album_metadata
from app.tools.gcs_backfill import load_dotenv


SPOTIFY_TRACK_RE = re.compile(r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]+)")


@dataclass(frozen=True)
class SongAlbumSnapshot:
    id: str
    title: str
    artists: list[str]
    analysis_json: dict[str, Any] = field(default_factory=dict)
    analysis_json_invalid: bool = False
    has_album_link: bool = False

    @classmethod
    def from_row(cls, row: Any) -> "SongAlbumSnapshot":
        parsed, invalid = _parse_json_object(_row_get(row, "analysis_json"))
        return cls(
            id=str(_row_get(row, "id")),
            title=str(_row_get(row, "title") or ""),
            artists=[str(item) for item in (_row_get(row, "artists") or []) if str(item or "").strip()],
            analysis_json=parsed,
            analysis_json_invalid=invalid,
            has_album_link=bool(_row_get(row, "has_album_link")),
        )


@dataclass(frozen=True)
class AlbumBackfillAction:
    song_id: str
    title: str
    artists: list[str]
    action_type: str
    source: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    review_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def album_key(self) -> str | None:
        album_id = _clean_text(self.metadata.get("album_id") or self.metadata.get("source_album_id"))
        return f"spotify:{album_id}" if album_id else None

    @property
    def auto_apply(self) -> bool:
        return self.action_type == "upsert_album_link" and self.album_key is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "song_id": self.song_id,
            "title": self.title,
            "artists": self.artists,
            "action_type": self.action_type,
            "source": self.source,
            "reason": self.reason,
            "album_key": self.album_key,
            "metadata": _json_safe(self.metadata),
            "review_candidates": _json_safe(self.review_candidates),
        }


@dataclass
class AlbumBackfillStats:
    songs_scanned: int = 0
    actions_planned: int = 0
    actions_applied: int = 0
    actions_failed: int = 0
    unique_albums_planned: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "songs_scanned": self.songs_scanned,
            "actions_planned": self.actions_planned,
            "actions_applied": self.actions_applied,
            "actions_failed": self.actions_failed,
            "unique_albums_planned": self.unique_albums_planned,
            "action_counts": dict(self.action_counts),
            "source_counts": dict(self.source_counts),
            "errors": self.errors[:50],
        }


class SpotifyTrackCache:
    def __init__(self, path: Path):
        self.path = path
        self.values: dict[str, dict[str, Any]] = {}
        if path.exists():
            self.values = json.loads(path.read_text(encoding="utf-8"))

    async def get(self, spotify_id: str) -> dict[str, Any]:
        if spotify_id not in self.values:
            self.values[spotify_id] = _format_spotify_track(await _get_spotify_track(spotify_id))
            self.flush()
        return dict(self.values[spotify_id])

    async def preload(self, spotify_ids: list[str], *, batch_size: int = 50) -> None:
        missing = self.missing_ids(spotify_ids)
        if not missing:
            return
        for index in range(0, len(missing), batch_size):
            batch = missing[index : index + batch_size]
            fetched = await _get_spotify_tracks(batch)
            for spotify_id in batch:
                self.values[spotify_id] = fetched.get(spotify_id) or {"spotify_id": spotify_id}
            self.flush()

    def missing_ids(self, spotify_ids: list[str]) -> list[str]:
        seen = set()
        missing = []
        for value in spotify_ids:
            spotify_id = _clean_text(value)
            if not spotify_id or spotify_id in seen or spotify_id in self.values:
                continue
            seen.add(spotify_id)
            missing.append(spotify_id)
        return missing

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.values, indent=2, sort_keys=True), encoding="utf-8")


async def _get_spotify_tracks(track_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not track_ids:
        return {}
    try:
        data = await _spotify_get_json(
            "https://api.spotify.com/v1/tracks",
            params={"ids": ",".join(track_ids[:50])},
        )
    except Exception:
        if len(track_ids) == 1:
            return {}
        midpoint = max(1, len(track_ids) // 2)
        return {
            **await _get_spotify_tracks(track_ids[:midpoint]),
            **await _get_spotify_tracks(track_ids[midpoint:]),
        }
    tracks: dict[str, dict[str, Any]] = {}
    for track in data.get("tracks") or []:
        if not isinstance(track, dict) or not track.get("id"):
            continue
        tracks[str(track["id"])] = _format_spotify_track(track)
    return tracks


async def plan_album_action(
    song: SongAlbumSnapshot,
    cache: SpotifyTrackCache,
    *,
    allow_review_search: bool = True,
    allow_search_auto_apply: bool = False,
) -> AlbumBackfillAction:
    metadata = _metadata_from_analysis(song.analysis_json)
    spotify_id = _spotify_id_from_song(song, metadata)
    if _has_spotify_album_identity(metadata):
        return AlbumBackfillAction(
            song_id=song.id,
            title=song.title,
            artists=song.artists,
            action_type="upsert_album_link",
            source="analysis_json",
            reason="analysis_json already contains Spotify album identity",
            metadata=metadata,
        )

    if spotify_id:
        try:
            resolved = await cache.get(spotify_id)
        except Exception as exc:
            return AlbumBackfillAction(
                song_id=song.id,
                title=song.title,
                artists=song.artists,
                action_type="fetch_failed",
                source="spotify_track",
                reason=str(exc)[:500],
                metadata={**metadata, "spotify_id": spotify_id},
            )
        if _has_spotify_album_identity(resolved):
            return AlbumBackfillAction(
                song_id=song.id,
                title=song.title,
                artists=song.artists,
                action_type="upsert_album_link",
                source="spotify_track",
                reason="Spotify track ID resolved to album identity",
                metadata=_merge_metadata(resolved, metadata),
            )

    review_candidates = await _review_candidates_for_song(song) if allow_review_search else []
    trusted_search_match = _trusted_search_match(song, review_candidates) if allow_search_auto_apply else None
    if trusted_search_match:
        return AlbumBackfillAction(
            song_id=song.id,
            title=song.title,
            artists=song.artists,
            action_type="upsert_album_link",
            source="spotify_search_auto",
            reason="title/artist Spotify search produced one exact album-backed match",
            metadata=trusted_search_match,
            review_candidates=review_candidates,
        )
    return AlbumBackfillAction(
        song_id=song.id,
        title=song.title,
        artists=song.artists,
        action_type="review_required" if review_candidates else "no_trusted_album_source",
        source="spotify_search" if review_candidates else "none",
        reason=(
            "no Spotify track ID with album identity; title/artist search is review-only"
            if allow_review_search
            else "no Spotify track ID with album identity; review search skipped"
        ),
        metadata=metadata,
        review_candidates=review_candidates,
    )


async def _review_candidates_for_song(song: SongAlbumSnapshot) -> list[dict[str, Any]]:
    if not song.title or not song.artists:
        return []
    query = f'track:"{song.title}" artist:"{song.artists[0]}"'
    try:
        tracks = await _search_spotify_tracks(query, limit=5)
    except Exception:
        return []
    candidates = []
    for track in tracks:
        candidates.append(
            {
                "spotify_id": track.get("spotify_id"),
                "title": track.get("title"),
                "artist": track.get("artist"),
                "artists": track.get("artists") or [],
                "artist_ids": track.get("artist_ids") or [],
                "album_id": track.get("album_id"),
                "album": track.get("album"),
                "album_artists": track.get("album_artists") or [],
                "album_type": track.get("album_type"),
                "release_date": track.get("release_date"),
                "total_tracks": track.get("total_tracks"),
                "album_art_url": track.get("album_art_url"),
                "album_art_highres": track.get("album_art_highres"),
                "album_art_medres": track.get("album_art_medres"),
                "album_art_lowres": track.get("album_art_lowres"),
            }
        )
    return candidates


async def run_backfill(args: argparse.Namespace) -> AlbumBackfillStats:
    load_dotenv(args.env_file)
    _ensure_spotify_env_aliases()
    database_url = args.database_url or os.getenv("PROD_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL or PROD_DATABASE_URL is required")
    if args.apply and not args.confirm_production:
        raise RuntimeError("--apply requires --confirm-production")

    import asyncpg

    conn = await asyncpg.connect(database_url)
    stats = AlbumBackfillStats()
    actions: list[AlbumBackfillAction] = []
    cache = SpotifyTrackCache(args.cache_path)
    try:
        songs = await fetch_songs(
            conn,
            song_id=args.song_id or None,
            limit=args.limit,
            offset=args.offset,
            trusted_only=args.trusted_only,
            missing_only=args.missing_only,
        )
        stats.songs_scanned = len(songs)
        await cache.preload(trusted_spotify_ids(songs))
        for song in songs:
            action = await plan_album_action(
                song,
                cache,
                allow_review_search=not args.skip_review_search,
                allow_search_auto_apply=args.search_auto_apply,
            )
            actions.append(action)
        stats.actions_planned = len(actions)
        stats.action_counts.update(action.action_type for action in actions)
        stats.source_counts.update(action.source for action in actions)
        stats.unique_albums_planned = unique_album_count(actions)
        _write_jsonl(args.report_jsonl, actions)
        _write_review(args.review_output, actions)

        if args.apply:
            for action in actions:
                if not action.auto_apply:
                    continue
                try:
                    async with conn.transaction():
                        await upsert_song_album_metadata(conn, song_id=action.song_id, metadata=action.metadata)
                    stats.actions_applied += 1
                except Exception as exc:
                    stats.actions_failed += 1
                    stats.errors.append(f"{action.song_id}:{str(exc)[:500]}")
    finally:
        await conn.close()
    return stats


async def fetch_songs(
    conn: Any,
    *,
    song_id: str | None,
    limit: int,
    offset: int,
    trusted_only: bool = False,
    missing_only: bool = False,
) -> list[SongAlbumSnapshot]:
    params: list[Any] = []
    clauses = []
    if song_id:
        params.append(song_id)
        clauses.append(f"s.id = ${len(params)}")
    if trusted_only:
        clauses.append(
            """
            (
                COALESCE(s.analysis_json::text, '') LIKE '%spotify_metadata%'
                OR COALESCE(s.analysis_json::text, '') LIKE '%"spotify"%'
                OR COALESCE(s.analysis_json::text, '') LIKE '%spotify:track:%'
                OR COALESCE(s.analysis_json::text, '') LIKE '%open.spotify.com/track/%'
            )
            """
        )
    if missing_only:
        clauses.append("NOT EXISTS (SELECT 1 FROM song_albums sal_missing WHERE sal_missing.song_id = s.id)")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_clause = ""
    if limit:
        params.append(limit)
        limit_clause = f"LIMIT ${len(params)}"
    offset_clause = ""
    if offset:
        params.append(offset)
        offset_clause = f"OFFSET ${len(params)}"
    rows = await conn.fetch(
        f"""
        SELECT
            s.id,
            s.title,
            s.analysis_json,
            EXISTS (SELECT 1 FROM song_albums sal WHERE sal.song_id = s.id) AS has_album_link,
            COALESCE(
                array_agg(DISTINCT a.name) FILTER (WHERE a.name IS NOT NULL),
                ARRAY[]::text[]
            ) AS artists
        FROM songs s
        LEFT JOIN song_artists sa ON s.id = sa.song_id
        LEFT JOIN artists a ON sa.artist_id = a.id
        {where}
        GROUP BY s.id
        ORDER BY s.created_at ASC NULLS LAST, s.id ASC
        {limit_clause}
        {offset_clause}
        """,
        *params,
    )
    return [SongAlbumSnapshot.from_row(row) for row in rows]


def unique_album_count(actions: list[AlbumBackfillAction]) -> int:
    return len({action.album_key for action in actions if action.album_key})


def trusted_spotify_ids(songs: list[SongAlbumSnapshot]) -> list[str]:
    ids = []
    for song in songs:
        metadata = _metadata_from_analysis(song.analysis_json)
        if _has_spotify_album_identity(metadata):
            continue
        spotify_id = _spotify_id_from_song(song, metadata)
        if spotify_id:
            ids.append(spotify_id)
    return ids


def _trusted_search_match(song: SongAlbumSnapshot, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    matching = []
    for candidate in candidates:
        if not _has_spotify_album_identity(candidate):
            continue
        if _title_match(song.title, str(candidate.get("title") or "")) and _artist_match(
            song.artists,
            candidate.get("artists") if isinstance(candidate.get("artists"), list) else [candidate.get("artist")],
        ):
            matching.append(candidate)
    album_keys = {
        _clean_text(candidate.get("album_id") or candidate.get("source_album_id"))
        for candidate in matching
        if _clean_text(candidate.get("album_id") or candidate.get("source_album_id"))
    }
    if len(matching) == 1 or len(album_keys) == 1:
        return matching[0]
    return None


def _title_match(left: str, right: str) -> bool:
    left_key = _normalized_title_key(left)
    right_key = _normalized_title_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _artist_match(song_artists: list[str], candidate_artists: list[Any]) -> bool:
    song_keys = {_normalized_text_key(artist) for artist in song_artists if _normalized_text_key(artist)}
    candidate_keys = {_normalized_text_key(artist) for artist in candidate_artists if _normalized_text_key(artist)}
    return bool(song_keys and candidate_keys and song_keys.intersection(candidate_keys))


def _normalized_title_key(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\((?:feat|featuring|ft|with|prod)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(?:feat|featuring|ft|with|prod)[^\]]*\]", " ", text, flags=re.IGNORECASE)
    return _normalized_text_key(text)


def _normalized_text_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _metadata_from_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    gpu = payload.get("gpu_ingestion") if isinstance(payload.get("gpu_ingestion"), dict) else {}
    metadata = gpu.get("spotify_metadata") if isinstance(gpu.get("spotify_metadata"), dict) else {}
    if metadata:
        return dict(metadata)
    spotify = payload.get("spotify") if isinstance(payload.get("spotify"), dict) else {}
    return dict(spotify) if spotify else {}


def _spotify_id_from_song(song: SongAlbumSnapshot, metadata: dict[str, Any]) -> str:
    gpu = song.analysis_json.get("gpu_ingestion") if isinstance(song.analysis_json.get("gpu_ingestion"), dict) else {}
    spotify = song.analysis_json.get("spotify") if isinstance(song.analysis_json.get("spotify"), dict) else {}
    for value in (
        metadata.get("spotify_id"),
        metadata.get("id"),
        gpu.get("spotify_id") if isinstance(gpu, dict) else None,
        spotify.get("id") if isinstance(spotify, dict) else None,
        _spotify_track_id_from_text(gpu.get("source") if isinstance(gpu, dict) else None),
    ):
        text = _clean_text(value)
        if text:
            return text
    return ""


def _spotify_track_id_from_text(value: Any) -> str:
    match = SPOTIFY_TRACK_RE.search(str(value or ""))
    return match.group(1) if match else ""


def _has_spotify_album_identity(metadata: dict[str, Any]) -> bool:
    return bool(_clean_text(metadata.get("album_id") or metadata.get("source_album_id")) and _clean_text(metadata.get("album")))


def _merge_metadata(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key, value in fallback.items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    return merged


def _parse_json_object(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict):
        return value, False
    if not value:
        return {}, False
    try:
        parsed = json.loads(value)
    except Exception:
        return {}, True
    return (parsed, False) if isinstance(parsed, dict) else ({}, True)


def _write_jsonl(path: Path | None, actions: list[AlbumBackfillAction]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(action.to_dict(), sort_keys=True) for action in actions) + "\n", encoding="utf-8")


def _write_review(path: Path | None, actions: list[AlbumBackfillAction]) -> None:
    if not path:
        return
    review = [action.to_dict() for action in actions if action.action_type == "review_required"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_spotify_env_aliases() -> None:
    if not os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SCOUT_SPOTIFY_CLIENT_ID"):
        os.environ["SPOTIFY_CLIENT_ID"] = os.environ["SCOUT_SPOTIFY_CLIENT_ID"]
    if not os.getenv("SPOTIFY_CLIENT_SECRET") and os.getenv("SCOUT_SPOTIFY_CLIENT_SECRET"):
        os.environ["SPOTIFY_CLIENT_SECRET"] = os.environ["SCOUT_SPOTIFY_CLIENT_SECRET"]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[key]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill normalized album links for library songs.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--song-id", default="")
    parser.add_argument("--cache-path", type=Path, default=Path("docs/reports/spotify_album_backfill_cache.json"))
    parser.add_argument("--report-jsonl", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Compatibility flag; dry-run is default unless --apply is passed.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument(
        "--trusted-only",
        action="store_true",
        help="Only scan songs whose analysis_json contains Spotify metadata or a Spotify track URL.",
    )
    parser.add_argument(
        "--skip-review-search",
        action="store_true",
        help="Do not run title/artist Spotify search for songs missing a trusted track ID.",
    )
    parser.add_argument(
        "--search-auto-apply",
        action="store_true",
        help="Auto-apply title/artist Spotify search matches only when exactly one album-backed candidate matches.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only scan songs that do not already have a song_albums row.",
    )
    parser.add_argument("--env-file", type=Path, default=Path("../shibuya-api/.env"))
    parser.add_argument("--database-url", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = asyncio.run(run_backfill(args))
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    return 1 if stats.errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
