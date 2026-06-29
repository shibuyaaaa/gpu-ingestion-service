from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.crawler.types import ChartCandidate


SPOTIFY_TRACK_URL_RE = re.compile(r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]+)")
SPOTIFY_ALBUM_URL_RE = re.compile(r"(?:spotify:album:|open\.spotify\.com/album/)([A-Za-z0-9]+)")

_spotify_access_token: str | None = None
_spotify_token_expires_at = 0.0
_spotify_api_cooldown_until = 0.0
_spotify_request_lock = asyncio.Lock()


class SpotifyRateLimited(RuntimeError):
    pass


class SpotifyPublicPageUnavailable(RuntimeError):
    pass


class AlbumMetadataResolver:
    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        resolver: str = "public-first",
        timeout_seconds: float = 30.0,
    ):
        self.cache_path = cache_path or Path(os.getenv("ALBUM_METADATA_CACHE_PATH", "/var/lib/gpu-ingestion/album_metadata_cache.json"))
        self.resolver = resolver
        self.timeout_seconds = timeout_seconds
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            try:
                parsed = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    self.cache = {str(key): value for key, value in parsed.items() if isinstance(value, dict)}
            except Exception:
                self.cache = {}

    async def resolve_track(self, spotify_id: str, *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
        spotify_id = _clean_text(spotify_id)
        existing = dict(existing or {})
        if not spotify_id:
            return existing
        if has_spotify_album_identity(existing):
            return existing
        cached = self.cache.get(spotify_id)
        if has_spotify_album_identity(cached or {}):
            return merge_metadata(cached or {}, existing)
        if self.resolver == "api-only":
            return merge_metadata(await self._fetch_api_track(spotify_id), existing)
        if self.resolver == "api-first":
            try:
                metadata = await self._fetch_api_track(spotify_id)
            except Exception:
                metadata = await self._fetch_public_track_metadata(spotify_id)
        else:
            metadata = await self._fetch_public_track_metadata(spotify_id)
        metadata = merge_metadata(metadata, existing)
        self.cache[spotify_id] = metadata
        self.flush()
        return metadata

    async def resolve_many(
        self,
        spotify_ids: list[str],
        *,
        existing_by_id: dict[str, dict[str, Any]] | None = None,
        prefer_api: bool = True,
    ) -> dict[str, dict[str, Any]]:
        existing_by_id = existing_by_id or {}
        normalized_ids = _dedupe([_clean_text(value) for value in spotify_ids if _clean_text(value)])
        results: dict[str, dict[str, Any]] = {}
        unresolved = []
        for spotify_id in normalized_ids:
            existing = dict(existing_by_id.get(spotify_id) or {})
            if has_spotify_album_identity(existing):
                results[spotify_id] = existing
            elif has_spotify_album_identity(self.cache.get(spotify_id) or {}):
                results[spotify_id] = merge_metadata(self.cache[spotify_id], existing)
            else:
                unresolved.append(spotify_id)

        api_results: dict[str, dict[str, Any]] = {}
        if self.resolver == "public-first":
            prefer_api = False
        if prefer_api and self.resolver != "api-first" and self.resolver != "api-only":
            prefer_api = False
        if prefer_api or self.resolver == "api-only":
            try:
                api_results = await self._fetch_api_tracks(unresolved)
            except Exception:
                api_results = {}
            for spotify_id, metadata in api_results.items():
                merged = merge_metadata(metadata, existing_by_id.get(spotify_id) or {})
                results[spotify_id] = merged
                self.cache[spotify_id] = merged

        public_needed = [spotify_id for spotify_id in unresolved if spotify_id not in results and self.resolver != "api-only"]
        if public_needed:
            for index in range(0, len(public_needed), self._public_concurrency()):
                batch = public_needed[index : index + self._public_concurrency()]
                public_results = await asyncio.gather(
                    *(self._fetch_public_track_metadata(spotify_id) for spotify_id in batch),
                    return_exceptions=True,
                )
                for spotify_id, result in zip(batch, public_results, strict=False):
                    if isinstance(result, Exception):
                        continue
                    merged = merge_metadata(result, existing_by_id.get(spotify_id) or {})
                    results[spotify_id] = merged
                    self.cache[spotify_id] = merged
        if api_results or public_needed:
            self.flush()
        return results

    async def enrich_candidates(self, candidates: list[ChartCandidate], *, prefer_api: bool = True) -> list[ChartCandidate]:
        ids = [candidate.spotify_id for candidate in candidates if not candidate.album_id]
        existing_by_id = {candidate.spotify_id: candidate.to_metadata() for candidate in candidates}
        resolved = await self.resolve_many(ids, existing_by_id=existing_by_id, prefer_api=prefer_api)
        enriched: list[ChartCandidate] = []
        for candidate in candidates:
            metadata = resolved.get(candidate.spotify_id) or {}
            if not has_spotify_album_identity(metadata):
                enriched.append(candidate)
                continue
            enriched.append(
                replace(
                    candidate,
                    artist_ids=_list_or_none(metadata.get("artist_ids")) or candidate.artist_ids,
                    album_id=_clean_text(metadata.get("album_id") or metadata.get("source_album_id")) or candidate.album_id,
                    album=_clean_text(metadata.get("album")) or candidate.album,
                    album_art_url=_clean_text(metadata.get("album_art_url")) or candidate.album_art_url,
                    album_art_highres=_clean_text(metadata.get("album_art_highres")) or candidate.album_art_highres,
                    album_art_medres=_clean_text(metadata.get("album_art_medres")) or candidate.album_art_medres,
                    album_art_lowres=_clean_text(metadata.get("album_art_lowres")) or candidate.album_art_lowres,
                    album_artists=_artist_entries_or_none(metadata.get("album_artists")) or candidate.album_artists,
                    album_type=_clean_text(metadata.get("album_type")) or candidate.album_type,
                    release_date=_clean_text(metadata.get("release_date")) or candidate.release_date,
                    total_tracks=_int_or_none(metadata.get("total_tracks")) or candidate.total_tracks,
                    duration_ms=_int_or_none(metadata.get("duration_ms")) or candidate.duration_ms,
                    isrc=_clean_text(metadata.get("isrc")) or candidate.isrc,
                )
            )
        return enriched

    async def _fetch_api_track(self, spotify_id: str) -> dict[str, Any]:
        tracks = await self._fetch_api_tracks([spotify_id])
        return tracks.get(spotify_id) or {"spotify_id": spotify_id}

    async def _fetch_api_tracks(self, spotify_ids: list[str]) -> dict[str, dict[str, Any]]:
        spotify_ids = _dedupe([_clean_text(value) for value in spotify_ids if _clean_text(value)])
        if not spotify_ids:
            return {}
        results: dict[str, dict[str, Any]] = {}
        for index in range(0, len(spotify_ids), 50):
            batch = spotify_ids[index : index + 50]
            data = await _spotify_api_get_json(
                "https://api.spotify.com/v1/tracks",
                params={"ids": ",".join(batch)},
                timeout_seconds=self.timeout_seconds,
            )
            for track in data.get("tracks") or []:
                if isinstance(track, dict) and track.get("id"):
                    results[str(track["id"])] = format_spotify_track(track, metadata_source="spotify_api")
        return results

    async def _fetch_public_track_metadata(self, spotify_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True, headers=_public_headers()) as client:
            response = await client.get(f"https://open.spotify.com/track/{spotify_id}")
            if response.status_code >= 400:
                raise SpotifyPublicPageUnavailable(f"Spotify public track page returned {response.status_code}")
            track_metadata = parse_spotify_public_track_page(response.text, spotify_id=spotify_id)
            album_id = _clean_text(track_metadata.get("album_id"))
            if album_id:
                album_response = await client.get(f"https://open.spotify.com/album/{album_id}")
                if album_response.status_code < 400:
                    album_metadata = parse_spotify_public_album_page(album_response.text, album_id=album_id)
                    track_metadata = merge_metadata(album_metadata, track_metadata)
                    track_metadata["spotify_id"] = spotify_id
            return track_metadata

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _public_concurrency() -> int:
        try:
            return max(1, min(int(os.getenv("ALBUM_METADATA_PUBLIC_CONCURRENCY", "8")), 32))
        except ValueError:
            return 8


def parse_spotify_public_track_page(markup: str, *, spotify_id: str) -> dict[str, Any]:
    meta = _parse_html_metadata(markup)
    description = _meta_value(meta, "og:description", "twitter:description")
    description_parts = _description_parts(description)
    artist = description_parts[0] if len(description_parts) > 0 else _meta_value(meta, "music:musician_description")
    album_title = description_parts[1] if len(description_parts) > 1 else ""
    album_id = _spotify_album_id_from_text(_meta_value(meta, "music:album"))
    artist_id = _spotify_artist_id_from_text(_meta_value(meta, "music:musician"))
    cover = _meta_value(meta, "og:image", "twitter:image")
    metadata = {
        "spotify_id": spotify_id,
        "title": _meta_value(meta, "og:title", "twitter:title"),
        "artist": artist,
        "artists": [artist] if artist else [],
        "artist_ids": [artist_id] if artist_id else [],
        "album_id": album_id,
        "album": album_title,
        "album_artists": [{"id": artist_id, "name": artist}] if artist else [],
        "release_date": _meta_value(meta, "music:release_date"),
        "duration_ms": (_int_or_none(_meta_value(meta, "music:duration")) or 0) * 1000,
        "album_art_url": cover,
        "album_art_highres": cover,
        "album_art_medres": cover,
        "album_art_lowres": cover,
        "metadata_source": "spotify_public_page",
    }
    return _drop_empty(metadata)


def parse_spotify_public_album_page(markup: str, *, album_id: str) -> dict[str, Any]:
    meta = _parse_html_metadata(markup)
    description = _meta_value(meta, "og:description", "description")
    description_parts = _description_parts(description)
    title = _album_title_from_og_title(_meta_value(meta, "og:title"))
    artist = description_parts[0] if len(description_parts) > 0 else ""
    album_type = description_parts[1].lower() if len(description_parts) > 1 else ""
    total_tracks = _total_tracks_from_description(description)
    artist_id = _spotify_artist_id_from_text(_meta_value(meta, "music:musician"))
    cover = _meta_value(meta, "og:image", "twitter:image")
    return _drop_empty(
        {
            "album_id": album_id,
            "album": title,
            "album_type": album_type,
            "total_tracks": total_tracks,
            "release_date": _meta_value(meta, "music:release_date"),
            "album_artists": [{"id": artist_id, "name": artist}] if artist else [],
            "album_art_url": cover,
            "album_art_highres": cover,
            "album_art_medres": cover,
            "album_art_lowres": cover,
            "metadata_source": "spotify_public_page",
        }
    )


def format_spotify_track(track: dict[str, Any], *, metadata_source: str = "spotify_api") -> dict[str, Any]:
    album = track.get("album", {}) if isinstance(track.get("album"), dict) else {}
    images = sorted(
        [image for image in album.get("images", []) if isinstance(image, dict)],
        key=lambda image: image.get("height", 0) or 0,
        reverse=True,
    )
    artist_items = [artist for artist in track.get("artists", []) if isinstance(artist, dict)]
    artists = [str(artist.get("name") or "").strip() for artist in artist_items if artist.get("name")]
    artist_ids = [str(artist.get("id") or "").strip() for artist in artist_items if artist.get("id")]
    album_artist_items = [artist for artist in album.get("artists", []) if isinstance(artist, dict)]
    album_artists = [
        {"id": str(artist.get("id") or "").strip(), "name": str(artist.get("name") or "").strip()}
        for artist in album_artist_items
        if artist.get("name")
    ]
    album_art_highres = images[0]["url"] if len(images) > 0 and images[0].get("url") else None
    album_art_medres = images[1]["url"] if len(images) > 1 and images[1].get("url") else album_art_highres
    album_art_lowres = images[2]["url"] if len(images) > 2 and images[2].get("url") else album_art_medres
    return _drop_empty(
        {
            "spotify_id": track.get("id"),
            "title": track.get("name", ""),
            "artist": artists[0] if artists else "",
            "artists": artists,
            "artist_ids": artist_ids,
            "album_id": album.get("id"),
            "album": album.get("name", ""),
            "album_artists": album_artists,
            "album_type": album.get("album_type"),
            "release_date": album.get("release_date"),
            "total_tracks": album.get("total_tracks"),
            "duration_ms": track.get("duration_ms", 0),
            "album_art_url": album_art_highres,
            "album_art_highres": album_art_highres,
            "album_art_medres": album_art_medres,
            "album_art_lowres": album_art_lowres,
            "isrc": (track.get("external_ids") or {}).get("isrc") if isinstance(track.get("external_ids"), dict) else None,
            "popularity": track.get("popularity", 0),
            "metadata_source": metadata_source,
        }
    )


def merge_metadata(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary or {})
    for key, value in (fallback or {}).items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    return merged


def has_spotify_album_identity(metadata: dict[str, Any] | None) -> bool:
    metadata = metadata or {}
    return bool(_clean_text(metadata.get("album_id") or metadata.get("source_album_id")) and _clean_text(metadata.get("album")))


def extract_spotify_track_id(value: Any) -> str:
    match = SPOTIFY_TRACK_URL_RE.search(str(value or ""))
    return match.group(1) if match else ""


async def _spotify_api_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    global _spotify_api_cooldown_until
    if time.time() < _spotify_api_cooldown_until:
        raise SpotifyRateLimited("Spotify API cooldown is active")
    async with _spotify_request_lock:
        token = await _spotify_token(timeout_seconds=timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response)
            _spotify_api_cooldown_until = time.time() + retry_after
            raise SpotifyRateLimited(f"Spotify API rate limited; retry after {retry_after:.1f}s")
        response.raise_for_status()
        return response.json()


async def _spotify_token(*, timeout_seconds: float) -> str:
    global _spotify_access_token, _spotify_token_expires_at
    if _spotify_access_token and time.time() < _spotify_token_expires_at - 60:
        return _spotify_access_token
    client_id = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("SCOUT_SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("SCOUT_SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set")
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
        )
        response.raise_for_status()
        data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Spotify token response missing access_token")
    _spotify_access_token = str(token)
    _spotify_token_expires_at = time.time() + float(data.get("expires_in", 3600))
    return _spotify_access_token


class _MetadataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        key = values.get("property") or values.get("name")
        content = values.get("content")
        if key and content and key not in self.meta:
            self.meta[key] = html.unescape(content)


def _parse_html_metadata(markup: str) -> dict[str, str]:
    parser = _MetadataParser()
    parser.feed(markup)
    return parser.meta


def _meta_value(meta: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = _clean_text(meta.get(key))
        if value:
            return value
    return ""


def _description_parts(value: str) -> list[str]:
    return [_clean_text(part) for part in value.split("·") if _clean_text(part)]


def _album_title_from_og_title(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"\s+\|\s+Spotify$", "", text)
    return re.sub(r"\s+-\s+(Album|Single|EP)\s+by\s+.+$", "", text, flags=re.IGNORECASE)


def _total_tracks_from_description(value: str) -> int | None:
    match = re.search(r"(\d+)\s+songs?", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _spotify_album_id_from_text(value: str) -> str:
    match = SPOTIFY_ALBUM_URL_RE.search(value)
    return match.group(1) if match else ""


def _spotify_artist_id_from_text(value: str) -> str:
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "artist":
        return parts[-1]
    return ""


def _retry_after_seconds(response: httpx.Response) -> float:
    value = response.headers.get("retry-after") or response.headers.get("Retry-After")
    try:
        if value is not None:
            return min(max(float(value), 30.0), 900.0)
    except ValueError:
        pass
    return 300.0


def _public_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; ShibuyaAlbumResolver/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [])}


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _list_or_none(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [_clean_text(item) for item in value if _clean_text(item)]
    return items or None


def _artist_entries_or_none(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    entries = []
    for item in value:
        if isinstance(item, dict):
            name = _clean_text(item.get("name"))
            source_id = _clean_text(item.get("id") or item.get("source_artist_id"))
        else:
            name = _clean_text(item)
            source_id = ""
        if name:
            entries.append({"id": source_id, "name": name})
    return entries or None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
