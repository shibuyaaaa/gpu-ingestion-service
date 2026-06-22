import hashlib
import re
import unicodedata
from typing import Any

from app.legacy.utils.source import extract_youtube_video_id


def ingestion_identity_key(
    *,
    source: str | None = None,
    youtube_url: str | None = None,
    youtube_match: dict[str, Any] | None = None,
    spotify_metadata: dict[str, Any] | None = None,
) -> str:
    video_id = extract_youtube_video_id(youtube_url or "")
    if not video_id and isinstance(youtube_match, dict):
        video_id = str(youtube_match.get("video_id") or "").strip() or None
    if video_id:
        return f"youtube:{video_id}"

    metadata = spotify_metadata or {}
    spotify_id = str(metadata.get("spotify_id") or metadata.get("id") or "").strip()
    if spotify_id:
        return f"spotify:{spotify_id}"

    isrc = str(metadata.get("isrc") or "").strip().upper()
    if isrc:
        return f"isrc:{isrc}"

    title = normalize_identity_text(str(metadata.get("title") or ""))
    artist = normalize_identity_text(_metadata_artist(metadata))
    if title and artist:
        return f"title-artist:{_short_digest(f'{title}|{artist}')}"

    normalized_source = normalize_identity_text(source or "")
    if normalized_source:
        return f"source:{_short_digest(normalized_source)}"
    return "unknown"


def identity_key_from_artifacts(artifacts: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    existing = str(
        artifacts.get("ingestion_identity_key")
        or payload.get("ingestion_identity_key")
        or ""
    ).strip()
    if existing:
        return existing
    return ingestion_identity_key(
        source=str(artifacts.get("source") or payload.get("source") or ""),
        youtube_url=str(artifacts.get("youtube_url") or payload.get("youtube_url") or ""),
        youtube_match=artifacts.get("youtube_match") if isinstance(artifacts.get("youtube_match"), dict) else None,
        spotify_metadata=(
            artifacts.get("spotify_metadata")
            if isinstance(artifacts.get("spotify_metadata"), dict)
            else payload.get("spotify_metadata")
            if isinstance(payload.get("spotify_metadata"), dict)
            else None
        ),
    )


def safe_identity_part(identity_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", identity_key).strip("-") or "unknown"


def normalize_identity_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value).lower()
    normalized = re.sub(r"[\u0300-\u036f]", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _metadata_artist(metadata: dict[str, Any]) -> str:
    value = metadata.get("artist")
    if str(value or "").strip():
        return str(value).strip()
    artists = metadata.get("artists")
    if isinstance(artists, list):
        for artist in artists:
            if isinstance(artist, dict):
                name = artist.get("name")
            else:
                name = artist
            if str(name or "").strip():
                return str(name).strip()
    return ""


def _short_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]
