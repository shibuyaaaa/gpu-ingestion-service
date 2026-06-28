import os
import re
import time
from typing import Any

import httpx

from app.crawler.types import ChartCandidate

SPOTIFY_PLAYLIST_RE = re.compile(r"(?:spotify:playlist:|open\.spotify\.com/playlist/)([A-Za-z0-9]+)")

_access_token: str | None = None
_token_expires_at = 0.0


class SpotifyChartPlaylistClient:
    def __init__(self, *, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds

    async def fetch_candidates(self, playlist_urls: list[str], *, max_pages: int) -> list[ChartCandidate]:
        candidates: list[ChartCandidate] = []
        for playlist_url in playlist_urls:
            playlist_id = extract_playlist_id(playlist_url)
            if not playlist_id:
                continue
            candidates.extend(
                await self._fetch_playlist_candidates(
                    playlist_id,
                    playlist_source=playlist_url,
                    max_pages=max_pages,
                )
            )
        return _dedupe_and_sort(candidates)

    async def _fetch_playlist_candidates(
        self,
        playlist_id: str,
        *,
        playlist_source: str,
        max_pages: int,
    ) -> list[ChartCandidate]:
        token = await _spotify_token(timeout_seconds=self.timeout_seconds)
        candidates: list[ChartCandidate] = []
        offset = 0
        rank = 0
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for _ in range(max(1, max_pages)):
                response = await client.get(
                    f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
                    params={
                        "limit": 100,
                        "offset": offset,
                        "fields": (
                            "items(track(id,name,popularity,artists(id,name),"
                            "album(id,name,images,release_date,album_type,total_tracks,artists(id,name)),"
                            "duration_ms,external_ids)),next"
                        ),
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                data = response.json()
                for item in data.get("items", []):
                    track = item.get("track") or {}
                    candidate = _candidate_from_track(track, playlist_source=playlist_source, rank=rank)
                    rank += 1
                    if candidate:
                        candidates.append(candidate)
                if not data.get("next"):
                    break
                offset += 100
        return candidates


def extract_playlist_id(value: str) -> str | None:
    match = SPOTIFY_PLAYLIST_RE.search(value)
    return match.group(1) if match else None


async def _spotify_token(*, timeout_seconds: float) -> str:
    global _access_token, _token_expires_at
    if _access_token and time.time() < _token_expires_at - 60:
        return _access_token

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
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
    _access_token = str(token)
    _token_expires_at = time.time() + float(data.get("expires_in", 3600))
    return _access_token


def _candidate_from_track(track: dict[str, Any], *, playlist_source: str, rank: int) -> ChartCandidate | None:
    spotify_id = str(track.get("id") or "").strip()
    title = str(track.get("name") or "").strip()
    artist_items = [artist for artist in track.get("artists", []) if isinstance(artist, dict)]
    artists = [str(artist.get("name")).strip() for artist in artist_items if artist.get("name")]
    artist_ids = [str(artist.get("id")).strip() for artist in artist_items if artist.get("id")]
    if not spotify_id or not title or not artists:
        return None
    album = track.get("album") if isinstance(track.get("album"), dict) else {}
    images = sorted(
        album.get("images") or [],
        key=lambda image: image.get("height", 0) if isinstance(image, dict) else 0,
        reverse=True,
    )
    album_art_highres = images[0]["url"] if len(images) > 0 and images[0].get("url") else None
    album_art_medres = images[1]["url"] if len(images) > 1 and images[1].get("url") else album_art_highres
    album_art_lowres = images[2]["url"] if len(images) > 2 and images[2].get("url") else album_art_medres
    album_artist_items = [artist for artist in album.get("artists", []) if isinstance(artist, dict)]
    album_artists = [
        {"id": str(artist.get("id") or "").strip(), "name": str(artist.get("name") or "").strip()}
        for artist in album_artist_items
        if artist.get("name")
    ]
    return ChartCandidate(
        spotify_id=spotify_id,
        title=title,
        artist=artists[0],
        artists=artists,
        popularity=int(track.get("popularity") or 0),
        playlist_source=playlist_source,
        rank=rank,
        artist_ids=artist_ids,
        album_id=str(album.get("id") or "").strip() or None,
        album=str(album.get("name") or "").strip() or None,
        album_art_url=album_art_highres,
        album_art_highres=album_art_highres,
        album_art_medres=album_art_medres,
        album_art_lowres=album_art_lowres,
        album_artists=album_artists,
        album_type=str(album.get("album_type") or "").strip() or None,
        release_date=str(album.get("release_date") or "").strip() or None,
        total_tracks=int(album["total_tracks"]) if album.get("total_tracks") is not None else None,
        duration_ms=int(track["duration_ms"]) if track.get("duration_ms") is not None else None,
        isrc=(track.get("external_ids") or {}).get("isrc"),
    )


def _dedupe_and_sort(candidates: list[ChartCandidate]) -> list[ChartCandidate]:
    best: dict[str, ChartCandidate] = {}
    for candidate in candidates:
        previous = best.get(candidate.spotify_id)
        if previous is None or (candidate.popularity, -candidate.rank) > (previous.popularity, -previous.rank):
            best[candidate.spotify_id] = candidate
    return sorted(best.values(), key=lambda candidate: (-candidate.popularity, candidate.rank, candidate.spotify_id))
