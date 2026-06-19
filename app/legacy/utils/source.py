import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SPOTIFY_TRACK_RE = re.compile(r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]+)")
YOUTUBE_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})")

_spotify_access_token: str | None = None
_spotify_token_expires_at = 0.0
_spotify_request_lock = asyncio.Lock()


async def resolve_spotify_source(source: str, *, max_youtube_results: int = 5) -> dict[str, Any]:
    """Resolve a Spotify song name/link into a YouTube URL and metadata."""
    resolved = await resolve_source_metadata(source)
    return await resolve_youtube_match(resolved, max_youtube_results=max_youtube_results)


async def resolve_source_metadata(source: str) -> dict[str, Any]:
    """Resolve source metadata without doing YouTube search/download for Spotify inputs."""
    youtube_id = extract_youtube_video_id(source)
    if youtube_id:
        youtube_match = await get_youtube_metadata(source)
        return {
            "source": source,
            "spotify_metadata": {
                "spotify_id": None,
                "title": youtube_match.get("title") or source,
                "artist": youtube_match.get("channel") or "",
                "artists": [],
                "album": "",
                "duration_ms": int(float(youtube_match.get("duration_seconds") or 0) * 1000),
                "album_art_url": youtube_match.get("thumbnail"),
                "album_art_highres": youtube_match.get("thumbnail"),
                "album_art_medres": youtube_match.get("thumbnail"),
                "album_art_lowres": youtube_match.get("thumbnail"),
                "isrc": None,
                "popularity": 0,
                "query_only": True,
                "source_type": "youtube",
            },
            "youtube_match": youtube_match,
            "youtube_url": youtube_match["url"],
        }
    spotify_track = await _spotify_track_from_source(source)
    return {
        "source": source,
        "spotify_metadata": spotify_track,
    }


async def resolve_youtube_match(resolved: dict[str, Any], *, max_youtube_results: int = 5) -> dict[str, Any]:
    if resolved.get("youtube_url") and resolved.get("youtube_match"):
        return resolved
    spotify_track = resolved["spotify_metadata"]
    youtube_match = await _find_youtube_match(spotify_track, max_results=max_youtube_results)
    if youtube_match is None:
        raise RuntimeError(f"could not find YouTube match for Spotify source: {resolved.get('source')}")
    return {
        **resolved,
        "spotify_metadata": spotify_track,
        "youtube_match": youtube_match,
        "youtube_url": youtube_match["url"],
    }


async def resolve_and_download_spotify_source(source: str, output_dir: str | Path) -> dict[str, Any]:
    resolved = await resolve_spotify_source(source)
    audio_path = await download_youtube_audio(resolved["youtube_url"], output_dir)
    return {**resolved, "audio_path": audio_path}


async def download_youtube_audio(youtube_url: str, output_dir: str | Path) -> str:
    """Download a YouTube URL to a local audio file via yt-dlp."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    template = str(output / "source.%(ext)s")
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        *_yt_dlp_js_args(),
        "-x",
        "--audio-format",
        "wav",
        "-o",
        template,
        youtube_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])
    matches = sorted(output.glob("source.*"))
    if not matches:
        raise RuntimeError("yt-dlp did not produce an audio file")
    return str(matches[0])


def extract_youtube_video_id(source: str) -> str | None:
    match = YOUTUBE_URL_RE.search(source)
    return match.group(1) if match else None


async def get_youtube_metadata(youtube_url: str) -> dict[str, Any]:
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed")
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        *_yt_dlp_js_args(),
        "--dump-json",
        "--no-download",
        "--no-warnings",
        youtube_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])
    data = json.loads(stdout.decode("utf-8", errors="replace").splitlines()[-1])
    video_id = data.get("id") or extract_youtube_video_id(youtube_url)
    return {
        "video_id": video_id,
        "title": data.get("title"),
        "channel": data.get("channel") or data.get("uploader"),
        "duration_seconds": data.get("duration"),
        "url": data.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": data.get("thumbnail"),
    }


async def download_http_file(url: str, output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
    return str(target)


async def _spotify_track_from_source(source: str) -> dict[str, Any]:
    track_id = _extract_spotify_track_id(source)
    if track_id:
        if os.getenv("SPOTIFY_TRACK_METADATA_SOURCE", "embed").strip().lower() == "embed":
            return await _get_spotify_embed_track(track_id)
        track = await _get_spotify_track(track_id)
        return _format_spotify_track(track)
    tracks = await _search_spotify_tracks(source, limit=1)
    if not tracks:
        return {
            "spotify_id": None,
            "title": source,
            "artist": "",
            "artists": [],
            "album": "",
            "duration_ms": 0,
            "album_art_url": None,
            "album_art_highres": None,
            "album_art_medres": None,
            "album_art_lowres": None,
            "isrc": None,
            "popularity": 0,
            "query_only": True,
        }
    return tracks[0]


def _extract_spotify_track_id(source: str) -> str | None:
    match = SPOTIFY_TRACK_RE.search(source)
    return match.group(1) if match else None


async def _spotify_token() -> str:
    global _spotify_access_token, _spotify_token_expires_at
    if _spotify_access_token and time.time() < _spotify_token_expires_at - 60:
        return _spotify_access_token

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set")

    async with httpx.AsyncClient(timeout=30.0) as client:
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


async def _get_spotify_track(track_id: str) -> dict[str, Any]:
    return await _spotify_get_json(f"https://api.spotify.com/v1/tracks/{track_id}")


async def _get_spotify_embed_track(track_id: str) -> dict[str, Any]:
    oembed_url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/track/{track_id}"
    embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=oembed"
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
        "spotify_id": track_id,
        "title": title or track_id,
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


async def _search_spotify_tracks(query: str, *, limit: int) -> list[dict[str, Any]]:
    data = await _spotify_get_json(
        "https://api.spotify.com/v1/search",
        params={"q": query, "type": "track", "limit": limit},
    )
    return [_format_spotify_track(track) for track in data.get("tracks", {}).get("items", [])]


async def _spotify_get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    for attempt in range(5):
        async with _spotify_request_lock:
            token = await _spotify_token()
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()

            retry_after = _retry_after_seconds(response)
            logger.warning("spotify rate limited; sleeping %.1fs before retry", retry_after)
        await asyncio.sleep(retry_after)

    response.raise_for_status()
    return response.json()


def _retry_after_seconds(response: httpx.Response) -> float:
    value = response.headers.get("retry-after") or response.headers.get("Retry-After")
    try:
        if value is not None:
            return min(max(float(value), 1.0), 120.0)
    except ValueError:
        pass
    return 5.0


def _format_spotify_track(track: dict[str, Any]) -> dict[str, Any]:
    images = sorted(
        track.get("album", {}).get("images", []),
        key=lambda image: image.get("height", 0),
        reverse=True,
    )
    artists = [artist.get("name", "") for artist in track.get("artists", []) if artist.get("name")]
    album_art_highres = images[0]["url"] if len(images) > 0 else None
    album_art_medres = images[1]["url"] if len(images) > 1 else album_art_highres
    album_art_lowres = images[2]["url"] if len(images) > 2 else album_art_medres
    return {
        "spotify_id": track.get("id"),
        "title": track.get("name", ""),
        "artist": artists[0] if artists else "",
        "artists": artists,
        "album": track.get("album", {}).get("name", ""),
        "duration_ms": track.get("duration_ms", 0),
        "album_art_url": album_art_highres,
        "album_art_highres": album_art_highres,
        "album_art_medres": album_art_medres,
        "album_art_lowres": album_art_lowres,
        "isrc": track.get("external_ids", {}).get("isrc"),
        "popularity": track.get("popularity", 0),
    }


async def _find_youtube_match(track: dict[str, Any], *, max_results: int) -> dict[str, Any] | None:
    title = str(track.get("title") or "").strip()
    artist = str(track.get("artist") or "").strip()
    query = f'"{artist}" "{title}" official audio' if artist else f"{title} official audio"
    results = await _search_youtube(query, max_results=max_results)
    if not results:
        results = await _search_youtube(f"{artist} {title} official audio".strip(), max_results=max_results)
    expected_duration = float(track.get("duration_ms") or 0) / 1000
    for result in results:
        if _is_youtube_match(result, artist=artist, title=title, expected_duration=expected_duration):
            return _format_youtube_match(result)
    return _format_youtube_match(results[0]) if results else None


async def _search_youtube(query: str, *, max_results: int) -> list[dict[str, Any]]:
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed")
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        *_yt_dlp_js_args(),
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--no-download",
        "--flat-playlist",
        "--no-warnings",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode != 0:
        logger.warning("yt-dlp search failed: %s", stderr.decode("utf-8", errors="replace")[-1000:])
        return []
    results = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results


def _yt_dlp_js_args() -> list[str]:
    deno = shutil.which("deno")
    if deno:
        return ["--js-runtimes", f"deno:{deno}", "--remote-components", "ejs:github"]
    node = shutil.which("node")
    if node:
        return ["--js-runtimes", f"node:{node}", "--remote-components", "ejs:github"]
    return []


def _is_youtube_match(result: dict[str, Any], *, artist: str, title: str, expected_duration: float) -> bool:
    video_title = str(result.get("title") or "").lower()
    normalized_video_title = re.sub(r"[^\w\s]", "", video_title)
    artist_clean = re.sub(r"[^\w\s]", "", artist.lower())
    title_clean = re.sub(r"[^\w\s]", "", title.lower())
    has_artist = not artist or artist.lower() in video_title or artist_clean in normalized_video_title
    has_title = title.lower() in video_title or title_clean in normalized_video_title
    duration = result.get("duration")
    duration_ok = not expected_duration or not duration or abs(float(duration) - expected_duration) <= 20
    banned = any(
        keyword in video_title and "official" not in video_title
        for keyword in (
            "live",
            "remix",
            "cover",
            "karaoke",
            "instrumental",
            "slowed",
            "reverb",
            "sped up",
            "nightcore",
            "tutorial",
        )
    )
    return has_artist and has_title and duration_ok and not banned


def _format_youtube_match(result: dict[str, Any]) -> dict[str, Any]:
    video_id = result.get("id")
    return {
        "video_id": video_id,
        "title": result.get("title"),
        "channel": result.get("channel") or result.get("uploader"),
        "duration_seconds": result.get("duration"),
        "url": result.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": result.get("thumbnail"),
    }
