import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SPOTIFY_TRACK_RE = re.compile(r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]+)")
YOUTUBE_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})")
TRANSIENT_YTDLP_ERROR_RE = re.compile(
    r"(HTTP Error (?:403|408|409|425|429|5\d\d)|timed out|timeout|temporar(?:y|ily)|unavailable|try again)",
    re.IGNORECASE,
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_spotify_access_token: str | None = None
_spotify_token_expires_at = 0.0
_spotify_request_lock = asyncio.Lock()
_youtube_token_cache: dict[str, Any] | None = None
_youtube_token_cache_time: datetime | None = None
_youtube_cookies_path: str | None = None
_youtube_cookies_cache_time: datetime | None = None


class DownloadError(RuntimeError):
    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


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
                "genre": youtube_match.get("genre") or "",
                "genres": youtube_match.get("categories") or [],
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
    attempts = max(1, _int_env("YTDLP_DOWNLOAD_ATTEMPTS", 3))
    retry_delay = max(0.0, _float_env("YTDLP_RETRY_DELAY_SECONDS", 2.0))
    timeout_seconds = max(10.0, _float_env("YTDLP_DOWNLOAD_TIMEOUT_SECONDS", 120.0))
    last_error = ""
    last_category = "unknown"
    for strategy in _yt_dlp_download_strategies():
        for attempt in range(1, attempts + 1):
            _remove_partial_ytdlp_outputs(output)
            error_text = await _run_ytdlp_download_strategy(
                youtube_url=youtube_url,
                template=template,
                strategy=strategy,
                timeout_seconds=timeout_seconds,
            )
            if error_text is None:
                matches = sorted(output.glob("source.*"))
                if matches:
                    logger.info("yt-dlp download succeeded with strategy %s", strategy["name"])
                    return str(matches[0])
                error_text = "yt-dlp did not produce an audio file"
            last_error = error_text
            last_category = _categorize_ytdlp_error(error_text)
            if _is_terminal_ytdlp_error(last_category):
                raise DownloadError(last_error, last_category)
            if not _is_transient_ytdlp_error(error_text) or attempt >= attempts:
                break
            error_lines = error_text.strip().splitlines()
            logger.warning(
                "yt-dlp %s failed transiently on attempt %d/%d: %s",
                strategy["name"],
                attempt,
                attempts,
                error_lines[-1] if error_lines else error_text,
            )
            if retry_delay:
                await asyncio.sleep(retry_delay)
        if last_error:
            logger.warning(
                "yt-dlp strategy %s failed (%s): %s",
                strategy["name"],
                last_category,
                _last_error_line(last_error),
            )
    raise DownloadError(last_error or "yt-dlp did not produce an audio file", last_category)


def _yt_dlp_download_strategies() -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    payload = _load_youtube_token_payload() or {}
    po_token = str(payload.get("po_token") or "").strip()
    visitor_data = str(payload.get("visitor_data") or "").strip()
    if po_token and visitor_data:
        strategies.append(
            {
                "name": "po_token_web_cookies",
                "use_cookies": True,
                "extractor_args": [f"youtube:player_client=web;po_token=web+{po_token};visitor_data={visitor_data}"],
            }
        )
        strategies.append(
            {
                "name": "po_token_web_no_cookies",
                "use_cookies": False,
                "extractor_args": [f"youtube:player_client=web;po_token=web+{po_token};visitor_data={visitor_data}"],
            }
        )
    strategies.extend(
        [
            {"name": "auto_cookies", "use_cookies": True, "extractor_args": []},
            {"name": "auto_no_cookies", "use_cookies": False, "extractor_args": []},
            {"name": "web_js_cookies", "use_cookies": True, "extractor_args": ["youtube:player_client=web"]},
            {"name": "web_js_no_cookies", "use_cookies": False, "extractor_args": ["youtube:player_client=web"]},
            {"name": "android_vr_cookies", "use_cookies": True, "extractor_args": ["youtube:player_client=android_vr"]},
            {"name": "android_vr_no_cookies", "use_cookies": False, "extractor_args": ["youtube:player_client=android_vr"]},
            {"name": "ios_cookies", "use_cookies": True, "extractor_args": ["youtube:player_client=ios"]},
            {"name": "ios_no_cookies", "use_cookies": False, "extractor_args": ["youtube:player_client=ios"]},
        ]
    )
    return strategies


async def _run_ytdlp_download_strategy(
    *,
    youtube_url: str,
    template: str,
    strategy: dict[str, Any],
    timeout_seconds: float,
) -> str | None:
    cmd = [
        "yt-dlp",
        *_yt_dlp_common_args(
            use_cookies=bool(strategy.get("use_cookies", True)),
            extractor_args=list(strategy.get("extractor_args") or []),
        ),
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "192K",
        "-o",
        template,
        "--socket-timeout",
        str(max(5, int(timeout_seconds / 4))),
        "--retries",
        "1",
        "--user-agent",
        USER_AGENT,
        "--no-warnings",
        youtube_url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        await _terminate_process(proc)
        return f"yt-dlp strategy {strategy['name']} timed out after {timeout_seconds:.0f}s"
    output_text = (stderr + b"\n" + stdout).decode("utf-8", errors="replace")[-2000:]
    if proc.returncode == 0:
        return None
    return output_text or f"yt-dlp exited with status {proc.returncode}"


def extract_youtube_video_id(source: str) -> str | None:
    match = YOUTUBE_URL_RE.search(source)
    return match.group(1) if match else None


def _is_transient_ytdlp_error(error_text: str) -> bool:
    return bool(TRANSIENT_YTDLP_ERROR_RE.search(error_text or ""))


def _categorize_ytdlp_error(error_text: str) -> str:
    lowered = (error_text or "").lower()
    if "sign in" in lowered or "login" in lowered or "not a bot" in lowered or "cookies-from-browser" in lowered:
        return "auth_required"
    if "429" in lowered or "rate" in lowered or "too many" in lowered:
        return "rate_limit"
    if "private" in lowered:
        return "private_video"
    if "age" in lowered or "confirm your age" in lowered:
        return "age_restricted"
    if "copyright" in lowered or "blocked" in lowered:
        return "copyright_blocked"
    if "unavailable" in lowered or "removed" in lowered or "deleted" in lowered:
        return "unavailable"
    if "not exist" in lowered or "invalid" in lowered:
        return "invalid_url"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    return "unknown"


def _is_terminal_ytdlp_error(category: str) -> bool:
    return category in {"private_video", "copyright_blocked", "invalid_url", "unavailable", "age_restricted"}


def _last_error_line(error_text: str) -> str:
    lines = [line.strip() for line in (error_text or "").splitlines() if line.strip()]
    return lines[-1] if lines else error_text


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.kill()
    try:
        await asyncio.wait_for(proc.communicate(), timeout=5)
    except asyncio.TimeoutError:
        logger.warning("timed out waiting for yt-dlp process to exit after kill")


def _remove_partial_ytdlp_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("source.*"):
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            logger.warning("failed to remove partial yt-dlp output: %s", path)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


async def get_youtube_metadata(youtube_url: str) -> dict[str, Any]:
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed")
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        *_yt_dlp_common_args(),
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
        "genre": data.get("genre") or _first_text(data.get("categories")),
        "categories": data.get("categories") or [],
        "tags": data.get("tags") or [],
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
        return await _with_spotify_artist_genres(_format_spotify_track(track))
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
    return await _with_spotify_artist_genres(tracks[0])


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
    from app.album_metadata import AlbumMetadataResolver

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
    metadata = {
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
    try:
        return await AlbumMetadataResolver(resolver="public-first").resolve_track(track_id, existing=metadata)
    except Exception:
        return metadata


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
    album = track.get("album", {}) if isinstance(track.get("album"), dict) else {}
    images = sorted(
        album.get("images", []),
        key=lambda image: image.get("height", 0),
        reverse=True,
    )
    artist_items = [artist for artist in track.get("artists", []) if isinstance(artist, dict)]
    artists = [artist.get("name", "") for artist in artist_items if artist.get("name")]
    artist_ids = [artist.get("id", "") for artist in artist_items if artist.get("id")]
    album_artist_items = [artist for artist in album.get("artists", []) if isinstance(artist, dict)]
    album_artists = [
        {
            "id": str(artist.get("id") or "").strip(),
            "name": str(artist.get("name") or "").strip(),
        }
        for artist in album_artist_items
        if artist.get("name")
    ]
    genres = _spotify_artist_genres_from_track(track)
    album_art_highres = images[0]["url"] if len(images) > 0 else None
    album_art_medres = images[1]["url"] if len(images) > 1 else album_art_highres
    album_art_lowres = images[2]["url"] if len(images) > 2 else album_art_medres
    return {
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
        "isrc": track.get("external_ids", {}).get("isrc"),
        "popularity": track.get("popularity", 0),
        "genre": genres[0] if genres else "",
        "genres": genres,
    }


async def _with_spotify_artist_genres(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("genres"):
        return metadata
    artist_ids = [str(value) for value in metadata.get("artist_ids") or [] if str(value or "").strip()]
    if not artist_ids:
        return metadata
    try:
        data = await _spotify_get_json("https://api.spotify.com/v1/artists", params={"ids": ",".join(artist_ids[:50])})
    except Exception:
        return metadata
    genres = []
    for artist in data.get("artists") or []:
        if not isinstance(artist, dict):
            continue
        for genre in artist.get("genres") or []:
            text = str(genre or "").strip()
            if text and text not in genres:
                genres.append(text)
    if not genres:
        return metadata
    return {
        **metadata,
        "genre": genres[0],
        "genres": genres,
    }


def _spotify_artist_genres_from_track(track: dict[str, Any]) -> list[str]:
    genres = []
    for artist in track.get("artists", []):
        if not isinstance(artist, dict):
            continue
        for genre in artist.get("genres") or []:
            text = str(genre or "").strip()
            if text and text not in genres:
                genres.append(text)
    return genres


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
    return ""


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
        *_yt_dlp_common_args(),
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


def _yt_dlp_common_args(*, use_cookies: bool = True, extractor_args: list[str] | None = None) -> list[str]:
    args = _yt_dlp_js_args()
    for extractor_arg in extractor_args or []:
        args.extend(["--extractor-args", extractor_arg])
    cookies_path = _get_youtube_cookies_path() if use_cookies else None
    if cookies_path:
        args.extend(["--cookies", cookies_path])
    return args


def _yt_dlp_js_args() -> list[str]:
    deno = shutil.which("deno")
    if deno:
        return ["--js-runtimes", f"deno:{deno}", "--remote-components", "ejs:github"]
    node = shutil.which("node")
    if node:
        return ["--js-runtimes", f"node:{node}", "--remote-components", "ejs:github"]
    return []


def _get_youtube_cookies_path() -> str | None:
    global _youtube_cookies_cache_time, _youtube_cookies_path

    configured_path = (os.getenv("YTDLP_COOKIES_PATH") or os.getenv("YOUTUBE_COOKIES_PATH") or "").strip()
    if configured_path:
        if Path(configured_path).exists():
            return configured_path
        logger.warning("configured YouTube cookies path does not exist: %s", configured_path)

    if _youtube_cookies_path and Path(_youtube_cookies_path).exists() and _cache_fresh(_youtube_cookies_cache_time):
        return _youtube_cookies_path

    payload = _load_youtube_token_payload()
    cookies = payload.get("cookies") if isinstance(payload, dict) else None
    if isinstance(cookies, list) and cookies:
        path = Path(tempfile.gettempdir()) / "yt_cookies_from_current.txt"
        _write_netscape_youtube_cookies(cookies, path)
        _youtube_cookies_path = str(path)
        _youtube_cookies_cache_time = datetime.utcnow()
        logger.info("wrote YouTube cookies from current token payload")
        return _youtube_cookies_path

    cookie_blob = _download_youtube_gcs_text(os.getenv("YOUTUBE_COOKIES_GCS_PATH", "yt-tokens/cookies.txt"))
    if cookie_blob:
        path = Path(tempfile.gettempdir()) / "yt_cookies.txt"
        path.write_text(cookie_blob, encoding="utf-8")
        _youtube_cookies_path = str(path)
        _youtube_cookies_cache_time = datetime.utcnow()
        logger.info("downloaded YouTube cookies from GCS")
        return _youtube_cookies_path

    return None


def _load_youtube_token_payload() -> dict[str, Any] | None:
    global _youtube_token_cache, _youtube_token_cache_time

    if _youtube_token_cache and _cache_fresh(_youtube_token_cache_time):
        return _youtube_token_cache

    token_text = _download_youtube_gcs_text(os.getenv("YOUTUBE_TOKEN_GCS_PATH", "yt-tokens/current.json"))
    if not token_text:
        return None
    try:
        payload = json.loads(token_text)
    except json.JSONDecodeError:
        logger.warning("YouTube token payload from GCS is not valid JSON")
        return None
    if not isinstance(payload, dict):
        return None
    _youtube_token_cache = payload
    _youtube_token_cache_time = datetime.utcnow()
    return payload


def youtube_auth_status() -> dict[str, Any]:
    payload = _load_youtube_token_payload()
    cookies = payload.get("cookies") if isinstance(payload, dict) else None
    extracted_at = payload.get("extracted_at") if isinstance(payload, dict) else None
    extracted_epoch = _parse_utc_epoch(str(extracted_at)) if extracted_at else None
    return {
        "has_po_token": bool(payload and payload.get("po_token")),
        "has_visitor_data": bool(payload and payload.get("visitor_data")),
        "has_cookies": bool(isinstance(cookies, list) and cookies),
        "cookie_count": len(cookies) if isinstance(cookies, list) else 0,
        "is_authenticated": bool(payload and payload.get("is_authenticated")),
        "is_logged_in": bool(payload and payload.get("is_logged_in")),
        "extracted_at": extracted_at,
        "extracted_epoch": extracted_epoch,
        "age_seconds": round(time.time() - extracted_epoch, 3) if extracted_epoch else None,
    }


def _parse_utc_epoch(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.timestamp()


def _download_youtube_gcs_text(blob_path: str) -> str | None:
    try:
        from google.cloud import storage

        project_id = os.getenv("GCP_PROJECT_ID", "imposing-kayak-422917-b0")
        bucket_name = os.getenv("GCP_BUCKET_NAME", "shibuya-assets")
        client = storage.Client(project=project_id)
        blob = client.bucket(bucket_name).blob(blob_path)
        if not blob.exists():
            return None
        return blob.download_as_text()
    except Exception as exc:
        logger.warning("failed to load YouTube auth blob %s from GCS: %s", blob_path, exc)
        return None


def _write_netscape_youtube_cookies(cookies: list[Any], path: Path) -> None:
    far_future = int(time.time()) + 365 * 24 * 60 * 60
    lines = ["# Netscape HTTP Cookie File"]
    for cookie in cookies:
        if not isinstance(cookie, dict):
            continue
        domain = str(cookie.get("domain") or "").strip()
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        if not domain or not name:
            continue
        expires = cookie.get("expires", 0)
        try:
            expires_int = int(expires)
        except (TypeError, ValueError):
            expires_int = 0
        if expires_int <= 0:
            expires_int = far_future
        lines.append(
            "\t".join(
                [
                    domain,
                    "TRUE" if domain.startswith(".") else "FALSE",
                    str(cookie.get("path") or "/"),
                    "TRUE" if cookie.get("secure", False) else "FALSE",
                    str(expires_int),
                    name,
                    value,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cache_fresh(cache_time: datetime | None, *, minutes: int = 5) -> bool:
    return bool(cache_time and datetime.utcnow() - cache_time < timedelta(minutes=minutes))


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
