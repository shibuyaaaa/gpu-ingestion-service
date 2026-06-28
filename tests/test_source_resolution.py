import pytest

import app.legacy.utils.source as source_module
from app.jobs.adapters import BulkDissectAdapter
from app.jobs.adapters import (
    _enrich_payload_spotify_metadata_if_needed,
    _merge_metadata_prefer_enriched,
    _payload_spotify_metadata_needs_enrichment,
    _with_default_music_genre,
)
from app.legacy.utils.source import (
    _artists_from_embed_html,
    _first_text,
    _format_spotify_track,
    _with_spotify_artist_genres,
    download_youtube_audio,
    extract_youtube_video_id,
)


def test_source_field_is_canonical_source():
    source = BulkDissectAdapter._source_from_payload({"source": "Daft Punk One More Time"})

    assert source == "Daft Punk One More Time"


def test_youtube_url_is_accepted_as_source():
    source = BulkDissectAdapter._source_from_payload({"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})

    assert source == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_extract_youtube_video_id(url):
    assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("field", ["spotify_source", "spotify_url", "spotify_query"])
def test_source_aliases_are_accepted(field):
    source = BulkDissectAdapter._source_from_payload({field: "https://open.spotify.com/track/abc"})

    assert source == "https://open.spotify.com/track/abc"


def test_source_is_required():
    with pytest.raises(RuntimeError):
        BulkDissectAdapter._source_from_payload({"local_path": "/tmp/song.wav"})


def test_partial_crawler_spotify_metadata_needs_enrichment():
    assert _payload_spotify_metadata_needs_enrichment(
        source="spotify:track:abc",
        metadata={"title": "Song", "artist": "Artist", "popularity": 90},
    )
    assert not _payload_spotify_metadata_needs_enrichment(
        source="spotify:track:abc",
        metadata={
            "title": "Song",
            "artist": "Artist",
            "album_id": "album-1",
            "album": "Album",
            "album_art_url": "https://img",
            "album_art_highres": "https://img",
            "genre": "pop",
        },
    )


def test_enriched_metadata_wins_over_partial_payload():
    merged = _merge_metadata_prefer_enriched(
        {
            "title": "Spotify Title",
            "artist": "Spotify Artist",
            "album_art_url": "https://cover",
            "album_art_highres": "https://cover-hi",
            "genre": "pop",
        },
        {
            "title": "Chart Title",
            "artist": "Chart Artist",
            "popularity": 90,
        },
    )

    assert merged["title"] == "Spotify Title"
    assert merged["album_art_url"] == "https://cover"
    assert merged["genre"] == "pop"
    assert merged["popularity"] == 90


def test_spotify_metadata_without_genre_gets_honest_default():
    metadata = _with_default_music_genre(
        {
            "title": "Song",
            "artist": "Artist",
            "album_art_url": "https://cover",
            "genres": [],
        }
    )

    assert metadata["genre"] == "Music"
    assert metadata["genres"] == ["Music"]
    assert metadata["genre_source"] == "fallback_music"


def test_spotify_track_format_preserves_album_identity_and_creators():
    metadata = _format_spotify_track(
        {
            "id": "track-1",
            "name": "Song",
            "artists": [{"id": "artist-1", "name": "Artist"}],
            "album": {
                "id": "album-1",
                "name": "Album",
                "album_type": "single",
                "release_date": "2026-01-01",
                "total_tracks": 1,
                "artists": [{"id": "artist-1", "name": "Artist"}],
                "images": [
                    {"url": "https://hi", "height": 640},
                    {"url": "https://med", "height": 300},
                    {"url": "https://low", "height": 64},
                ],
            },
            "duration_ms": 123000,
            "external_ids": {"isrc": "US123"},
            "popularity": 50,
        }
    )

    assert metadata["album_id"] == "album-1"
    assert metadata["album"] == "Album"
    assert metadata["album_type"] == "single"
    assert metadata["album_artists"] == [{"id": "artist-1", "name": "Artist"}]
    assert metadata["album_art_highres"] == "https://hi"
    assert metadata["album_art_medres"] == "https://med"
    assert metadata["album_art_lowres"] == "https://low"


@pytest.mark.asyncio
async def test_partial_crawler_metadata_is_enriched_from_spotify_source(monkeypatch):
    async def fake_resolve_source_metadata(source: str):
        assert source == "spotify:track:abc"
        return {
            "source": source,
            "spotify_metadata": {
                "spotify_id": "abc",
                "title": "Resolved Song",
                "artist": "Resolved Artist",
                "album_id": "album-1",
                "album": "Resolved Album",
                "album_art_url": "https://cover",
                "album_art_highres": "https://cover-hi",
                "album_art_medres": "https://cover-med",
                "album_art_lowres": "https://cover-low",
                "genre": "indie pop",
                "genres": ["indie pop"],
            },
        }

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", fake_resolve_source_metadata)

    metadata = await _enrich_payload_spotify_metadata_if_needed(
        source="spotify:track:abc",
        metadata={"spotify_id": "abc", "title": "Chart Song", "artist": "Chart Artist", "popularity": 80},
    )

    assert metadata["title"] == "Resolved Song"
    assert metadata["album_id"] == "album-1"
    assert metadata["album"] == "Resolved Album"
    assert metadata["album_art_url"] == "https://cover"
    assert metadata["genre"] == "indie pop"
    assert metadata["popularity"] == 80


def test_spotify_embed_artist_parser():
    html = '"artists":[{"name":"Bello\\u0026Dallas","uri":"spotify:artist:2zW"}]'

    assert _artists_from_embed_html(html) == ["Bello&Dallas"]


def test_first_text_uses_first_non_empty_list_value():
    assert _first_text(["", None, "Music"]) == "Music"
    assert _first_text("Music") == ""


@pytest.mark.asyncio
async def test_spotify_artist_genre_enrichment_fails_open(monkeypatch):
    async def fake_spotify_get_json(*args, **kwargs):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr("app.legacy.utils.source._spotify_get_json", fake_spotify_get_json)
    metadata = {"artist_ids": ["artist-1"], "title": "Song"}

    assert await _with_spotify_artist_genres(metadata) == metadata


def test_chorus_fallback_uses_longest_useful_segment():
    segment = BulkDissectAdapter._find_chorus_segment(
        [
            {"id": "seg-0", "start": 0.0, "end": 0.08, "label": "start"},
            {"id": "seg-1", "start": 0.08, "end": 27.59, "label": "intro"},
            {"id": "seg-2", "start": 27.59, "end": 49.78, "label": "solo"},
            {"id": "seg-3", "start": 49.78, "end": 70.09, "label": "solo"},
            {"id": "seg-4", "start": 70.09, "end": 110.09, "label": "solo"},
            {"id": "seg-5", "start": 200.1, "end": 205.19, "label": "end"},
        ]
    )

    assert segment["id"] == "seg-4"


@pytest.mark.asyncio
async def test_download_youtube_audio_retries_transient_yt_dlp_error(monkeypatch, tmp_path):
    calls = []
    sleeps = []

    class FakeProc:
        def __init__(self, returncode: int, stderr: bytes, *, write_output: bool = False):
            self.returncode = returncode
            self.stderr = stderr
            self.write_output = write_output

        async def communicate(self):
            if self.write_output:
                (tmp_path / "source.wav").write_bytes(b"audio")
            else:
                (tmp_path / "source.part").write_bytes(b"partial")
            return b"", self.stderr

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        calls.append(cmd)
        if len(calls) == 1:
            return FakeProc(1, b"ERROR: unable to download video data: HTTP Error 403: Forbidden\n")
        return FakeProc(0, b"", write_output=True)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setenv("YTDLP_DOWNLOAD_ATTEMPTS", "3")
    monkeypatch.setenv("YTDLP_RETRY_DELAY_SECONDS", "0.5")
    monkeypatch.setattr("app.legacy.utils.source.shutil.which", lambda binary: "/usr/bin/yt-dlp")
    monkeypatch.setattr("app.legacy.utils.source.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("app.legacy.utils.source.asyncio.sleep", fake_sleep)

    result = await download_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert result == str(tmp_path / "source.wav")
    assert len(calls) == 2
    assert sleeps == [0.5]
    assert not (tmp_path / "source.part").exists()


@pytest.mark.asyncio
async def test_download_youtube_audio_passes_configured_cookies(monkeypatch, tmp_path):
    calls = []
    cookies_path = tmp_path / "cookies.txt"
    cookies_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    class FakeProc:
        returncode = 0

        async def communicate(self):
            (tmp_path / "source.wav").write_bytes(b"audio")
            return b"", b""

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setenv("YTDLP_COOKIES_PATH", str(cookies_path))
    monkeypatch.setattr("app.legacy.utils.source.shutil.which", lambda binary: "/usr/bin/yt-dlp")
    monkeypatch.setattr("app.legacy.utils.source.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    result = await download_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert result == str(tmp_path / "source.wav")
    assert "--cookies" in calls[0]
    assert calls[0][calls[0].index("--cookies") + 1] == str(cookies_path)


def test_yt_dlp_common_args_writes_current_payload_cookies(monkeypatch, tmp_path):
    source_module._youtube_token_cache = None
    source_module._youtube_token_cache_time = None
    source_module._youtube_cookies_path = None
    source_module._youtube_cookies_cache_time = None

    monkeypatch.setattr("app.legacy.utils.source.shutil.which", lambda binary: None)
    monkeypatch.setattr("app.legacy.utils.source.tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "app.legacy.utils.source._download_youtube_gcs_text",
        lambda blob_path: '{"cookies":[{"domain":".youtube.com","path":"/","secure":true,"expires":-1,"name":"SAPISID","value":"secret"}]}'
        if blob_path == "yt-tokens/current.json"
        else None,
    )

    args = source_module._yt_dlp_common_args()

    cookies_path = tmp_path / "yt_cookies_from_current.txt"
    assert args == ["--cookies", str(cookies_path)]
    content = cookies_path.read_text(encoding="utf-8")
    assert ".youtube.com\tTRUE\t/\tTRUE\t" in content
    assert "SAPISID\tsecret" in content


@pytest.mark.asyncio
async def test_download_youtube_audio_does_not_retry_permanent_yt_dlp_error(monkeypatch, tmp_path):
    calls = []

    class FakeProc:
        returncode = 1

        async def communicate(self):
            return b"", b"ERROR: Private video\n"

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setenv("YTDLP_DOWNLOAD_ATTEMPTS", "3")
    monkeypatch.setenv("YTDLP_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr("app.legacy.utils.source.shutil.which", lambda binary: "/usr/bin/yt-dlp")
    monkeypatch.setattr("app.legacy.utils.source.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match="Private video"):
        await download_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert len(calls) == 1
