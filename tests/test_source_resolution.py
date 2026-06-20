import pytest

from app.jobs.adapters import BulkDissectAdapter
from app.legacy.utils.source import _artists_from_embed_html, download_youtube_audio, extract_youtube_video_id


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


def test_spotify_embed_artist_parser():
    html = '"artists":[{"name":"Bello\\u0026Dallas","uri":"spotify:artist:2zW"}]'

    assert _artists_from_embed_html(html) == ["Bello&Dallas"]


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
