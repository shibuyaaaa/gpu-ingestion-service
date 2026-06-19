import pytest

from app.jobs.adapters import BulkDissectAdapter
from app.legacy.utils.source import _artists_from_embed_html, extract_youtube_video_id


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
