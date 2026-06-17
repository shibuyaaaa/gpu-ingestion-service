import pytest

from app.jobs.adapters import BulkDissectAdapter
from app.legacy.utils.source import extract_youtube_video_id


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
