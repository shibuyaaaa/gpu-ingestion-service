import httpx

from app.album_metadata import (
    AlbumMetadataResolver,
    SpotifyRateLimited,
    parse_spotify_public_album_page,
    parse_spotify_public_track_page,
)


def test_public_track_page_parser_extracts_album_identity_and_art():
    metadata = parse_spotify_public_track_page(
        """
        <meta property="og:title" content="Big Poppa - 2005 Remaster"/>
        <meta property="og:description" content="The Notorious B.I.G. · Ready to Die (The Remaster) · Song · 1994"/>
        <meta property="og:image" content="https://i.scdn.co/image/cover"/>
        <meta name="music:album" content="https://open.spotify.com/album/2HTbQ0RHwukKVXAlTmCZP2"/>
        <meta name="music:release_date" content="1994-09-13"/>
        <meta name="music:duration" content="253"/>
        <meta name="music:musician" content="https://open.spotify.com/artist/5me0Irg2ANcsgc93uaYrpb"/>
        """,
        spotify_id="2g8HN35AnVGIk7B8yMucww",
    )

    assert metadata["album_id"] == "2HTbQ0RHwukKVXAlTmCZP2"
    assert metadata["album"] == "Ready to Die (The Remaster)"
    assert metadata["album_art_highres"] == "https://i.scdn.co/image/cover"
    assert metadata["release_date"] == "1994-09-13"
    assert metadata["album_artists"] == [{"id": "5me0Irg2ANcsgc93uaYrpb", "name": "The Notorious B.I.G."}]
    assert metadata["duration_ms"] == 253000


def test_public_album_page_parser_extracts_release_shape():
    metadata = parse_spotify_public_album_page(
        """
        <meta property="og:title" content="Ready to Die (The Remaster) - Album by The Notorious B.I.G. | Spotify"/>
        <meta property="og:description" content="The Notorious B.I.G. · album · 1994 · 19 songs"/>
        <meta property="og:image" content="https://i.scdn.co/image/cover"/>
        <meta name="music:release_date" content="1994-09-13"/>
        <meta name="music:musician" content="https://open.spotify.com/artist/5me0Irg2ANcsgc93uaYrpb"/>
        """,
        album_id="2HTbQ0RHwukKVXAlTmCZP2",
    )

    assert metadata["album"] == "Ready to Die (The Remaster)"
    assert metadata["album_type"] == "album"
    assert metadata["total_tracks"] == 19
    assert metadata["release_date"] == "1994-09-13"
    assert metadata["album_artists"] == [{"id": "5me0Irg2ANcsgc93uaYrpb", "name": "The Notorious B.I.G."}]


async def test_resolver_prefers_api_fields_when_api_first(tmp_path, monkeypatch):
    resolver = AlbumMetadataResolver(cache_path=tmp_path / "cache.json", resolver="api-first")

    async def fake_api_tracks(spotify_ids):
        return {
            "track-1": {
                "spotify_id": "track-1",
                "album_id": "api-album",
                "album": "API Album",
                "album_art_url": "https://api-cover",
                "metadata_source": "spotify_api",
            }
        }

    async def forbidden_public(spotify_id):
        raise AssertionError("public fallback should not be used when API succeeds")

    monkeypatch.setattr(resolver, "_fetch_api_tracks", fake_api_tracks)
    monkeypatch.setattr(resolver, "_fetch_public_track_metadata", forbidden_public)

    result = await resolver.resolve_many(["track-1"], prefer_api=True)

    assert result["track-1"]["album_id"] == "api-album"
    assert result["track-1"]["album"] == "API Album"
    assert result["track-1"]["metadata_source"] == "spotify_api"


async def test_resolver_uses_public_fallback_when_api_rate_limited(tmp_path, monkeypatch):
    resolver = AlbumMetadataResolver(cache_path=tmp_path / "cache.json", resolver="api-first")

    async def fake_api_tracks(spotify_ids):
        raise SpotifyRateLimited("limited")

    async def fake_public(spotify_id):
        return {
            "spotify_id": spotify_id,
            "album_id": "public-album",
            "album": "Public Album",
            "album_art_url": "https://public-cover",
            "metadata_source": "spotify_public_page",
        }

    monkeypatch.setattr(resolver, "_fetch_api_tracks", fake_api_tracks)
    monkeypatch.setattr(resolver, "_fetch_public_track_metadata", fake_public)

    result = await resolver.resolve_many(["track-1"], prefer_api=True)

    assert result["track-1"]["album_id"] == "public-album"
    assert result["track-1"]["metadata_source"] == "spotify_public_page"


async def test_public_fetch_merges_track_and_album_pages(tmp_path, monkeypatch):
    resolver = AlbumMetadataResolver(cache_path=tmp_path / "cache.json", resolver="public-first")

    class FakeResponse:
        def __init__(self, text):
            self.status_code = 200
            self.text = text

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            if "/track/" in url:
                return FakeResponse(
                    """
                    <meta property="og:title" content="Song"/>
                    <meta property="og:description" content="Artist · Single Name · Song · 2026"/>
                    <meta name="music:album" content="https://open.spotify.com/album/album1"/>
                    """
                )
            return FakeResponse(
                """
                <meta property="og:title" content="Single Name - Single by Artist | Spotify"/>
                <meta property="og:description" content="Artist · single · 2026 · 1 songs"/>
                <meta property="og:image" content="https://cover"/>
                <meta name="music:release_date" content="2026-01-01"/>
                <meta name="music:musician" content="https://open.spotify.com/artist/artist-1"/>
                """
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    metadata = await resolver._fetch_public_track_metadata("track-1")

    assert metadata["album_id"] == "album1"
    assert metadata["album"] == "Single Name"
    assert metadata["album_type"] == "single"
    assert metadata["total_tracks"] == 1
    assert metadata["album_art_highres"] == "https://cover"
