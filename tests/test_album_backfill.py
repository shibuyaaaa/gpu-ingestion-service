from app.tools.album_backfill import (
    AlbumBackfillAction,
    SongAlbumSnapshot,
    _metadata_from_analysis,
    _spotify_id_from_song,
    plan_album_action,
    unique_album_count,
)


def test_album_backfill_unique_album_count_dedupes_repeated_spotify_album_ids():
    actions = [
        AlbumBackfillAction(
            song_id="song-1",
            title="Song 1",
            artists=["Artist"],
            action_type="upsert_album_link",
            source="analysis_json",
            reason="test",
            metadata={"album_id": "album-1", "album": "Album"},
        ),
        AlbumBackfillAction(
            song_id="song-2",
            title="Song 2",
            artists=["Artist"],
            action_type="upsert_album_link",
            source="spotify_track",
            reason="test",
            metadata={"album_id": "album-1", "album": "Album"},
        ),
        AlbumBackfillAction(
            song_id="song-3",
            title="Song 3",
            artists=["Artist"],
            action_type="review_required",
            source="spotify_search",
            reason="test",
            metadata={"album": "Text Only"},
        ),
    ]

    assert unique_album_count(actions) == 1


def test_album_backfill_extracts_gpu_spotify_metadata_before_legacy_spotify_block():
    payload = {
        "spotify": {"id": "legacy-track", "album": "Legacy Album"},
        "gpu_ingestion": {
            "spotify_metadata": {
                "spotify_id": "track-1",
                "album_id": "album-1",
                "album": "Album",
            }
        },
    }
    song = SongAlbumSnapshot(id="song-1", title="Song", artists=["Artist"], analysis_json=payload)
    metadata = _metadata_from_analysis(payload)

    assert metadata["spotify_id"] == "track-1"
    assert metadata["album_id"] == "album-1"
    assert _spotify_id_from_song(song, metadata) == "track-1"


async def test_album_backfill_can_skip_review_search_for_no_id_rows():
    class Cache:
        async def get(self, spotify_id):
            raise AssertionError("cache should not be used without a Spotify track ID")

    song = SongAlbumSnapshot(id="song-1", title="Loose Song", artists=["Artist"], analysis_json={})

    action = await plan_album_action(song, Cache(), allow_review_search=False)

    assert action.action_type == "no_trusted_album_source"
    assert action.source == "none"
    assert "review search skipped" in action.reason
    assert action.review_candidates == []
