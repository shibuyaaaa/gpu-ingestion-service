import asyncio
import json
import os
from contextlib import contextmanager
import tempfile
import uuid
from pathlib import Path

from app.config import Settings
from app.jobs.adapters import BulkDissectAdapter
from app.jobs.context import JobContext
from app.library_membership import LibraryLookupResult, LibraryMembershipChecker, LibrarySong
from app.library_writer import LibraryPublishResult, LibraryWriter, _analysis_payload, _analysis_bpms
from app.legacy.db import DBClient
from app.legacy.utils import GCSClient
from app.queue import JobStage, JobStore, JobType


def _row(song_id: str, title: str, artists: list[str]) -> dict:
    return {
        "id": song_id,
        "title": title,
        "artists": artists,
        "key": "C major",
        "genre": "Pop",
        "cover_art_url": None,
    }


def test_library_publish_result_serializes_uuid_rows():
    stem_id = uuid.uuid4()
    result = LibraryPublishResult(
        enabled=True,
        song_id=str(uuid.uuid4()),
        status="partial",
        inserted_stems=[{"id": stem_id, "stem_type": "chord"}],
    )

    assert result.to_dict()["inserted_stems"][0]["id"] == str(stem_id)


def test_beat_analysis_bpm_uses_beat_times_not_beat_positions():
    analysis = {
        "bpm": 105,
        "beats": [0.09, 0.65, 1.23, 1.81, 2.36, 2.94, 3.53],
        "beat_positions": [1, 2, 3, 4, 1, 2, 3],
    }

    all_in_one_bpm, beat_analysis_bpm = _analysis_bpms(analysis)

    assert all_in_one_bpm == 105
    assert beat_analysis_bpm is not None
    assert 100 <= beat_analysis_bpm <= 110


def test_analysis_payload_exposes_enriched_beat_grid_key_and_genre(tmp_path):
    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue(
        {
            "job_id": "analysis-payload",
            "job_type": "bulk_dissect",
            "source": "song",
        }
    )
    job.artifacts.update(
        {
            "analysis": {
                "bpm": 105,
                "beat_analysis_bpm": 105.1,
                "key": "F# minor",
                "beats": [0.0, 0.5, 1.0, 1.5],
                "downbeats": [0.0],
                "upbeats": [0.5, 1.0, 1.5],
                "beat_grid": [
                    {"time": 0.0, "position": 1, "is_downbeat": True, "is_upbeat": False},
                    {"time": 0.5, "position": 2, "is_downbeat": False, "is_upbeat": True},
                ],
                "segments": [{"id": "seg-0", "start": 0, "end": 4, "label": "chorus"}],
            },
            "spotify_metadata": {"title": "Song", "artist": "Artist", "genre": "indietronica"},
        }
    )

    payload = _analysis_payload(job, status="partial")

    assert payload["gpu_ingestion"]["summary"]["key"] == "F# minor"
    assert payload["gpu_ingestion"]["summary"]["genre"] == "indietronica"
    assert payload["gpu_ingestion"]["summary"]["beat_count"] == 4
    assert payload["gpu_ingestion"]["summary"]["downbeat_count"] == 1
    assert payload["gpu_ingestion"]["summary"]["upbeat_count"] == 3
    assert payload["analysis"]["beat_grid"][1]["is_upbeat"] is True


async def test_library_writer_skips_db_when_job_requests_no_library_write():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue(
            {
                "job_id": "skip-write",
                "job_type": "bulk_dissect",
                "source": "song",
                "skip_library_write": True,
            }
        )
        writer = LibraryWriter(
            db=DBClient(database_url=""),
            settings=settings,
            membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        )

        published = await writer.publish_segment(
            job=job,
            segment={"id": "seg-1", "start": 0.0, "end": 10.0, "label": "chorus"},
            segment_result={"outputs": {"other": "https://cdn.test/chord.mp3"}},
            status="partial",
        )
        completed = await writer.mark_complete(job=job)

        assert published.to_dict() == {
            "enabled": False,
            "song_id": None,
            "status": "skipped_by_job",
            "inserted_stems": [],
            "error": None,
        }
        assert completed.status == "skipped_by_job"
        assert completed.enabled is False


async def test_library_writer_locks_song_identity_and_writes_source_audio_and_bpm(tmp_path):
    class FakeTransaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAcquire:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def __init__(self, conn):
            self.conn = conn

        def acquire(self):
            return FakeAcquire(self.conn)

    class FakeConn:
        def __init__(self):
            self.executed = []
            self.fetchrow_queries = []
            self.song_insert_args = None
            self.stem_insert_args = None

        def transaction(self):
            return FakeTransaction()

        async def execute(self, query, *args):
            self.executed.append((query, args))
            return "OK"

        async def fetchrow(self, query, *args):
            self.fetchrow_queries.append(query)
            if "POSITION($1 IN COALESCE(analysis_json::text" in query:
                return None
            if "WHERE youtube_url = $1" in query:
                return None
            if "LOWER(TRIM(s.title))" in query:
                return None
            if "INSERT INTO songs" in query:
                self.song_insert_args = args
                return {"id": "song-1"}
            if "INSERT INTO artists" in query:
                return {"id": "artist-1"}
            if "SELECT artist_id" in query:
                return {"artist_id": "artist-1"}
            if "SELECT id\n            FROM stems" in query:
                return None
            if "INSERT INTO stems" in query:
                self.stem_insert_args = args
                return {
                    "id": "stem-1",
                    "song_id": "song-1",
                    "stem_type": args[1],
                    "audio_url": args[2],
                    "segment": args[7],
                    "start_time": args[4],
                    "end_time": args[5],
                }
            if "FROM songs s" in query and "GROUP BY s.id" in query:
                return {"id": "song-1", "title": "Song", "artists": ["Artist"]}
            raise AssertionError(f"unexpected fetchrow query: {query}")

        async def fetchval(self, query, *args):
            if "MAX(mix_id)" in query:
                return 100000
            if "SELECT analysis_json" in query:
                return None
            raise AssertionError(f"unexpected fetchval query: {query}")

        async def fetch(self, query, *args):
            if "FROM stems" in query:
                return [
                    {
                        "stem_type": "chord",
                        "segment": "chorus",
                        "audio_url": "https://cdn.test/chord.mp3",
                        "start_time": 0.0,
                        "end_time": 8.0,
                    }
                ]
            raise AssertionError(f"unexpected fetch query: {query}")

    class FakeDBClient(DBClient):
        def __init__(self, conn):
            self.conn = conn
            self.min_size = 1
            self.max_size = 5

        async def pool(self):
            return FakePool(self.conn)

        def status(self):
            return {"configured": True}

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "writer", "job_type": "bulk_dissect", "source": "song"})
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "source": "song",
            "source_audio_url": "https://cdn.test/full.mp3",
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "spotify_metadata": {"title": "Song", "artist": "Artist"},
            "analysis": {"bpm": 115},
        },
    )
    job = store.get(job.id)
    conn = FakeConn()
    writer = LibraryWriter(
        db=FakeDBClient(conn),
        settings=settings,
        membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
    )

    result = await writer.publish_segment(
        job=job,
        segment={"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"},
        segment_result={"outputs": {"other": "https://cdn.test/chord.mp3"}},
        status="partial",
    )

    assert result.error is None
    assert conn.executed[0][0] == "SELECT pg_advisory_xact_lock(hashtext($1))"
    assert conn.executed[0][1] == ("gpu_ingestion_song:youtube:dQw4w9WgXcQ",)
    assert conn.song_insert_args[2] == 115.0
    assert conn.song_insert_args[3] == 115.0
    assert conn.song_insert_args[5] == "https://cdn.test/full.mp3"
    assert conn.stem_insert_args[1] == "chord"
    assert conn.stem_insert_args[2] == "https://cdn.test/chord.mp3"
    stem_lookup_queries = [query for query in conn.fetchrow_queries if "FROM stems" in query]
    assert stem_lookup_queries
    assert "ROUND(COALESCE(start_time, -1)::numeric, 3)" in stem_lookup_queries[0]
    assert "ROUND(COALESCE($4::numeric, -1), 3)" in stem_lookup_queries[0]
    assert "ROUND(COALESCE($5::numeric, -1), 3)" in stem_lookup_queries[0]


class FakeDB:
    def __init__(self, batches: list[list[dict]], *, delay: float = 0.0, fail: bool = False):
        self.batches = list(batches)
        self.delay = delay
        self.fail = fail
        self.fetch_calls = 0
        self.warmup_calls = 0

    async def warmup(self) -> bool:
        self.warmup_calls += 1
        if self.fail:
            raise RuntimeError("db down")
        return True

    async def fetch(self, query: str, *args):
        self.fetch_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("db down")
        if self.batches:
            return self.batches.pop(0)
        return []

    def status(self) -> dict:
        return {"configured": True, "pool_created": True, "min_size": 1, "max_size": 5}


async def test_cache_hit_skips_db_fallback():
    settings = Settings(dry_run_mode=True)
    db = FakeDB([[_row("song-1", "One More Time", ["Daft Punk"])]])
    checker = LibraryMembershipChecker(db=db, settings=settings)

    first = await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})
    second = await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})

    assert first.exists is True
    assert first.source == "cache"
    assert second.exists is True
    assert second.source == "cache"
    assert db.fetch_calls == 1
    assert checker.status()["db_fallbacks"] == 0


async def test_cache_hit_preserves_partial_ingestion_status():
    settings = Settings(dry_run_mode=True)
    db = FakeDB(
        [
            [
                {
                    **_row("song-1", "One More Time", ["Daft Punk"]),
                    "ingestion_status": "partial",
                }
            ]
        ]
    )
    checker = LibraryMembershipChecker(db=db, settings=settings)

    result = await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})

    assert result.exists is True
    assert result.ingestion_status == "partial"
    assert result.is_complete is False
    assert result.to_dict()["song"]["ingestion_status"] == "partial"


async def test_cache_miss_uses_targeted_db_fallback_and_updates_cache():
    settings = Settings(dry_run_mode=True)
    db = FakeDB(
        [
            [_row("song-other", "Harder Better Faster Stronger", ["Daft Punk"])],
            [_row("song-1", "One More Time", ["Daft Punk"])],
        ]
    )
    checker = LibraryMembershipChecker(db=db, settings=settings)

    result = await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})
    cached = await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})

    assert result.exists is True
    assert result.source == "db"
    assert cached.source == "cache"
    assert db.fetch_calls == 2
    assert checker.status()["db_fallbacks"] == 1


async def test_concurrent_cache_loads_share_one_db_fetch():
    settings = Settings(dry_run_mode=True)
    db = FakeDB([[_row("song-1", "One More Time", ["Daft Punk"])]], delay=0.02)
    checker = LibraryMembershipChecker(db=db, settings=settings)

    results = await asyncio.gather(
        *(checker.lookup({"title": "One More Time", "artist": "Daft Punk"}) for _ in range(5))
    )

    assert all(result.exists for result in results)
    assert db.fetch_calls == 1
    assert checker.status()["refreshes"] == 1


async def test_idle_ttl_evicts_cache_after_no_use():
    settings = Settings(
        dry_run_mode=True,
        library_cache_idle_ttl_seconds=0.02,
        library_cache_max_age_seconds=60,
    )
    db = FakeDB([[_row("song-1", "One More Time", ["Daft Punk"])]])
    checker = LibraryMembershipChecker(db=db, settings=settings)

    await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})
    assert checker.status()["cache_loaded"] is True

    await asyncio.sleep(0.04)

    assert checker.status()["cache_loaded"] is False
    assert checker.status()["evictions"] == 1


async def test_each_lookup_resets_idle_eviction_window():
    settings = Settings(
        dry_run_mode=True,
        library_cache_idle_ttl_seconds=0.05,
        library_cache_max_age_seconds=60,
    )
    db = FakeDB([[_row("song-1", "One More Time", ["Daft Punk"])]])
    checker = LibraryMembershipChecker(db=db, settings=settings)

    await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})
    await asyncio.sleep(0.03)
    await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})
    await asyncio.sleep(0.03)

    assert checker.status()["cache_loaded"] is True


async def test_db_failure_returns_error_result_without_raising():
    settings = Settings(dry_run_mode=True)
    checker = LibraryMembershipChecker(db=FakeDB([], fail=True), settings=settings)

    result = await checker.lookup({"title": "One More Time", "artist": "Daft Punk"})

    assert result.exists is False
    assert result.source == "error"
    assert "db down" in result.error


async def test_existing_library_song_completes_before_audio_download(monkeypatch):
    async def fake_resolve_metadata(source: str):
        return {
            "source": source,
            "spotify_metadata": {"title": "One More Time", "artist": "Daft Punk"},
        }

    async def forbidden_youtube_match(*args, **kwargs):
        raise AssertionError("youtube match should not run for existing library songs")

    async def forbidden_download(*args, **kwargs):
        raise AssertionError("download should not run for existing library songs")

    class ExistingLibrary:
        async def lookup(self, metadata):
            return LibraryLookupResult(
                checked=True,
                exists=True,
                source="cache",
                song=LibrarySong(id="song-1", title="One More Time", artists=["Daft Punk"], artist="Daft Punk"),
            )

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", fake_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", forbidden_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", forbidden_download)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue({"job_id": "existing", "job_type": "bulk_dissect", "source": "One More Time"})
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=ExistingLibrary(),
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )

        result = await BulkDissectAdapter().download(job, context)

        assert result.next_stage is None
        assert result.artifacts["final_outputs"]["status"] == "skipped_existing_library_song"
        assert result.artifacts["library_precheck"]["song"]["id"] == "song-1"
        assert result.artifacts["download_timings"]["source_metadata_seconds"] >= 0
        assert result.artifacts["download_timings"]["library_precheck_seconds"] >= 0
        assert result.artifacts["download_timings"]["download_total_seconds"] >= 0


async def test_partial_library_song_continues_download_and_reuses_song_id(monkeypatch):
    async def fake_resolve_metadata(source: str):
        return {
            "source": source,
            "spotify_metadata": {"title": "One More Time", "artist": "Daft Punk"},
        }

    async def fake_youtube_match(resolved: dict):
        return {
            **resolved,
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube_match": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        }

    async def fake_download(youtube_url: str, output_dir: str | Path):
        target = Path(output_dir) / "source.wav"
        target.write_bytes(b"audio")
        return str(target)

    class PartialLibrary:
        async def lookup(self, metadata):
            return LibraryLookupResult(
                checked=True,
                exists=True,
                source="cache",
                song=LibrarySong(
                    id="song-partial",
                    title="One More Time",
                    artists=["Daft Punk"],
                    artist="Daft Punk",
                    ingestion_status="partial",
                ),
            )

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", fake_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", fake_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", fake_download)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
            source_audio_upload_enabled=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue({"job_id": "partial", "job_type": "bulk_dissect", "source": "One More Time"})
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=PartialLibrary(),
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )

        result = await BulkDissectAdapter().download(job, context)

        assert result.next_stage == JobStage.ANALYZE
        assert result.artifacts["existing_library_song_id"] == "song-partial"
        assert result.artifacts["library_precheck"]["ingestion_status"] == "partial"
        assert result.artifacts["library_precheck"]["is_complete"] is False
        assert Path(result.artifacts["audio_path"]).exists()


async def test_download_uploads_full_source_audio_to_identity_path(monkeypatch, tmp_path):
    async def fake_resolve_metadata(source: str):
        return {
            "source": source,
            "spotify_metadata": {"title": "Never Gonna Give You Up", "artist": "Rick Astley"},
        }

    async def fake_youtube_match(resolved: dict):
        return {
            **resolved,
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube_match": {"video_id": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        }

    async def fake_download(youtube_url: str, output_dir: str | Path):
        target = Path(output_dir) / "source.wav"
        target.write_bytes(b"audio")
        return str(target)

    async def fake_convert_to_mp3(source, target):
        Path(target).write_bytes(Path(source).read_bytes())

    class MissingLibrary:
        async def lookup(self, metadata):
            return LibraryLookupResult(checked=True, exists=False, source="miss")

    class FakeGCS:
        def __init__(self):
            self.uploads = []

        async def exists(self, gcs_path):
            return False

        async def upload(self, local_path, gcs_path, *, content_type):
            self.uploads.append((Path(local_path).name, gcs_path, content_type))
            return f"https://cdn.test/{gcs_path}"

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", fake_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", fake_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", fake_download)
    monkeypatch.setattr("app.jobs.adapters.AudioOps.convert_to_mp3", fake_convert_to_mp3)

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        cdn_base_url="https://cdn.test",
        source_audio_upload_enabled=True,
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "source-audio", "job_type": "bulk_dissect", "source": "song"})
    gcs = FakeGCS()
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=gcs,
        db=DBClient(database_url=""),
        library=MissingLibrary(),
        library_writer=LibraryWriter(
            db=DBClient(database_url=""),
            settings=settings,
            membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        ),
    )

    result = await BulkDissectAdapter().download(job, context)

    assert result.artifacts["ingestion_identity_key"] == "youtube:dQw4w9WgXcQ"
    assert result.artifacts["source_audio_url"] == (
        "https://cdn.test/gpu-ingestion/cache/source-audio/youtube-dQw4w9WgXcQ/full.mp3"
    )
    assert gcs.uploads == [
        (
            "source-full.mp3",
            "gpu-ingestion/cache/source-audio/youtube-dQw4w9WgXcQ/full.mp3",
            "audio/mpeg",
        )
    ]
    assert result.artifacts["download_timings"]["source_audio_mp3_convert_seconds"] >= 0


async def test_skip_library_precheck_forces_download_even_when_song_exists(monkeypatch):
    async def fake_resolve_metadata(source: str):
        return {
            "source": source,
            "spotify_metadata": {"title": "One More Time", "artist": "Daft Punk"},
        }

    async def fake_youtube_match(resolved: dict):
        return {
            **resolved,
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube_match": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        }

    async def fake_download(youtube_url: str, output_dir: str | Path):
        target = Path(output_dir) / "source.wav"
        target.write_bytes(b"audio")
        return str(target)

    class ForbiddenLibrary:
        async def lookup(self, metadata):
            raise AssertionError("library lookup should be skipped")

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", fake_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", fake_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", fake_download)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue(
            {
                "job_id": "force-download",
                "job_type": "bulk_dissect",
                "source": "One More Time",
                "skip_library_precheck": True,
                "skip_library_write": True,
            }
        )
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=ForbiddenLibrary(),
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )

        result = await BulkDissectAdapter().download(job, context)

        assert result.next_stage == JobStage.ANALYZE
        assert result.artifacts["library_precheck"]["source"] == "skipped_by_job"
        assert result.artifacts["skip_library_precheck"] is True
        assert result.artifacts["skip_library_write"] is True
        assert result.artifacts["download_timings"]["source_metadata_seconds"] >= 0
        assert result.artifacts["download_timings"]["library_precheck_seconds"] == 0.0
        assert result.artifacts["download_timings"]["youtube_match_seconds"] >= 0
        assert result.artifacts["download_timings"]["youtube_download_seconds"] >= 0
        assert result.artifacts["download_timings"]["wav_copy_seconds"] >= 0
        assert result.artifacts["download_timings"]["download_total_seconds"] >= 0
        assert Path(result.artifacts["audio_path"]).exists()


def test_analysis_payload_preserves_bpm_identity_and_source_audio():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue({"job_id": "payload", "job_type": "bulk_dissect", "source": "song"})
        store.complete_stage(
            job.id,
            next_stage=JobStage.PROCESS,
            artifacts={
                "source": "song",
                "source_audio_url": "https://cdn.test/full.mp3",
                "analyzer_result_url": "https://cdn.test/analyzer_result.json",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "spotify_metadata": {"title": "Song", "artist": "Artist", "key": "D minor"},
                "analysis": {
                    "bpm": 115,
                    "beats": [0.0, 0.5, 1.0, 1.5],
                    "beat_positions": [1, 2, 3, 4],
                    "segments": [{"id": "seg-0"}],
                },
            },
        )
        job = store.get(job.id)

        payload = _analysis_payload(job, status="partial")

        assert _analysis_bpms(job.artifacts["analysis"]) == (115.0, 120.0)
        assert payload["gpu_ingestion"]["identity_key"] == "youtube:dQw4w9WgXcQ"
        assert payload["gpu_ingestion"]["source_audio_url"] == "https://cdn.test/full.mp3"
        assert payload["gpu_ingestion"]["analyzer_result_url"] == "https://cdn.test/analyzer_result.json"
        assert payload["gpu_ingestion"]["summary"]["bpm"] == 115.0
        assert payload["gpu_ingestion"]["summary"]["beat_analysis_bpm"] == 120.0
        assert payload["gpu_ingestion"]["summary"]["key"] == "D minor"


async def test_skip_library_precheck_direct_youtube_source_skips_metadata_probe(monkeypatch):
    async def forbidden_resolve_metadata(source: str):
        raise AssertionError("direct YouTube skip-precheck jobs should not call metadata probe")

    async def forbidden_youtube_match(resolved: dict):
        raise AssertionError("direct YouTube skip-precheck jobs should not search YouTube")

    async def fake_download(youtube_url: str, output_dir: str | Path):
        target = Path(output_dir) / "source.wav"
        target.write_bytes(b"audio")
        return str(target)

    class ForbiddenLibrary:
        async def lookup(self, metadata):
            raise AssertionError("library lookup should be skipped")

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", forbidden_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", forbidden_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", fake_download)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue(
            {
                "job_id": "direct-youtube",
                "job_type": "bulk_dissect",
                "source": "https://youtu.be/dQw4w9WgXcQ",
                "skip_library_precheck": True,
                "skip_library_write": True,
            }
        )
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=ForbiddenLibrary(),
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )

        result = await BulkDissectAdapter().download(job, context)

        assert result.next_stage == JobStage.ANALYZE
        assert result.artifacts["youtube_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result.artifacts["youtube_match"]["metadata_source"] == "skipped_for_direct_youtube_download"
        assert result.artifacts["spotify_metadata"]["metadata_source"] == "skipped_for_direct_youtube_download"
        assert result.artifacts["download_timings"]["source_metadata_seconds"] == 0.0
        assert result.artifacts["download_timings"]["library_precheck_seconds"] == 0.0
        assert result.artifacts["download_timings"]["youtube_match_seconds"] <= 0.001
        assert result.artifacts["download_timings"]["youtube_download_seconds"] >= 0
        assert Path(result.artifacts["audio_path"]).exists()


async def test_direct_youtube_download_reuses_source_audio_cache(monkeypatch):
    calls = {"downloads": 0}

    async def forbidden_resolve_metadata(source: str):
        raise AssertionError("direct YouTube skip path should not fetch metadata")

    async def forbidden_youtube_match(resolved: dict):
        raise AssertionError("direct YouTube skip path should not search YouTube")

    async def fake_download(youtube_url: str, output_dir: str | Path):
        calls["downloads"] += 1
        target = Path(output_dir) / "source.wav"
        target.write_bytes(b"cached-audio")
        return str(target)

    class ForbiddenLibrary:
        async def lookup(self, metadata):
            raise AssertionError("library lookup should be skipped")

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", forbidden_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", forbidden_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", fake_download)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
            source_audio_cache_enabled=True,
        )
        store = JobStore(settings.queue_db_path)
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=ForbiddenLibrary(),
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )
        first, _ = store.enqueue(
            {
                "job_id": "cache-first",
                "job_type": "bulk_dissect",
                "source": "https://youtu.be/dQw4w9WgXcQ",
                "skip_library_precheck": True,
                "skip_library_write": True,
            }
        )
        second, _ = store.enqueue(
            {
                "job_id": "cache-second",
                "job_type": "bulk_dissect",
                "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "skip_library_precheck": True,
                "skip_library_write": True,
            }
        )

        first_result = await BulkDissectAdapter().download(first, context)
        second_result = await BulkDissectAdapter().download(second, context)

        assert calls["downloads"] == 1
        assert Path(first_result.artifacts["audio_path"]).read_bytes() == b"cached-audio"
        assert Path(second_result.artifacts["audio_path"]).read_bytes() == b"cached-audio"
        assert first_result.artifacts["download_timings"]["youtube_download_seconds"] >= 0
        assert second_result.artifacts["download_timings"]["youtube_download_seconds"] == 0.0
        assert second_result.artifacts["download_timings"]["source_audio_cache_restore_seconds"] >= 0
        assert second_result.artifacts["download_timings"]["source_audio_cache_copy_seconds"] >= 0
        cache_path = settings.work_dir / "source-cache" / "dQw4w9WgXcQ.wav"
        if hasattr(os, "link"):
            assert os.stat(second_result.artifacts["audio_path"]).st_ino == os.stat(cache_path).st_ino


async def test_analyze_reuses_cached_all_in_one_outputs_for_same_youtube_source():
    class FakeAllInOne:
        def __init__(self):
            self.calls = 0

        async def analyze(self, audio_path, output_dir):
            self.calls += 1
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            analysis = {
                "duration": 30.0,
                "segments": [{"id": "seg-0", "start": 0.0, "end": 10.0, "label": "chorus"}],
            }
            result_path = output / "result.json"
            result_path.write_text(json.dumps(analysis), encoding="utf-8")
            stems_dir = output / "fake_stems"
            stems_dir.mkdir(parents=True, exist_ok=True)
            stems = {}
            for stem in ("bass", "drums", "other", "vocals"):
                path = stems_dir / f"{stem}.wav"
                path.write_bytes(f"{stem}-audio".encode())
                stems[stem] = str(path)
            return {
                "analysis": analysis,
                "analyzer_result_path": str(result_path),
                "stem_paths": stems,
                "timings": {"allin1_analyze_seconds": 12.0, "demix_total_seconds": 4.0},
            }

    class FakeGpuUsage:
        def summary(self):
            return {"gpu_utilization_avg_pct": 50.0}

    class FakeModels:
        def __init__(self):
            self.gpu_lock = asyncio.Lock()
            self.all_in_one = FakeAllInOne()

        @contextmanager
        def track_gpu_work(self, *, job_id, model_name):
            yield FakeGpuUsage()

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
            analysis_cache_enabled=True,
            analysis_cache_max_entries=4,
        )
        store = JobStore(settings.queue_db_path, max_depth=20)
        context = JobContext(
            settings=settings,
            store=store,
            models=FakeModels(),
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=None,
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )
        jobs = []
        for index in range(2):
            work_dir = settings.work_dir / "jobs" / f"analysis-cache-{index}"
            work_dir.mkdir(parents=True, exist_ok=True)
            audio_path = work_dir / "input.wav"
            audio_path.write_bytes(b"audio")
            job, _ = store.enqueue(
                {
                    "job_id": f"analysis-cache-{index}",
                    "job_type": "bulk_dissect",
                    "source": "song",
                },
                initial_stage=JobStage.ANALYZE,
                initial_artifacts={
                    "work_dir": str(work_dir),
                    "audio_path": str(audio_path),
                    "source": "song",
                    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                },
            )
            jobs.append(job)

        first = await BulkDissectAdapter().analyze(jobs[0], context)
        second = await BulkDissectAdapter().analyze(jobs[1], context)

        assert context.models.all_in_one.calls == 1
        assert first.artifacts["analysis_timings"]["analysis_cache_hit"] == 0
        assert first.artifacts["analysis_timings"]["allin1_analyze_seconds"] == 12.0
        assert second.artifacts["analysis_timings"]["analysis_cache_hit"] == 1
        assert second.artifacts["analysis_timings"]["allin1_analyze_seconds"] == 0.0
        assert second.artifacts["analysis_timings"]["demix_total_seconds"] == 0.0
        assert second.artifacts["analysis_timings"]["fanout_enqueue_seconds"] >= 0.0
        assert second.artifacts["analysis_timings"]["fanout_child_count"] == 2
        assert second.artifacts["analysis_timings"]["fanout_chord_child_count"] == 1
        assert second.artifacts["analysis_timings"]["fanout_other_child_count"] == 1
        assert Path(second.artifacts["full_stem_paths"]["other"]).exists()
        assert second.artifacts["fanout"]["strategy"] == "segment_chord_and_other_jobs_priority_fanout"
        assert second.artifacts["fanout"]["child_count"] == 2
        assert [child["process_mode"] for child in second.artifacts["fanout"]["children"]] == [
            "segment_chord",
            "segment_other",
        ]
        fanout_children = store.child_jobs(jobs[1].id)
        assert fanout_children
        child_records = [store.get(child["id"]) for child in fanout_children]
        assert all(child is not None for child in child_records)
        assert all(child.artifacts["analysis"]["duration"] == 30.0 for child in child_records if child)


async def test_missing_library_song_continues_download_stage(monkeypatch):
    async def fake_resolve_metadata(source: str):
        return {
            "source": source,
            "spotify_metadata": {"title": "New Song", "artist": "New Artist"},
        }

    async def fake_youtube_match(resolved: dict):
        return {
            **resolved,
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube_match": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        }

    async def fake_download(youtube_url: str, output_dir: str | Path):
        target = Path(output_dir) / "source.wav"
        target.write_bytes(b"audio")
        return str(target)

    class MissingLibrary:
        async def lookup(self, metadata):
            return LibraryLookupResult(checked=True, exists=False, source="miss")

    monkeypatch.setattr("app.jobs.adapters.resolve_source_metadata", fake_resolve_metadata)
    monkeypatch.setattr("app.jobs.adapters.resolve_youtube_match", fake_youtube_match)
    monkeypatch.setattr("app.jobs.adapters.download_youtube_audio", fake_download)

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            queue_db_path=Path(tmp) / "queue.sqlite3",
            work_dir=Path(tmp) / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue({"job_id": "missing", "job_type": "bulk_dissect", "source": "New Song"})
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=GCSClient(
                project_id=settings.gcp_project_id,
                bucket_name=settings.gcp_bucket_name,
                cdn_base_url=settings.cdn_base_url,
            ),
            db=DBClient(database_url=""),
            library=MissingLibrary(),
            library_writer=LibraryWriter(
                db=DBClient(database_url=""),
                settings=settings,
                membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            ),
        )

        result = await BulkDissectAdapter().download(job, context)

        assert result.next_stage == JobStage.ANALYZE
        assert result.artifacts["library_precheck"]["source"] == "miss"
        assert result.artifacts["download_timings"]["library_precheck_seconds"] >= 0
        assert result.artifacts["download_timings"]["youtube_match_seconds"] >= 0
        assert result.artifacts["download_timings"]["youtube_download_seconds"] >= 0
        assert result.artifacts["download_timings"]["download_total_seconds"] >= 0
        assert Path(result.artifacts["audio_path"]).exists()


async def test_segment_processing_publishes_chord_before_remaining_stems(monkeypatch):
    events = []

    async def fake_extract_segment(source, target, *, start, duration):
        Path(target).write_bytes(b"segment")

    async def fake_convert_to_mp3(source, target):
        Path(target).write_bytes(Path(source).read_bytes())

    async def fake_extract_stem_segments(full_stem_paths, output_dir, segment):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        chord = output / "other.wav"
        vocal = output / "vocals.wav"
        chord.write_bytes(b"chord")
        vocal.write_bytes(b"vocal")
        return {"vocals": str(vocal), "other": str(chord)}

    class FakeGCS:
        async def upload(self, local_path, gcs_path, *, content_type):
            stem = Path(gcs_path).stem
            events.append(f"upload:{stem}")
            return f"https://cdn.test/{stem}.mp3"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            events.append(f"publish:{','.join(sorted(segment_result['outputs']))}:{status}")
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    monkeypatch.setattr("app.jobs.adapters.AudioOps.extract_segment", fake_extract_segment)
    monkeypatch.setattr("app.jobs.adapters.AudioOps.convert_to_mp3", fake_convert_to_mp3)
    monkeypatch.setattr(BulkDissectAdapter, "_extract_stem_segments", staticmethod(fake_extract_stem_segments))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.wav"
        source.write_bytes(b"audio")
        settings = Settings(
            queue_db_path=root / "queue.sqlite3",
            work_dir=root / "work",
            dry_run_mode=False,
        )
        store = JobStore(settings.queue_db_path)
        job, _ = store.enqueue({"job_id": "publish-order", "job_type": "bulk_dissect", "source": "song"})
        store.complete_stage(
            job.id,
            next_stage=JobStage.PROCESS,
            artifacts={
                "work_dir": str(root / "work" / "jobs" / job.id),
                "audio_path": str(source),
                "full_stem_paths": {"other": "unused", "vocals": "unused"},
            },
        )
        job = store.get(job.id)
        context = JobContext(
            settings=settings,
            store=store,
            models=None,
            gcs=FakeGCS(),
            db=DBClient(database_url=""),
            library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
            library_writer=FakeWriter(),
        )

        result = await BulkDissectAdapter()._process_segment(
            job,
            context,
            {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"},
            output_prefix="segments",
        )

        assert events[:3] == ["upload:other", "publish:other:partial", "upload:vocals"]
        assert result["early_library_publish"]["status"] == "partial"
        assert result["stem_source"] == "all_in_one_full_song_stems"
        assert "stem_segment_extract_seconds" in result["timings"]


async def test_segment_stem_cache_reuses_sliced_full_stem_segments(monkeypatch, tmp_path):
    calls = {"extract": 0}

    async def fake_extract_stem_segments(full_stem_paths, output_dir, segment):
        calls["extract"] += 1
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for stem in full_stem_paths:
            path = output / f"{stem}.mp3"
            path.write_bytes(f"{stem}-segment".encode())
            paths[stem] = str(path)
        return paths

    class FakeGCS:
        async def exists(self, gcs_path):
            return False

        async def upload(self, local_path, gcs_path, *, content_type):
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    monkeypatch.setattr(BulkDissectAdapter, "_extract_stem_segments", staticmethod(fake_extract_stem_segments))

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        segment_stem_cache_enabled=True,
        gcs_segment_upload_cache_enabled=False,
    )
    store = JobStore(settings.queue_db_path)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )
    jobs = []
    for index in range(2):
        job, _ = store.enqueue({"job_id": f"segment-cache-{index}", "job_type": "bulk_dissect", "source": "song"})
        work_dir = tmp_path / "work" / "jobs" / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / "source.wav"
        source.write_bytes(b"audio")
        store.complete_stage(
            job.id,
            next_stage=JobStage.PROCESS,
            artifacts={
                "work_dir": str(work_dir),
                "audio_path": str(source),
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "full_stem_paths": {"other": "unused-other", "vocals": "unused-vocals"},
            },
        )
        jobs.append(store.get(job.id))

    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}
    first = await BulkDissectAdapter()._process_segment(jobs[0], context, segment, output_prefix="segments")
    second = await BulkDissectAdapter()._process_segment(jobs[1], context, segment, output_prefix="segments")

    assert calls["extract"] == 1
    assert first["timings"]["segment_stem_cache_hit"] == 0
    assert second["timings"]["segment_stem_cache_hit"] == 1
    assert second["timings"]["stem_segment_extract_seconds"] == 0.0
    assert second["stem_source"] == "segment_stem_cache"
    assert Path(second["stem_paths"]["other"]).exists()
    assert Path(second["stem_paths"]["vocals"]).exists()


async def test_segment_stem_cache_reuses_partial_stems_and_extracts_only_misses(monkeypatch, tmp_path):
    extracted = []

    async def fake_extract_stem_segments(full_stem_paths, output_dir, segment):
        extracted.append(set(full_stem_paths))
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for stem in full_stem_paths:
            path = output / f"{stem}.mp3"
            path.write_bytes(f"{stem}-segment".encode())
            paths[stem] = str(path)
        return paths

    class FakeGCS:
        async def exists(self, gcs_path):
            return False

        async def upload(self, local_path, gcs_path, *, content_type):
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    monkeypatch.setattr(BulkDissectAdapter, "_extract_stem_segments", staticmethod(fake_extract_stem_segments))

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        segment_stem_cache_enabled=True,
        gcs_segment_upload_cache_enabled=False,
    )
    store = JobStore(settings.queue_db_path)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )
    job, _ = store.enqueue({"job_id": "segment-partial-cache", "job_type": "bulk_dissect", "source": "song"})
    work_dir = tmp_path / "work" / "jobs" / job.id
    work_dir.mkdir(parents=True, exist_ok=True)
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(work_dir),
            "audio_path": str(tmp_path / "source.wav"),
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "full_stem_paths": {
                "other": "unused-other",
                "vocals": "unused-vocals",
                "drums": "unused-drums",
                "bass": "unused-bass",
            },
        },
    )
    job = store.get(job.id)
    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}

    chord = await BulkDissectAdapter()._process_segment(
        job,
        context,
        segment,
        output_prefix="segments",
        stem_group="chord",
    )
    other = await BulkDissectAdapter()._process_segment(
        job,
        context,
        segment,
        output_prefix="segments",
        stem_group="all",
    )

    assert extracted == [{"other"}, {"vocals", "drums", "bass"}]
    assert chord["timings"]["segment_stem_cache_hit"] == 0
    assert other["timings"]["segment_stem_cache_hit"] == 0
    assert other["timings"]["segment_stem_cache_partial_hit"] == 1
    assert other["timings"]["segment_stem_cache_partial_hit_count"] == 1
    assert set(other["stem_paths"]) == {"other", "vocals", "drums", "bass"}
    assert Path(other["stem_paths"]["other"]).exists()


async def test_cached_segment_upload_reuses_existing_gcs_object(monkeypatch, tmp_path):
    from app.jobs import adapters

    converted = False

    async def fake_convert_to_mp3(source, target):
        nonlocal converted
        converted = True
        Path(target).write_bytes(b"converted")

    class FakeGCS:
        def __init__(self):
            self.uploads = []
            self.exists_calls = 0

        async def exists(self, gcs_path):
            self.exists_calls += 1
            return True

        async def upload(self, local_path, gcs_path, *, content_type):
            self.uploads.append(gcs_path)
            return f"https://cdn.test/{gcs_path}"

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        cdn_base_url="https://cdn.test",
        gcs_segment_upload_cache_enabled=True,
    )
    local_stem = tmp_path / "other.mp3"
    local_stem.write_bytes(b"mp3")
    context = JobContext(
        settings=settings,
        store=JobStore(settings.queue_db_path),
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=LibraryWriter(
            db=DBClient(database_url=""),
            settings=settings,
            membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        ),
    )
    monkeypatch.setattr("app.jobs.adapters.AudioOps.convert_to_mp3", fake_convert_to_mp3)
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()

    timings = {}
    url = await BulkDissectAdapter._prepare_and_upload_stem(
        stem="other",
        path=str(local_stem.with_suffix(".wav")),
        job_id="job-1",
        segment_id="seg-1",
        context=context,
        upload_cache_key="youtube-video-seg",
        timings=timings,
    )

    assert url == "https://cdn.test/gpu-ingestion/cache/segment-stems/youtube-video-seg/other.mp3"
    assert context.gcs.uploads == []
    assert converted is False
    assert context.gcs.exists_calls == 1
    assert timings["gcs_segment_upload_cache_lookup_count"] == 1
    assert timings["gcs_segment_upload_cache_hit_count"] == 1
    assert "gcs_segment_upload_count" not in timings

    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()
    second_timings = {}
    second_url = await BulkDissectAdapter._prepare_and_upload_stem(
        stem="other",
        path=str(local_stem.with_suffix(".wav")),
        job_id="job-2",
        segment_id="seg-1",
        context=context,
        upload_cache_key="youtube-video-seg",
        timings=second_timings,
    )

    assert second_url == url
    assert context.gcs.exists_calls == 1
    assert second_timings["gcs_segment_upload_disk_cache_hit_count"] == 1
    assert "gcs_segment_upload_cache_lookup_count" not in second_timings
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()


async def test_uploaded_segment_url_is_cached_after_gcs_miss(tmp_path):
    from app.jobs import adapters

    class FakeGCS:
        def __init__(self):
            self.exists_calls = 0
            self.uploads = []

        async def exists(self, gcs_path):
            self.exists_calls += 1
            return False

        async def upload(self, local_path, gcs_path, *, content_type):
            self.uploads.append(gcs_path)
            return f"https://cdn.test/{gcs_path}"

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        cdn_base_url="https://cdn.test",
        gcs_segment_upload_cache_enabled=True,
    )
    local_stem = tmp_path / "other.mp3"
    local_stem.write_bytes(b"mp3")
    context = JobContext(
        settings=settings,
        store=JobStore(settings.queue_db_path),
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=LibraryWriter(
            db=DBClient(database_url=""),
            settings=settings,
            membership=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        ),
    )
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()

    first_timings = {}
    first = await BulkDissectAdapter._prepare_and_upload_stem(
        stem="other",
        path=str(local_stem),
        job_id="job-1",
        segment_id="seg-1",
        context=context,
        upload_cache_key="youtube-new-seg",
        timings=first_timings,
    )
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()
    second_timings = {}
    second = await BulkDissectAdapter._prepare_and_upload_stem(
        stem="other",
        path=str(local_stem),
        job_id="job-2",
        segment_id="seg-1",
        context=context,
        upload_cache_key="youtube-new-seg",
        timings=second_timings,
    )

    assert second == first
    assert context.gcs.exists_calls == 1
    assert context.gcs.uploads == ["gpu-ingestion/cache/segment-stems/youtube-new-seg/other.mp3"]
    assert first_timings["gcs_segment_upload_cache_miss_count"] == 1
    assert first_timings["gcs_segment_upload_count"] == 1
    assert second_timings["gcs_segment_upload_disk_cache_hit_count"] == 1

    third_timings = {}
    third = await BulkDissectAdapter._prepare_and_upload_stem(
        stem="other",
        path=str(local_stem),
        job_id="job-3",
        segment_id="seg-1",
        context=context,
        upload_cache_key="youtube-new-seg",
        timings=third_timings,
    )

    assert third == first
    assert context.gcs.exists_calls == 1
    assert third_timings["gcs_segment_upload_memory_cache_hit_count"] == 1
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()


async def test_full_stem_segment_extraction_runs_stems_concurrently(monkeypatch, tmp_path):
    active = 0
    max_active = 0

    async def fake_extract_segment(source, target, *, start, duration):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        Path(target).write_bytes(b"segment")
        active -= 1

    monkeypatch.setattr("app.jobs.adapters.AudioOps.extract_segment", fake_extract_segment)

    stems = await BulkDissectAdapter._extract_stem_segments(
        {
            "other": str(tmp_path / "other.wav"),
            "vocals": str(tmp_path / "vocals.wav"),
            "drums": str(tmp_path / "drums.wav"),
        },
        tmp_path / "segments",
        {"id": "seg-1", "start": 0.0, "end": 8.0},
    )

    assert set(stems) == {"other", "vocals", "drums"}
    assert max_active > 1


async def test_other_stem_uploads_run_concurrently(monkeypatch, tmp_path):
    active = 0
    max_active = 0
    uploaded_paths = []

    class FakeGCS:
        async def upload(self, local_path, gcs_path, *, content_type):
            nonlocal active, max_active
            uploaded_paths.append(gcs_path)
            if gcs_path.endswith(".mp3"):
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    paths = {}
    for stem in ("vocals", "drums", "bass"):
        path = tmp_path / f"{stem}.mp3"
        path.write_bytes(b"mp3")
        paths[stem] = str(path)

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "parallel-upload", "job_type": "bulk_dissect", "source": "song"})
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(tmp_path / "work" / "jobs" / job.id),
            "audio_path": str(tmp_path / "source.wav"),
            "stem_paths": paths,
        },
    )
    job = store.get(job.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_segment(
        job,
        context,
        {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"},
        output_prefix="segments",
        stem_group="other",
    )

    assert set(result["outputs"]) == {"vocals", "drums", "bass"}
    assert max_active > 1
    assert result["manifest_url"] is None
    assert not any(path.endswith("manifest.json") for path in uploaded_paths)


async def test_other_stem_uploads_use_memory_cache_without_gcs_calls(tmp_path):
    from app.jobs import adapters

    class FakeGCS:
        def __init__(self):
            self.exists_calls = 0
            self.uploads = []

        async def exists(self, gcs_path):
            self.exists_calls += 1
            return True

        async def upload(self, local_path, gcs_path, *, content_type):
            self.uploads.append(gcs_path)
            return f"https://cdn.test/{gcs_path}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        cdn_base_url="https://cdn.test",
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "hot-upload", "job_type": "bulk_dissect", "source": "song"})
    work_dir = tmp_path / "work" / "jobs" / job.id
    work_dir.mkdir(parents=True, exist_ok=True)
    stem_paths = {}
    for stem in ("vocals", "drums", "bass"):
        path = tmp_path / f"{stem}.mp3"
        path.write_bytes(b"mp3")
        stem_paths[stem] = str(path)
    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    cache_key = adapters._segment_stem_cache_key({"youtube_url": youtube_url}, segment)
    assert cache_key
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()
    for stem in stem_paths:
        adapters._remember_gcs_upload_url(
            f"gpu-ingestion/cache/segment-stems/{cache_key}/{stem}.mp3",
            f"https://cdn.test/gpu-ingestion/cache/segment-stems/{cache_key}/{stem}.mp3",
            max_entries=100,
        )
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(work_dir),
            "audio_path": str(tmp_path / "source.wav"),
            "youtube_url": youtube_url,
            "segment_id": segment["id"],
            "stem_paths": stem_paths,
        },
    )
    job = store.get(job.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_segment(
        job,
        context,
        segment,
        output_prefix="segments",
        stem_group="other",
    )

    assert context.gcs.exists_calls == 0
    assert context.gcs.uploads == []
    assert result["timings"]["gcs_segment_upload_memory_cache_hit_count"] == 3
    assert "gcs_segment_upload_cache_lookup_count" not in result["timings"]
    assert "gcs_segment_upload_count" not in result["timings"]
    assert result["outputs"]["vocals"].endswith(f"{cache_key}/vocals.mp3")
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()


async def test_hot_full_stem_urls_skip_local_segment_restore(monkeypatch, tmp_path):
    from app.jobs import adapters

    async def fail_extract_stem_segments(full_stem_paths, output_dir, segment):
        raise AssertionError("should not slice full stems when URLs are hot")

    class FakeGCS:
        async def exists(self, gcs_path):
            raise AssertionError("should not call GCS exists when URLs are hot")

        async def upload(self, local_path, gcs_path, *, content_type):
            raise AssertionError("should not upload when URLs are hot")

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    monkeypatch.setattr(BulkDissectAdapter, "_extract_stem_segments", staticmethod(fail_extract_stem_segments))

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        cdn_base_url="https://cdn.test",
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "hot-full-stems", "job_type": "bulk_dissect", "source": "song"})
    work_dir = tmp_path / "work" / "jobs" / job.id
    work_dir.mkdir(parents=True, exist_ok=True)
    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    cache_key = adapters._segment_stem_cache_key({"youtube_url": youtube_url}, segment)
    assert cache_key
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()
    for stem in ("vocals", "drums", "bass"):
        adapters._remember_gcs_upload_url(
            f"gpu-ingestion/cache/segment-stems/{cache_key}/{stem}.mp3",
            f"https://cdn.test/gpu-ingestion/cache/segment-stems/{cache_key}/{stem}.mp3",
            max_entries=100,
        )
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(work_dir),
            "audio_path": str(tmp_path / "source.wav"),
            "youtube_url": youtube_url,
            "full_stem_paths": {
                "other": "unused-other.wav",
                "vocals": "unused-vocals.wav",
                "drums": "unused-drums.wav",
                "bass": "unused-bass.wav",
            },
        },
    )
    job = store.get(job.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_segment(
        job,
        context,
        segment,
        output_prefix="segments",
        stem_group="other",
    )

    assert result["stem_source"] == "gcs_upload_url_cache"
    assert result["stem_paths"] == {}
    assert result["timings"]["segment_output_url_cache_hit"] == 1
    assert result["timings"]["segment_output_url_cache_stem_count"] == 3
    assert result["timings"]["gcs_segment_upload_memory_cache_hit_count"] == 3
    assert set(result["outputs"]) == {"vocals", "drums", "bass"}
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()


async def test_hot_chord_url_shortcut_still_publishes_partial(tmp_path):
    from app.jobs import adapters

    events = []

    class FakeGCS:
        async def exists(self, gcs_path):
            raise AssertionError("should not call GCS exists when URLs are hot")

        async def upload(self, local_path, gcs_path, *, content_type):
            raise AssertionError("should not upload when URLs are hot")

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            events.append((status, dict(segment_result["outputs"])))
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        cdn_base_url="https://cdn.test",
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "hot-chord", "job_type": "bulk_dissect", "source": "song"})
    work_dir = tmp_path / "work" / "jobs" / job.id
    work_dir.mkdir(parents=True, exist_ok=True)
    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    cache_key = adapters._segment_stem_cache_key({"youtube_url": youtube_url}, segment)
    assert cache_key
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()
    adapters._remember_gcs_upload_url(
        f"gpu-ingestion/cache/segment-stems/{cache_key}/other.mp3",
        f"https://cdn.test/gpu-ingestion/cache/segment-stems/{cache_key}/other.mp3",
        max_entries=100,
    )
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(work_dir),
            "audio_path": str(tmp_path / "source.wav"),
            "youtube_url": youtube_url,
            "full_stem_paths": {
                "other": "unused-other.wav",
                "vocals": "unused-vocals.wav",
            },
        },
    )
    job = store.get(job.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_segment(
        job,
        context,
        segment,
        output_prefix="segments",
        stem_group="chord",
    )

    assert result["stem_source"] == "gcs_upload_url_cache"
    assert result["early_library_publish"]["status"] == "partial"
    assert events == [("partial", result["outputs"])]
    assert result["outputs"]["other"].endswith(f"{cache_key}/other.mp3")
    adapters._GCS_SEGMENT_UPLOAD_URL_CACHE.clear()


async def test_pre_enqueued_chord_child_does_not_enqueue_duplicate_other(tmp_path):
    class FakeGCS:
        async def upload(self, local_path, gcs_path, *, content_type):
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
    )
    store = JobStore(settings.queue_db_path)
    root, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}
    stem_path = tmp_path / "other.mp3"
    stem_path.write_bytes(b"mp3")
    chord = store.enqueue_process_child(
        parent_job=root,
        child_id="root:bulk:seg-1:chord",
        job_type=JobType.BULK_DISSECT,
        payload={"source": "song", "root_job_id": root.id, "process_mode": "segment_chord", "segment_id": "seg-1"},
        artifacts={
            "work_dir": str(tmp_path / "work" / "jobs" / root.id),
            "audio_path": str(tmp_path / "source.wav"),
            "process_mode": "segment_chord",
            "process_group": "bulk",
            "root_job_id": root.id,
            "segment_id": "seg-1",
            "segment": segment,
            "stem_paths": {"other": str(stem_path)},
            "other_stems_pre_enqueued": True,
            "other_stems_job_id": "root:bulk:seg-1:other",
            "requires_gpu": False,
        },
        priority=300,
    )
    store.enqueue_process_child(
        parent_job=root,
        child_id="root:bulk:seg-1:other",
        job_type=JobType.BULK_DISSECT,
        payload={"source": "song", "root_job_id": root.id, "process_mode": "segment_other", "segment_id": "seg-1"},
        artifacts={
            "work_dir": str(tmp_path / "work" / "jobs" / root.id),
            "audio_path": str(tmp_path / "source.wav"),
            "process_mode": "segment_other",
            "process_group": "bulk",
            "root_job_id": root.id,
            "segment_id": "seg-1",
            "segment": segment,
            "stem_paths": {},
            "requires_gpu": False,
        },
        priority=100,
    )
    store.claim_next([JobStage.PROCESS], "process-worker")
    chord = store.get(chord.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_fanout_job(chord, context)

    assert result.artifacts["other_stems_job_id"] == "root:bulk:seg-1:other"
    assert result.artifacts["final_outputs"]["other_stems_job_id"] == "root:bulk:seg-1:other"
    assert store.child_summary(root.id)["total"] == 2


async def test_segment_manifest_upload_is_optional(monkeypatch, tmp_path):
    uploaded_paths = []

    class FakeGCS:
        async def upload(self, local_path, gcs_path, *, content_type):
            uploaded_paths.append(gcs_path)
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    stem_path = tmp_path / "other.mp3"
    stem_path.write_bytes(b"mp3")
    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
        segment_manifest_upload_enabled=True,
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "manifest-enabled", "job_type": "bulk_dissect", "source": "song"})
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(tmp_path / "work" / "jobs" / job.id),
            "audio_path": str(tmp_path / "source.wav"),
            "stem_paths": {"other": str(stem_path)},
        },
    )
    job = store.get(job.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_segment(
        job,
        context,
        {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"},
        output_prefix="segments",
        stem_group="chord",
    )

    assert result["manifest_url"] == "https://cdn.test/manifest.json"
    assert any(path.endswith("manifest.json") for path in uploaded_paths)


async def test_chord_job_slices_only_chord_and_child_uses_full_stems(monkeypatch, tmp_path):
    extracted_keys = []

    async def fake_extract_stem_segments(full_stem_paths, output_dir, segment):
        extracted_keys.append(set(full_stem_paths))
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = {}
        for stem in full_stem_paths:
            path = output / f"{stem}.mp3"
            path.write_bytes(b"mp3")
            result[stem] = str(path)
        return result

    class FakeGCS:
        async def upload(self, local_path, gcs_path, *, content_type):
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

    monkeypatch.setattr(BulkDissectAdapter, "_extract_stem_segments", staticmethod(fake_extract_stem_segments))

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
    )
    store = JobStore(settings.queue_db_path)
    job, _ = store.enqueue({"job_id": "chord-only", "job_type": "bulk_dissect", "source": "song"})
    store.complete_stage(
        job.id,
        next_stage=JobStage.PROCESS,
        artifacts={
            "work_dir": str(tmp_path / "work" / "jobs" / job.id),
            "audio_path": str(tmp_path / "source.wav"),
            "full_stem_paths": {
                "other": str(tmp_path / "other.wav"),
                "vocals": str(tmp_path / "vocals.wav"),
                "drums": str(tmp_path / "drums.wav"),
            },
            "process_group": "bulk",
            "root_job_id": job.id,
        },
    )
    job = store.get(job.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    segment = {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"}
    adapter = BulkDissectAdapter()
    result = await adapter._process_segment(
        job,
        context,
        segment,
        output_prefix="segments",
        stem_group="chord",
    )
    other_job = adapter._enqueue_other_stems_job(job, context, segment, result)

    assert extracted_keys == [{"other"}]
    assert set(result["outputs"]) == {"other"}
    assert other_job is not None
    assert other_job.artifacts["stem_paths"] == {}
    assert set(other_job.artifacts["full_stem_paths"]) == {"other", "vocals", "drums"}


async def test_last_other_fanout_child_signals_parent_reconcile(monkeypatch, tmp_path):
    mark_complete_calls = 0

    class FakeGCS:
        async def upload(self, local_path, gcs_path, *, content_type):
            return f"https://cdn.test/{Path(gcs_path).name}"

    class FakeWriter:
        async def publish_segment(self, *, job, segment, segment_result, status):
            return LibraryPublishResult(enabled=True, song_id="song-1", status=status)

        async def mark_complete(self, *, job):
            nonlocal mark_complete_calls
            mark_complete_calls += 1
            return LibraryPublishResult(enabled=True, song_id="song-1", status="complete")

    for stem in ("vocals", "drums", "bass"):
        (tmp_path / f"{stem}.mp3").write_bytes(b"mp3")

    settings = Settings(
        queue_db_path=tmp_path / "queue.sqlite3",
        work_dir=tmp_path / "work",
        dry_run_mode=False,
    )
    store = JobStore(settings.queue_db_path)
    root, _ = store.enqueue({"job_id": "root", "job_type": "bulk_dissect", "source": "song"})
    child = store.enqueue_process_child(
        parent_job=root,
        child_id="root:bulk:seg-1:other",
        job_type=JobType.BULK_DISSECT,
        payload={"source": "song", "root_job_id": root.id},
        artifacts={
            "work_dir": str(tmp_path / "work" / "jobs" / root.id),
            "audio_path": str(tmp_path / "source.wav"),
            "process_mode": "segment_other",
            "process_group": "bulk",
            "root_job_id": root.id,
            "segment_id": "seg-1",
            "segment": {"id": "seg-1", "start": 0.0, "end": 8.0, "label": "chorus"},
            "stem_paths": {
                "vocals": str(tmp_path / "vocals.mp3"),
                "drums": str(tmp_path / "drums.mp3"),
                "bass": str(tmp_path / "bass.mp3"),
            },
            "requires_gpu": False,
        },
        priority=100,
    )
    store.claim_next([JobStage.PROCESS], "process-worker")
    child = store.get(child.id)
    context = JobContext(
        settings=settings,
        store=store,
        models=None,
        gcs=FakeGCS(),
        db=DBClient(database_url=""),
        library=LibraryMembershipChecker(db=DBClient(database_url=""), settings=settings),
        library_writer=FakeWriter(),
    )

    result = await BulkDissectAdapter()._process_fanout_job(child, context)

    assert result.artifacts["fanout_maybe_complete"] is True
    assert result.artifacts["library_complete"]["status"] == "complete"
    assert mark_complete_calls == 1
