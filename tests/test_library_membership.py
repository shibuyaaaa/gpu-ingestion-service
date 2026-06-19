import asyncio
import tempfile
import uuid
from pathlib import Path

from app.config import Settings
from app.jobs.adapters import BulkDissectAdapter
from app.jobs.context import JobContext
from app.library_membership import LibraryLookupResult, LibraryMembershipChecker, LibrarySong
from app.library_writer import LibraryPublishResult, LibraryWriter
from app.legacy.db import DBClient
from app.legacy.utils import GCSClient
from app.queue import JobStage, JobStore


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
