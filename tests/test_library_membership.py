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
        assert result.artifacts["download_timings"]["source_metadata_seconds"] >= 0
        assert result.artifacts["download_timings"]["library_precheck_seconds"] >= 0
        assert result.artifacts["download_timings"]["download_total_seconds"] >= 0


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
