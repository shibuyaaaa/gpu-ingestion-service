import os
from pathlib import Path

from app.cache_status import local_cache_status
from app.jobs import adapters


def test_local_cache_status_counts_source_and_analysis_entries(tmp_path: Path):
    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    (source_cache / "a.wav").write_bytes(b"a" * 10)
    (source_cache / "ignore.tmp").write_bytes(b"ignored")

    analysis_entry = tmp_path / "analysis-cache" / "youtube-abc"
    stems_dir = analysis_entry / "stems"
    stems_dir.mkdir(parents=True)
    (analysis_entry / "metadata.json").write_text("{}", encoding="utf-8")
    (analysis_entry / "analyzer_result.json").write_text("{}", encoding="utf-8")
    for stem in ("bass", "drums", "other", "vocals"):
        (stems_dir / f"{stem}.wav").write_bytes(b"s" * 5)

    status = local_cache_status(
        work_dir=tmp_path,
        source_audio_enabled=True,
        source_audio_max_entries=100,
        source_audio_max_bytes=1024,
        analysis_enabled=True,
        analysis_max_entries=4,
        analysis_max_bytes=2048,
        segment_stem_enabled=True,
        segment_stem_max_entries=1000,
        segment_stem_max_bytes=4096,
        lock_status={"source_audio_locks": 1},
    )

    assert status["source_audio"]["entries"] == 1
    assert status["source_audio"]["bytes"] == 10
    assert status["source_audio"]["max_bytes"] == 1024
    assert status["analysis"]["entries"] == 1
    assert status["analysis"]["complete_entries"] == 1
    assert status["analysis"]["bytes"] >= 20
    assert status["analysis"]["max_bytes"] == 2048
    assert status["segment_stem"]["max_bytes"] == 4096
    assert status["total_bytes"] == (
        status["source_audio"]["bytes"] + status["analysis"]["bytes"] + status["segment_stem"]["bytes"]
    )
    assert status["locks"] == {"source_audio_locks": 1}


async def test_cache_lock_maps_prune_unlocked_entries():
    adapters._SOURCE_AUDIO_CACHE_LOCKS.clear()
    try:
        for index in range(adapters._MAX_CACHE_LOCKS + 5):
            adapters._source_audio_cache_lock(f"video-{index}")

        assert len(adapters._SOURCE_AUDIO_CACHE_LOCKS) <= adapters._MAX_CACHE_LOCKS
        assert len(adapters._SOURCE_AUDIO_CACHE_LOCKS) <= (adapters._MAX_CACHE_LOCKS // 2) + 5
    finally:
        adapters._SOURCE_AUDIO_CACHE_LOCKS.clear()


async def test_cache_lock_pruning_keeps_locked_entries():
    adapters._ANALYSIS_CACHE_LOCKS.clear()
    try:
        locked = adapters._analysis_cache_lock("locked")
        await locked.acquire()
        try:
            for index in range(adapters._MAX_CACHE_LOCKS + 5):
                adapters._analysis_cache_lock(f"analysis-{index}")
        finally:
            locked.release()

        locked_keys = [key for key in adapters._ANALYSIS_CACHE_LOCKS if key[1] == "locked"]
        assert locked_keys
        assert len(adapters._ANALYSIS_CACHE_LOCKS) <= adapters._MAX_CACHE_LOCKS
    finally:
        adapters._ANALYSIS_CACHE_LOCKS.clear()


async def test_source_audio_cache_prunes_by_byte_limit(tmp_path: Path):
    cache_dir = tmp_path / "source-cache"
    cache_dir.mkdir()
    old = cache_dir / "old.wav"
    new = cache_dir / "new.wav"
    old.write_bytes(b"o" * 10)
    new.write_bytes(b"n" * 10)
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    await adapters._prune_source_audio_cache(cache_dir, max_entries=10, max_bytes=10)

    assert not old.exists()
    assert new.exists()


async def test_analysis_cache_prunes_by_byte_limit(tmp_path: Path):
    cache_root = tmp_path / "analysis-cache"
    old = cache_root / "old"
    new = cache_root / "new"
    _write_analysis_entry(old, b"o" * 10)
    _write_analysis_entry(new, b"n" * 10)
    os.utime(old / "metadata.json", (1, 1))
    os.utime(new / "metadata.json", (2, 2))

    await adapters._prune_analysis_cache(cache_root, max_entries=10, max_bytes=80)

    assert not old.exists()
    assert new.exists()


def _write_analysis_entry(path: Path, payload: bytes) -> None:
    stems = path / "stems"
    stems.mkdir(parents=True)
    (path / "metadata.json").write_text("{}", encoding="utf-8")
    (path / "analyzer_result.json").write_bytes(payload)
    for stem in ("bass", "drums", "other", "vocals"):
        (stems / f"{stem}.wav").write_bytes(payload)
