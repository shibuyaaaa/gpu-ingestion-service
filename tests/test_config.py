from app.config import Settings, _bytes_env


def test_default_worker_counts_match_l4_cpu_split(monkeypatch):
    for name in (
        "DOWNLOAD_WORKERS",
        "PREP_WORKERS",
        "DOWNLOAD_BATCH_SIZE",
        "PREP_BATCH_SIZE",
        "PROCESS_WORKERS",
        "POSTPROCESS_WORKERS",
        "PROCESS_BATCH_SIZE",
        "POSTPROCESS_BATCH_SIZE",
        "FFMPEG_CONCURRENCY",
        "WORKER_POLL_SECONDS",
        "GPU_JOB_SAMPLE_INTERVAL_SECONDS",
        "SOURCE_AUDIO_CACHE_MAX_BYTES",
        "ANALYSIS_CACHE_MAX_BYTES",
        "SEGMENT_STEM_CACHE_MAX_BYTES",
        "GCS_SEGMENT_UPLOAD_URL_CACHE_MAX_ENTRIES",
        "GCS_SEGMENT_UPLOAD_DISK_CACHE_ENABLED",
        "GCS_SEGMENT_UPLOAD_DISK_CACHE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.download_workers == 2
    assert settings.download_batch_size == 1
    assert settings.process_workers == 6
    assert settings.process_batch_size == 1
    assert settings.ffmpeg_concurrency == 4
    assert settings.worker_poll_seconds == 0.10
    assert settings.gpu_job_sample_interval_seconds == 0.5
    assert settings.source_audio_cache_max_bytes == 8 * 1024**3
    assert settings.analysis_cache_max_bytes == 2 * 1024**3
    assert settings.segment_stem_cache_max_bytes == 8 * 1024**3
    assert settings.gcs_segment_upload_cache_enabled is True
    assert settings.gcs_segment_upload_url_cache_max_entries == 10000
    assert settings.gcs_segment_upload_disk_cache_enabled is True
    assert settings.gcs_segment_upload_disk_cache_path == ""


def test_cache_byte_env_accepts_human_readable_units(monkeypatch):
    monkeypatch.setenv("SOURCE_AUDIO_CACHE_MAX_BYTES", "1.5gb")
    monkeypatch.setenv("ANALYSIS_CACHE_MAX_BYTES", "512mb")
    monkeypatch.setenv("SEGMENT_STEM_CACHE_MAX_BYTES", "2g")

    assert _bytes_env("SOURCE_AUDIO_CACHE_MAX_BYTES", 0) == int(1.5 * 1024**3)
    assert _bytes_env("ANALYSIS_CACHE_MAX_BYTES", 0) == 512 * 1024**2
    assert _bytes_env("SEGMENT_STEM_CACHE_MAX_BYTES", 0) == 2 * 1024**3
