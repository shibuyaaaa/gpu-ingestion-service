from app.config import Settings


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
        "WORKER_POLL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.download_workers == 2
    assert settings.download_batch_size == 1
    assert settings.process_workers == 4
    assert settings.process_batch_size == 1
    assert settings.worker_poll_seconds == 0.10
