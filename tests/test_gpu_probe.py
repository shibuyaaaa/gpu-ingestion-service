from app.models.gpu import GPUProbe, GPUState


def test_gpu_probe_reuses_cached_snapshot_within_ttl(monkeypatch):
    calls = 0
    now = 100.0

    class FakeProbe(GPUProbe):
        def _snapshot_uncached(self):
            nonlocal calls
            calls += 1
            return GPUState(available=True, name=f"gpu-{calls}")

    monkeypatch.setattr("app.models.gpu.time.monotonic", lambda: now)
    probe = FakeProbe(cache_seconds=1.0)

    first = probe.snapshot()
    second = probe.snapshot()

    assert first is second
    assert second.name == "gpu-1"
    assert calls == 1


def test_gpu_probe_refreshes_after_ttl(monkeypatch):
    calls = 0
    current_time = {"value": 100.0}

    class FakeProbe(GPUProbe):
        def _snapshot_uncached(self):
            nonlocal calls
            calls += 1
            return GPUState(available=True, name=f"gpu-{calls}")

    monkeypatch.setattr("app.models.gpu.time.monotonic", lambda: current_time["value"])
    probe = FakeProbe(cache_seconds=1.0)

    first = probe.snapshot()
    current_time["value"] = 101.1
    second = probe.snapshot()

    assert first.name == "gpu-1"
    assert second.name == "gpu-2"
    assert calls == 2


def test_gpu_probe_cache_can_be_disabled(monkeypatch):
    calls = 0

    class FakeProbe(GPUProbe):
        def _snapshot_uncached(self):
            nonlocal calls
            calls += 1
            return GPUState(available=True, name=f"gpu-{calls}")

    monkeypatch.setattr("app.models.gpu.time.monotonic", lambda: 100.0)
    probe = FakeProbe(cache_seconds=0.0)

    first = probe.snapshot()
    second = probe.snapshot()

    assert first.name == "gpu-1"
    assert second.name == "gpu-2"
    assert calls == 2
