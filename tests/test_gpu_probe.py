from app.models.gpu import GPUProbe, GPUState, GPUUsageSampler


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


def test_gpu_usage_sampler_summarizes_uncached_samples(monkeypatch):
    samples = [
        GPUState(
            available=True,
            name="gpu",
            utilization_pct=10.0,
            memory_total_mb=100,
            memory_used_mb=20,
            memory_free_mb=80,
        ),
        GPUState(
            available=True,
            name="gpu",
            utilization_pct=70.0,
            memory_total_mb=100,
            memory_used_mb=50,
            memory_free_mb=50,
        ),
    ]
    calls = 0

    class FakeProbe(GPUProbe):
        def snapshot_uncached(self):
            nonlocal calls
            sample = samples[min(calls, len(samples) - 1)]
            calls += 1
            return sample

    times = iter([100.0, 101.0, 101.0])
    monkeypatch.setattr("app.models.gpu.time.monotonic", lambda: next(times))

    with GPUUsageSampler(FakeProbe(cache_seconds=99.0), interval_seconds=0.0) as sampler:
        pass

    summary = sampler.summary()

    assert calls == 2
    assert summary["gpu_sample_enabled"] is False
    assert summary["gpu_sample_count"] == 2
    assert summary["gpu_sample_available_count"] == 2
    assert summary["gpu_sample_elapsed_seconds"] == 1.0
    assert summary["gpu_utilization_avg_pct"] == 40.0
    assert summary["gpu_utilization_max_pct"] == 70.0
    assert summary["gpu_memory_used_avg_mb"] == 35.0
    assert summary["gpu_memory_used_max_mb"] == 50
    assert summary["gpu_memory_free_min_mb"] == 50
