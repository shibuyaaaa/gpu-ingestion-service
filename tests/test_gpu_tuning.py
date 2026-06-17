from app.config import Settings
from app.models import ModelRuntimeBundle
from app.models.tuning import PinnedMemoryPolicy


def test_pinned_memory_policy_default_budget_is_reasonable_for_one_l4():
    settings = Settings(dry_run_mode=True)

    policy = PinnedMemoryPolicy.from_settings(settings)

    assert policy.enabled is True
    assert policy.max_bytes == int(600 * 44100 * 2 * 4 * 2)
    assert 400 <= policy.to_dict()["max_mib"] <= 405


async def test_runtime_status_exposes_cuda_policy_in_dry_run():
    settings = Settings(dry_run_mode=True, cuda_graphs_enabled=False)
    runtimes = ModelRuntimeBundle.from_settings(settings)

    await runtimes.warmup()
    status = runtimes.status()

    assert status["models"]["htdemucs"]["resident_policy"] == "load-once-process-resident"
    assert status["pinned_audio"]["policy"]["enabled"] is True
    assert status["cuda_graph_policy"]["status"] == "disabled"
    assert status["torch_runtime"]["enabled"] in {True, False}

