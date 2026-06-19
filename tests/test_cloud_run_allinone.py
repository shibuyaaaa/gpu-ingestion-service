from pathlib import Path

from app.models.cloud_run_allinone import CloudRunAllInOneRuntime


async def test_cloud_run_runtime_transcodes_oversized_upload(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(b"x" * 128)

    runtime = CloudRunAllInOneRuntime(
        url="https://example.invalid",
        model_name="harmonix-fold0",
        audio_separator_model="Kim_Vocal_2.onnx",
        max_upload_bytes=64,
        upload_bitrate="320k",
    )

    calls = []

    async def fake_transcode(input_path: Path, target_path: Path) -> None:
        calls.append((input_path, target_path))
        target_path.write_bytes(b"mp3")

    monkeypatch.setattr(runtime, "_transcode_for_upload", fake_transcode)

    upload_path, cleanup_path = await runtime._upload_path(source)

    assert upload_path == tmp_path / "input.cloudrun-upload.mp3"
    assert cleanup_path == upload_path
    assert upload_path.read_bytes() == b"mp3"
    assert calls == [(source, upload_path)]


async def test_cloud_run_runtime_keeps_small_upload_original(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"x" * 32)

    runtime = CloudRunAllInOneRuntime(
        url="https://example.invalid",
        model_name="harmonix-fold0",
        audio_separator_model="Kim_Vocal_2.onnx",
        max_upload_bytes=64,
    )

    upload_path, cleanup_path = await runtime._upload_path(source)

    assert upload_path == source
    assert cleanup_path is None
