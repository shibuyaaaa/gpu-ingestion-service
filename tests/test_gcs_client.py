from app.legacy.utils import GCSClient


def test_gcs_exists_uses_storage_blob(monkeypatch):
    calls = []

    class FakeBlob:
        def exists(self):
            calls.append("exists")
            return True

    class FakeBucket:
        def blob(self, path):
            assert path == "path/object.mp3"
            return FakeBlob()

    client = GCSClient(project_id="project", bucket_name="bucket", cdn_base_url="https://cdn.test")
    monkeypatch.setattr(client, "_client", type("FakeClient", (), {"bucket": lambda self, name: FakeBucket()})())

    assert client._exists_sync("path/object.mp3", None) is True
    assert calls == ["exists"]
