import base64
import json

import pytest

from app.legacy.utils import parse_pubsub_envelope
from app.jobs import UnsupportedJobType, build_default_registry
from app.server import _validate_supported_job


def test_parse_pubsub_envelope():
    payload = {"job_id": "j1", "job_type": "bulk_dissect", "source": "song"}
    envelope = {
        "message": {
            "messageId": "m1",
            "data": base64.b64encode(json.dumps(payload).encode()).decode(),
        }
    }

    parsed, message_id = parse_pubsub_envelope(envelope)

    assert parsed == payload
    assert message_id == "m1"


def test_parse_pubsub_rejects_missing_message():
    with pytest.raises(ValueError):
        parse_pubsub_envelope({})


def test_registry_rejects_unknown_job_type():
    registry = build_default_registry()

    with pytest.raises(UnsupportedJobType) as exc:
        registry.get("not_real")

    assert "not_real" in str(exc.value)


def test_registry_only_supports_two_dissect_job_types():
    registry = build_default_registry()

    assert registry.supports("quick_dissect")
    assert registry.supports("bulk_dissect")
    assert not registry.supports("audio_dissect")
    assert not registry.supports("ugc_stem")


def test_job_validation_requires_source():
    with pytest.raises(ValueError) as exc:
        _validate_supported_job({"job_id": "missing-source", "job_type": "quick_dissect"})

    assert "source" in str(exc.value)


def test_job_validation_accepts_youtube_url():
    _validate_supported_job(
        {
            "job_id": "yt-source",
            "job_type": "quick_dissect",
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
    )
