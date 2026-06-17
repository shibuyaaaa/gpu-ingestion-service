import base64
import json
from typing import Any


def parse_pubsub_envelope(envelope: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    if not isinstance(envelope, dict) or "message" not in envelope:
        raise ValueError("invalid Pub/Sub envelope")
    message = envelope.get("message") or {}
    data = message.get("data")
    if not data:
        raise ValueError("Pub/Sub message missing data")
    decoded = base64.b64decode(data).decode("utf-8")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError("Pub/Sub payload must be an object")
    return payload, message.get("messageId") or message.get("message_id")

