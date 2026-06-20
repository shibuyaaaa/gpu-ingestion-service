#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-gpu-ingestion-service:local}"
SERVICE_NAME="${SERVICE_NAME:-gpu-ingestion.service}"
CRAWLER_SERVICE_NAME="${CRAWLER_SERVICE_NAME:-gpu-ingestion-crawler.service}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8080/health}"
DOCKER_IMAGE_PRUNE_ARGS="${DOCKER_IMAGE_PRUNE_ARGS:--a -f}"

cd "$(dirname "$0")/.."

git pull --ff-only
sudo docker build -t "${IMAGE}" -f deploy/Dockerfile .
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl restart "${CRAWLER_SERVICE_NAME}"

python3 - <<PY
import json
import time
import urllib.request

url = "${HEALTH_URL}"
last_error = None
for _ in range(30):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") == "healthy":
            print(json.dumps({"health": payload.get("status")}, sort_keys=True))
            break
        last_error = f"unexpected health status: {payload.get('status')}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(2)
else:
    raise SystemExit(f"service did not become healthy: {last_error}")
PY

sudo docker image prune ${DOCKER_IMAGE_PRUNE_ARGS}
sudo docker builder prune -f --filter until=24h >/dev/null || true
sudo docker system df
