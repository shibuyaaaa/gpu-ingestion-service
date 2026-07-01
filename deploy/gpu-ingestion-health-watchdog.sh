#!/usr/bin/env bash
set -euo pipefail

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8080/health}"
SERVICE_NAME="${SERVICE_NAME:-gpu-ingestion.service}"
CRAWLER_SERVICE_NAME="${CRAWLER_SERVICE_NAME:-gpu-ingestion-crawler.service}"
CONTAINER_NAME="${CONTAINER_NAME:-gpu-ingestion-service}"
MAX_ATTEMPTS="${GPU_INGESTION_HEALTH_ATTEMPTS:-3}"
SLEEP_SECONDS="${GPU_INGESTION_HEALTH_RETRY_SECONDS:-5}"

healthy=false
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  if curl -fsS --max-time 10 "${HEALTH_URL}" >/dev/null; then
    healthy=true
    break
  fi
  if [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; then
    sleep "${SLEEP_SECONDS}"
  fi
done

zombie_ytdlp=false
if sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if sudo docker exec "${CONTAINER_NAME}" sh -lc "ps -eo stat,cmd | awk '\$1 ~ /^Z/ && /yt-dlp/ { found = 1 } END { exit found ? 0 : 1 }'"; then
    zombie_ytdlp=true
  fi
fi

if [ "${healthy}" = true ] && [ "${zombie_ytdlp}" = false ]; then
  exit 0
fi

echo "gpu ingestion unhealthy: health=${healthy} zombie_ytdlp=${zombie_ytdlp}; restarting ${SERVICE_NAME} and ${CRAWLER_SERVICE_NAME}" >&2
sudo systemctl restart "${SERVICE_NAME}"
sleep 5
sudo systemctl restart "${CRAWLER_SERVICE_NAME}"
