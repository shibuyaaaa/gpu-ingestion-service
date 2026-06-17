#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi

distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo mkdir -p /var/lib/gpu-ingestion/tmp
sudo chmod 0775 /var/lib/gpu-ingestion

if [ ! -f /etc/gpu-ingestion.env ]; then
  sudo tee /etc/gpu-ingestion.env >/dev/null <<'ENV'
SERVICE_NAME=gpu-ingestion-service
QUEUE_DB_PATH=/var/lib/gpu-ingestion/queue.sqlite3
WORK_DIR=/var/lib/gpu-ingestion/tmp
START_WORKERS=true
DRY_RUN_MODE=false
MODEL_BACKEND=local
GPU_DEVICE=cuda:0
HTDEMUCS_MODEL=htdemucs
ALL_IN_ONE_MODEL=harmonix-fold0
CUDA_RUNTIME_TUNING=true
CUDA_ALLOW_TF32=true
CUDA_CUDNN_BENCHMARK=true
CUDA_MATMUL_PRECISION=high
CUDA_EMPTY_CACHE_AFTER_JOB=false
PINNED_AUDIO_STAGING=true
PINNED_AUDIO_SECONDS=600
PINNED_AUDIO_CHANNELS=2
PINNED_AUDIO_SAMPLE_RATE=44100
PINNED_AUDIO_SLOTS=2
CUDA_GRAPHS_ENABLED=false
CUDA_GRAPH_AUDIO_SECONDS=30
MAX_TOTAL_QUEUE_DEPTH=200
DOWNLOAD_WORKERS=4
DOWNLOAD_BATCH_SIZE=2
PROCESS_WORKERS=1
PROCESS_BATCH_SIZE=1
ANALYZE_BATCH_SIZE=1
WORKER_POLL_SECONDS=0.25
JOB_LEASE_TIMEOUT_SECONDS=1800
GCP_PROJECT_ID=imposing-kayak-422917-b0
GCP_BUCKET_NAME=shibuya-assets
CDN_BASE_URL=https://cdn.shibuyaaa.com
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
ENV
fi

sudo cp deploy/gpu-ingestion.service /etc/systemd/system/gpu-ingestion.service
sudo systemctl daemon-reload
echo "Edit /etc/gpu-ingestion.env, then run: sudo systemctl enable --now gpu-ingestion"
