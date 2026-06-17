FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG ALL_IN_ONE_REPO=https://github.com/ErickLuis00/all-in-one-audio-analyzer-separator.git
ARG ALL_IN_ONE_REF=246fe25487449823521756117af21dd7a520223f

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8
ENV APP_HOME=/srv/gpu-ingestion-service
ENV PYTHONPATH=/srv/gpu-ingestion-service:/opt/all-in-one-audio

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip

WORKDIR /opt
RUN git clone "${ALL_IN_ONE_REPO}" all-in-one-audio \
    && cd all-in-one-audio \
    && git checkout "${ALL_IN_ONE_REF}"

WORKDIR /opt/all-in-one-audio
RUN sed -i '/audio-separator/d' requirements.txt
RUN python3.10 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      torch==2.0.1+cu118 \
      torchaudio==2.0.2+cu118 \
      torchvision==0.15.2+cu118 \
    && python3.10 -m pip install --no-cache-dir -r requirements.txt \
    && python3.10 -m pip install --no-cache-dir --only-binary=:all: --trusted-host shi-labs.com \
      natten -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html

WORKDIR ${APP_HOME}
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
COPY app ./app

ENV QUEUE_DB_PATH=/var/lib/gpu-ingestion/queue.sqlite3
ENV WORK_DIR=/var/lib/gpu-ingestion/tmp
ENV MODEL_BACKEND=local
ENV DRY_RUN_MODE=false
ENV START_WORKERS=true
ENV CUDA_RUNTIME_TUNING=true
ENV CUDA_ALLOW_TF32=true
ENV CUDA_CUDNN_BENCHMARK=true
ENV CUDA_MATMUL_PRECISION=high
ENV CUDA_EMPTY_CACHE_AFTER_JOB=false
ENV PINNED_AUDIO_STAGING=true
ENV PINNED_AUDIO_SECONDS=600
ENV PINNED_AUDIO_CHANNELS=2
ENV PINNED_AUDIO_SAMPLE_RATE=44100
ENV PINNED_AUDIO_SLOTS=2
ENV CUDA_GRAPHS_ENABLED=false
ENV CUDA_GRAPH_AUDIO_SECONDS=30

EXPOSE 8080
CMD ["python3.10", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
