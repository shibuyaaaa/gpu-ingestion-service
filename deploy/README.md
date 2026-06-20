# Deployment Notes

Do not point production traffic at this service until parity tests pass against
the legacy worker. This repo is deployment-ready code, not a live deployment.

## Build Image

```bash
gcloud builds submit --config deploy/cloudbuild.yaml
```

## Prepare GCE VM

Use a `g2-standard-4` instance with one NVIDIA L4 and a persistent disk mounted
at `/var/lib/gpu-ingestion`.

```bash
bash deploy/gce-setup.sh
sudo nano /etc/gpu-ingestion.env
sudo systemctl enable --now gpu-ingestion
```

## Deploy On The VM

From `/opt/gpu-ingestion-service` on the VM:

```bash
bash deploy/vm-deploy.sh
```

The script fast-forwards the repo, rebuilds `gpu-ingestion-service:local`,
restarts the API and crawler units, waits for `/health`, then prunes unused
Docker images left by previous local builds. Set `DOCKER_IMAGE_PRUNE_ARGS="-f"`
to keep tagged-but-unused images.

## Health Checks

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ops/readiness
curl http://127.0.0.1:8080/ops/state
curl http://127.0.0.1:8080/ops/gpu
```

## Ingress

Use `POST /jobs` for VM-local crawler delivery, manual replay, and shadow
testing. `POST /pubsub` is kept only for legacy-compatible wrapped messages.
Jobs are committed directly to the local SQLite/WAL queue before the API returns
success.

## Rollback

Do not delete the legacy Cloud Run worker. Rollback is stopping the crawler or
repointing external publishers back to the old Cloud Run endpoint, then draining
this service:

```bash
curl -X POST http://127.0.0.1:8080/ops/drain
```
