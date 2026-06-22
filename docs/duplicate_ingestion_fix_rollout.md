# Duplicate Ingestion Fix Rollout

This rollout fixes duplicate Aime library rows caused by concurrent segment
fanout publishing chord-first results.

## Code Deploy

On the GPU VM, update the repo and rebuild the local Docker image:

```bash
cd /srv/gpu-ingestion-service
./deploy/vm-deploy.sh
```

The service env should include:

```bash
SOURCE_AUDIO_UPLOAD_ENABLED=true
ANALYZER_RESULT_UPLOAD_ENABLED=true
LIBRARY_PRECHECK_ENABLED=true
```

## Database Guard

Apply the companion migration from `shibuya-api`:

```bash
cd /srv/shibuya-api
./run-migration.sh shibuya_production migrations/089_dedupe_gpu_ingestion_stems.sql
```

If the production database name differs, replace `shibuya_production` with the
current database name. The migration only targets stems produced by
`gpu-ingestion` paths/models. It removes exact duplicate gpu-ingestion stem rows
and adds a partial unique index for future segment fanout writes.

## Smoke Checks

After deploy and migration:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ops/state
```

Submit one `quick_dissect` and one `bulk_dissect` job. Confirm:

- the song row has `audio_url` pointing at `gpu-ingestion/cache/source-audio/.../full.mp3`
- `all_in_one_bpm` or `beat_analysis_bpm` is populated after analysis
- chord stems appear first with `analysis_json.gpu_ingestion.status = partial`
- the song reaches `analysis_json.gpu_ingestion.status = complete` after all segment children finish
- duplicate stems do not appear for the same `song_id`, `stem_type`, `segment`,
  `start_time`, and `end_time`
