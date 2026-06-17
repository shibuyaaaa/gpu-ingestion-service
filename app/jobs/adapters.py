import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import httpx

from app.jobs.base import JobAdapter, StageResult
from app.jobs.context import JobContext
from app.legacy.audio import AudioOps
from app.legacy.utils.source import resolve_and_download_spotify_source
from app.queue import JobRecord, JobStage, JobType


class DissectAdapter(JobAdapter):
    job_types: set[str] = set()

    async def download(self, job: JobRecord, context: JobContext) -> StageResult:
        source = self._source_from_payload(job.payload)
        work_dir = context.job_work_dir(job)
        work_dir.mkdir(parents=True, exist_ok=True)

        if context.settings.dry_run_mode:
            source_path = work_dir / "source.wav"
            source_path.write_bytes(b"dry-run-audio")
            resolved = {
                "source": source,
                "youtube_url": "dry-run://youtube",
                "spotify_metadata": {"title": source, "artist": "", "query_only": True},
                "youtube_match": {"url": "dry-run://youtube"},
                "audio_path": str(source_path),
            }
        else:
            resolved = await resolve_and_download_spotify_source(source, work_dir)
            source_path = Path(resolved["audio_path"])

        wav_path = work_dir / "input.wav"
        if source_path.suffix.lower() in {".wav", ".wave"}:
            if source_path != wav_path:
                shutil.copyfile(source_path, wav_path)
        else:
            await AudioOps.convert_to_wav(source_path, wav_path)

        return StageResult(
            next_stage=JobStage.ANALYZE,
            artifacts={
                "work_dir": str(work_dir),
                "source": source,
                "source_path": str(source_path),
                "audio_path": str(wav_path),
                "youtube_url": resolved.get("youtube_url"),
                "spotify_metadata": resolved.get("spotify_metadata"),
                "youtube_match": resolved.get("youtube_match"),
            },
        )

    async def analyze(self, job: JobRecord, context: JobContext) -> StageResult:
        audio_path = job.artifacts["audio_path"]
        output_dir = Path(job.artifacts["work_dir"]) / "analysis"
        async with context.models.gpu_lock:
            context.models.active_gpu_job_id = job.id
            context.models.active_gpu_model = "all-in-one"
            try:
                result = await context.models.all_in_one.analyze(audio_path, output_dir)
            finally:
                context.models.active_gpu_job_id = None
                context.models.active_gpu_model = None
        analysis = result.get("analysis", {})
        segments = self._normalize_segments(analysis)
        chorus_segment = self._find_chorus_segment(segments)
        full_stem_urls = result.get("stem_urls") or {}
        full_stem_paths = {}
        if full_stem_urls and not context.settings.dry_run_mode:
            full_stem_paths = await self._download_full_stems(full_stem_urls, Path(job.artifacts["work_dir"]))
        return StageResult(
            next_stage=JobStage.PROCESS,
            artifacts={
                "analysis": analysis,
                "analyzer_result_path": result.get("analyzer_result_path"),
                "analyzer_result_url": result.get("analyzer_result_url"),
                "full_stem_urls": full_stem_urls,
                "full_stem_paths": full_stem_paths,
                "segments": segments,
                "chorus_segment": chorus_segment,
                "processed_segments": {},
                "skip_segment_ids": job.artifacts.get("skip_segment_ids", []),
            },
        )

    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        raise NotImplementedError

    async def _process_segment(
        self,
        job: JobRecord,
        context: JobContext,
        segment: dict[str, Any],
        *,
        output_prefix: str,
    ) -> dict[str, Any]:
        work_dir = Path(job.artifacts["work_dir"])
        segment_id = str(segment["id"])
        segment_dir = work_dir / output_prefix / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_path = segment_dir / "input.mp3"
        if context.settings.dry_run_mode:
            shutil.copyfile(job.artifacts["audio_path"], segment_path)
        else:
            await AudioOps.extract_segment(
                job.artifacts["audio_path"],
                segment_path,
                start=float(segment["start"]),
                duration=max(0.1, float(segment["end"]) - float(segment["start"])),
            )

        stems_dir = segment_dir / "stems"
        full_stem_paths = job.artifacts.get("full_stem_paths") or {}
        if full_stem_paths and not context.settings.dry_run_mode:
            stems = await self._extract_stem_segments(full_stem_paths, stems_dir, segment)
        else:
            async with context.models.gpu_lock:
                context.models.active_gpu_job_id = job.id
                context.models.active_gpu_model = "htdemucs"
                try:
                    stems = await context.models.htdemucs.separate(segment_path, stems_dir)
                finally:
                    context.models.active_gpu_job_id = None
                    context.models.active_gpu_model = None

        if context.settings.dry_run_mode:
            uploaded = {stem: path for stem, path in stems.items()}
        else:
            uploaded = {}
            for stem, path in stems.items():
                local_path = Path(path)
                if local_path.suffix.lower() != ".mp3":
                    mp3_path = local_path.with_suffix(".mp3")
                    await AudioOps.convert_to_mp3(local_path, mp3_path)
                    local_path = mp3_path
                gcs_path = f"gpu-ingestion/{job.id}/segments/{segment_id}/{stem}.mp3"
                uploaded[stem] = await context.gcs.upload(local_path, gcs_path, content_type="audio/mpeg")

        return {
            "segment": segment,
            "segment_path": str(segment_path),
            "stem_paths": stems,
            "outputs": uploaded,
        }

    @staticmethod
    async def _download_full_stems(stem_urls: dict[str, str], work_dir: Path) -> dict[str, str]:
        target_dir = work_dir / "full_stems"
        target_dir.mkdir(parents=True, exist_ok=True)

        async def download_one(client: httpx.AsyncClient, stem: str, url: str) -> tuple[str, str]:
            suffix = Path(url.split("?", 1)[0]).suffix or ".wav"
            target = target_dir / f"{stem}{suffix}"
            response = await client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)
            return stem, str(target)

        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            pairs = await asyncio.gather(
                *(download_one(client, stem, url) for stem, url in stem_urls.items() if isinstance(url, str) and url)
            )
        return dict(pairs)

    @staticmethod
    async def _extract_stem_segments(
        full_stem_paths: dict[str, str],
        output_dir: Path,
        segment: dict[str, Any],
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = max(0.1, float(segment["end"]) - float(segment["start"]))
        outputs: dict[str, str] = {}
        for stem, full_path in full_stem_paths.items():
            target = output_dir / f"{stem}.mp3"
            await AudioOps.extract_segment(full_path, target, start=float(segment["start"]), duration=duration)
            outputs[stem] = str(target)
        return outputs

    @staticmethod
    def _source_from_payload(payload: dict) -> str:
        source = (
            payload.get("source")
            or payload.get("youtube_url")
            or payload.get("spotify_source")
            or payload.get("spotify_url")
            or payload.get("spotify_query")
        )
        if not source or not str(source).strip():
            raise RuntimeError("job payload must include source, youtube_url, spotify_source, spotify_url, or spotify_query")
        return str(source).strip()

    @staticmethod
    def _normalize_segments(analysis: dict[str, Any]) -> list[dict[str, Any]]:
        raw_segments = analysis.get("segments") or []
        duration = float(analysis.get("duration") or 180.0)
        normalized = []
        for index, segment in enumerate(raw_segments):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start + 30.0))
            if end <= start:
                end = start + 30.0
            normalized.append(
                {
                    "id": str(segment.get("id") or f"seg-{index}"),
                    "start": start,
                    "end": end,
                    "label": str(segment.get("label") or f"segment-{index}"),
                }
            )
        if normalized:
            return normalized
        return [{"id": "seg-0", "start": 0.0, "end": min(duration, 30.0), "label": "segment"}]

    @staticmethod
    def _find_chorus_segment(segments: list[dict[str, Any]]) -> dict[str, Any]:
        choruses = [segment for segment in segments if "chorus" in segment["label"].lower()]
        if choruses:
            return choruses[2] if len(choruses) >= 3 else choruses[1] if len(choruses) >= 2 else choruses[0]

        def duration(segment: dict[str, Any]) -> float:
            return float(segment["end"]) - float(segment["start"])

        useful = [
            segment
            for segment in segments
            if duration(segment) >= 5.0
            and segment["label"].lower() not in {"start", "end", "outro", "silence"}
        ]
        return max(useful or segments, key=duration)


class BulkDissectAdapter(DissectAdapter):
    job_types = {JobType.BULK_DISSECT.value}

    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        segments = job.artifacts.get("segments") or []
        processed = dict(job.artifacts.get("processed_segments") or {})
        skipped = set(job.artifacts.get("skip_segment_ids") or [])

        next_segment = None
        for segment in segments:
            segment_id = str(segment["id"])
            if segment_id not in processed and segment_id not in skipped:
                next_segment = segment
                break

        if next_segment is None:
            final_outputs = {
                "job_type": job.job_type.value,
                "source": job.artifacts.get("source"),
                "processed_segment_ids": sorted(processed),
                "skipped_segment_ids": sorted(skipped),
                "dry_run": context.settings.dry_run_mode,
            }
            result_url = None
            if not context.settings.dry_run_mode:
                metadata_path = Path(job.artifacts["work_dir"]) / "bulk_result.json"
                metadata_path.write_text(json.dumps(final_outputs, indent=2), encoding="utf-8")
                result_url = await context.gcs.upload(
                    metadata_path,
                    f"gpu-ingestion/{job.id}/bulk_result.json",
                    content_type="application/json",
                )
                final_outputs["result_url"] = result_url
            return StageResult(next_stage=None, artifacts={"final_outputs": final_outputs, "result_url": result_url})

        result = await self._process_segment(job, context, next_segment, output_prefix="bulk_segments")
        processed[str(next_segment["id"])] = result
        remaining = [
            segment
            for segment in segments
            if str(segment["id"]) not in processed and str(segment["id"]) not in skipped
        ]
        return StageResult(
            next_stage=JobStage.PROCESS if remaining else None,
            artifacts={
                "processed_segments": processed,
                "last_processed_segment_id": str(next_segment["id"]),
                "final_outputs": {
                    "job_type": job.job_type.value,
                    "source": job.artifacts.get("source"),
                    "processed_segment_ids": sorted(processed),
                    "skipped_segment_ids": sorted(skipped),
                    "dry_run": context.settings.dry_run_mode,
                },
            },
        )


class QuickDissectAdapter(DissectAdapter):
    job_types = {JobType.QUICK_DISSECT.value}

    async def process(self, job: JobRecord, context: JobContext) -> StageResult:
        chorus_segment = job.artifacts.get("chorus_segment")
        if not chorus_segment:
            raise RuntimeError("quick_dissect job is missing chorus_segment")

        chorus_result = await self._process_segment(job, context, chorus_segment, output_prefix="quick_chorus")
        continuation = context.store.enqueue_continuation(
            parent_job=job,
            payload={
                "source": job.artifacts.get("source"),
                "parent_job_id": job.id,
            },
            artifacts={
                "work_dir": job.artifacts["work_dir"],
                "source": job.artifacts.get("source"),
                "source_path": job.artifacts.get("source_path"),
                "audio_path": job.artifacts["audio_path"],
                "youtube_url": job.artifacts.get("youtube_url"),
                "spotify_metadata": job.artifacts.get("spotify_metadata"),
                "youtube_match": job.artifacts.get("youtube_match"),
                "analysis": job.artifacts.get("analysis", {}),
                "analyzer_result_path": job.artifacts.get("analyzer_result_path"),
                "analyzer_result_url": job.artifacts.get("analyzer_result_url"),
                "full_stem_urls": job.artifacts.get("full_stem_urls", {}),
                "full_stem_paths": job.artifacts.get("full_stem_paths", {}),
                "segments": job.artifacts.get("segments", []),
                "chorus_segment": chorus_segment,
                "skip_segment_ids": [str(chorus_segment["id"])],
                "processed_segments": {},
            },
        )

        final_outputs = {
            "job_type": job.job_type.value,
            "quick_dissect": True,
            "source": job.artifacts.get("source"),
            "youtube_url": job.artifacts.get("youtube_url"),
            "spotify_metadata": job.artifacts.get("spotify_metadata"),
            "chorus_segment": chorus_segment,
            "chorus_outputs": chorus_result.get("outputs", {}),
            "dry_run": context.settings.dry_run_mode,
            "bulk_continuation_job_id": continuation.id,
        }
        return StageResult(
            next_stage=None,
            artifacts={
                "quick_chorus_result": chorus_result,
                "processed_segments": {str(chorus_segment["id"]): chorus_result},
                "skip_segment_ids": [str(chorus_segment["id"])],
                "bulk_continuation_job_id": continuation.id,
                "final_outputs": final_outputs,
            },
        )
