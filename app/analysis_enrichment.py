import math
import time
from pathlib import Path
from typing import Any


PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def enrich_analysis(analysis: dict[str, Any], audio_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    enriched = dict(analysis or {})
    timings: dict[str, float] = {}

    started = time.perf_counter()
    _add_beat_timeline(enriched)
    timings["beat_timeline_enrich_seconds"] = round(time.perf_counter() - started, 6)

    if not _has_key(enriched) and audio_path:
        started = time.perf_counter()
        key_result = estimate_key(audio_path)
        timings["key_estimation_seconds"] = round(time.perf_counter() - started, 6)
        if key_result:
            enriched["key"] = key_result["key"]
            enriched["key_confidence"] = key_result["confidence"]
            enriched["key_method"] = key_result["method"]

    return enriched, timings


def estimate_key(audio_path: str | Path) -> dict[str, Any] | None:
    try:
        import librosa
        import numpy as np
    except Exception:
        return None

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=180.0)
        if y.size == 0:
            return None
        try:
            y = librosa.effects.harmonic(y)
        except Exception:
            pass
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        if chroma.size == 0:
            return None
        weights = np.maximum(np.mean(chroma, axis=1), 0)
        if float(np.sum(weights)) <= 0:
            return None
        weights = weights / np.sum(weights)

        candidates: list[tuple[float, int, str]] = []
        for tonic in range(12):
            major = np.roll(np.array(MAJOR_PROFILE, dtype=float), tonic)
            minor = np.roll(np.array(MINOR_PROFILE, dtype=float), tonic)
            candidates.append((_correlation(weights, major), tonic, "major"))
            candidates.append((_correlation(weights, minor), tonic, "minor"))
        candidates.sort(reverse=True, key=lambda item: item[0])
        best_score, tonic, mode = candidates[0]
        next_score = candidates[1][0] if len(candidates) > 1 else 0.0
        confidence = max(0.0, min(1.0, (best_score - next_score + 0.15) / 0.30))
        return {
            "key": f"{PITCH_CLASS_NAMES[tonic]} {mode}",
            "confidence": round(confidence, 3),
            "method": "librosa_chroma_krumhansl",
        }
    except Exception:
        return None


def _add_beat_timeline(analysis: dict[str, Any]) -> None:
    beat_times = _number_list(
        analysis.get("beats")
        or analysis.get("beat_times")
        or analysis.get("beat_timestamps")
    )
    if not beat_times:
        return

    positions = _int_list(analysis.get("beat_positions"))
    if len(positions) != len(beat_times):
        positions = [index % 4 + 1 for index in range(len(beat_times))]

    raw_downbeats = _number_list(
        analysis.get("downbeats")
        or analysis.get("downbeat_times")
        or analysis.get("downbeat_timestamps")
    )
    if raw_downbeats:
        downbeat_set = {_rounded_time(value) for value in raw_downbeats}
    else:
        downbeat_set = {
            _rounded_time(time_value)
            for time_value, position in zip(beat_times, positions, strict=False)
            if position == 1
        }

    beat_grid = []
    upbeats = []
    downbeats = []
    for index, (time_value, position) in enumerate(zip(beat_times, positions, strict=False)):
        rounded = _rounded_time(time_value)
        is_downbeat = rounded in downbeat_set or position == 1
        is_upbeat = not is_downbeat
        if is_downbeat:
            downbeats.append(rounded)
        else:
            upbeats.append(rounded)
        beat_grid.append(
            {
                "index": index,
                "time": rounded,
                "second": rounded,
                "position": int(position),
                "is_downbeat": is_downbeat,
                "is_upbeat": is_upbeat,
            }
        )

    analysis["beats"] = [_rounded_time(value) for value in beat_times]
    analysis["beat_positions"] = [int(value) for value in positions]
    analysis["downbeats"] = sorted(set(downbeats))
    analysis["upbeats"] = upbeats
    analysis["beat_grid"] = beat_grid
    analysis["beats_by_second"] = _beats_by_second(beat_grid)

    if "beat_analysis_bpm" not in analysis:
        bpm = _estimate_bpm_from_times(beat_times)
        if bpm is not None:
            analysis["beat_analysis_bpm"] = bpm


def _beats_by_second(beat_grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}
    for beat in beat_grid:
        second = int(math.floor(float(beat["time"])))
        bucket = buckets.setdefault(second, {"second": second, "beats": [], "downbeats": [], "upbeats": []})
        entry = {
            "time": beat["time"],
            "position": beat["position"],
            "index": beat["index"],
        }
        bucket["beats"].append(entry)
        if beat["is_downbeat"]:
            bucket["downbeats"].append(entry)
        else:
            bucket["upbeats"].append(entry)
    return [buckets[key] for key in sorted(buckets)]


def _estimate_bpm_from_times(times: list[float]) -> float | None:
    if len(times) < 3:
        return None
    intervals = [
        times[index + 1] - times[index]
        for index in range(len(times) - 1)
        if times[index + 1] > times[index]
    ]
    if not intervals:
        return None
    average_interval = sum(intervals) / len(intervals)
    if average_interval <= 0:
        return None
    bpm = 60.0 / average_interval
    while bpm < 70:
        bpm *= 2
    while bpm > 210:
        bpm /= 2
    return round(bpm, 3)


def _number_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    numbers = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("time") or item.get("second") or item.get("start") or item.get("position")
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    numbers = []
    for item in value:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return numbers


def _rounded_time(value: float) -> float:
    return round(float(value), 3)


def _has_key(analysis: dict[str, Any]) -> bool:
    for key in ("key", "detected_key", "musical_key"):
        if str(analysis.get(key) or "").strip():
            return True
    return False


def _correlation(values: Any, profile: Any) -> float:
    import numpy as np

    left = np.asarray(values, dtype=float)
    right = np.asarray(profile, dtype=float)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)
