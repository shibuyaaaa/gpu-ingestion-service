from app.tools.gcs_inventory import canonical_stem_type, parse_gpu_ingestion_uri, summarize


def test_parse_crawler_gpu_ingestion_uri():
    row = parse_gpu_ingestion_uri(
        "gs://shibuya-assets/gpu-ingestion/"
        "crawler:1781760269-d1533aea:7y4ELJ02ZssUMSkJiTfq2o:bulk:seg-10:chord/"
        "segments/seg-10/other.mp3"
    )

    assert row is not None
    assert row.bucket == "shibuya-assets"
    assert row.spotify_id == "7y4ELJ02ZssUMSkJiTfq2o"
    assert row.crawler_session_id == "1781760269-d1533aea"
    assert row.segment_id == "seg-10"
    assert row.stem == "other"
    assert row.canonical_stem == "chord"


def test_parse_ignores_non_gpu_or_non_audio_uri():
    assert parse_gpu_ingestion_uri("gs://shibuya-assets/gpu-ingestion/job/segments/seg-0/manifest.json") is None
    assert parse_gpu_ingestion_uri("gs://shibuya-assets/other/job/segments/seg-0/other.mp3") is None


def test_stem_canonicalization():
    assert canonical_stem_type("vocals") == "voice"
    assert canonical_stem_type("drums") == "beat"
    assert canonical_stem_type("other") == "chord"
    assert canonical_stem_type("bass") == "bass"


def test_summarize_counts_segments_and_stems():
    rows = [
        parse_gpu_ingestion_uri(f"gs://shibuya-assets/gpu-ingestion/crawler:s:sp:bulk/segments/seg-0/{stem}.mp3")
        for stem in ["other", "vocals", "drums", "bass"]
    ]

    summary = summarize([row for row in rows if row])

    assert summary["objects"] == 4
    assert summary["crawler_songs"] == 1
    assert summary["segments"] == 1
    assert summary["segments_with_chord"] == 1
    assert summary["segments_with_all_four_stems"] == 1
    assert summary["stem_counts"] == {"chord": 1, "voice": 1, "beat": 1, "bass": 1}
