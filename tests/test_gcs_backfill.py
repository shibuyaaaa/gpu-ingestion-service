from pathlib import Path

from app.tools.gcs_backfill import _artists_from_embed_html, cdn_url_from_gcs_uri, group_inventory, load_inventory


def test_cdn_url_from_gcs_uri():
    assert (
        cdn_url_from_gcs_uri("gs://shibuya-assets/gpu-ingestion/job/segments/seg-0/other.mp3", cdn_base_url="https://cdn.test")
        == "https://cdn.test/gpu-ingestion/job/segments/seg-0/other.mp3"
    )


def test_group_inventory_keeps_complete_segments_and_marks_first_chorus(tmp_path: Path):
    inventory = tmp_path / "inventory.jsonl"
    lines = [
        '{"uri":"gs://shibuya-assets/gpu-ingestion/crawler:s:spotify123:bulk/segments/seg-1/other.mp3"}',
        '{"uri":"gs://shibuya-assets/gpu-ingestion/crawler:s:spotify123:bulk/segments/seg-1/vocals.mp3"}',
        '{"uri":"gs://shibuya-assets/gpu-ingestion/crawler:s:spotify123:bulk/segments/seg-1/drums.mp3"}',
        '{"uri":"gs://shibuya-assets/gpu-ingestion/crawler:s:spotify123:bulk/segments/seg-1/bass.mp3"}',
        '{"uri":"gs://shibuya-assets/gpu-ingestion/crawler:s:spotify123:bulk/segments/seg-2/other.mp3"}',
    ]
    inventory.write_text("\n".join(lines), encoding="utf-8")

    rows = load_inventory(inventory, cdn_base_url="https://cdn.test")
    songs = group_inventory(rows, cdn_base_url="https://cdn.test")

    assert len(songs) == 1
    assert songs[0].spotify_id == "spotify123"
    assert len(songs[0].segments) == 2
    assert songs[0].segments[0].segment == "chorus"
    assert songs[0].segments[0].complete is True
    assert songs[0].segments[1].segment == "seg_2"
    assert songs[0].segments[1].complete is False
    assert songs[0].segments[0].stems["chord"].startswith("https://cdn.test/gpu-ingestion/")


def test_artists_from_embed_html():
    html = '"artists":[{"name":"Bello\\u0026Dallas","uri":"spotify:artist:2zW"}]'

    assert _artists_from_embed_html(html) == ["Bello&Dallas"]
