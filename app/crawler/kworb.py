import html
import re
from urllib.parse import urljoin

import httpx

from app.crawler.types import ChartCandidate

KWORB_ROW_RE = re.compile(r"<tr>(?P<row>.*?)</tr>", re.DOTALL)
KWORB_TITLE_CELL_RE = re.compile(r'<td class="text mp"><div>(?P<title_cell>.*?)</div></td>', re.DOTALL)
KWORB_DAILY_RANK_RE = re.compile(r'^\s*<td class="np">(?P<rank>\d+)</td>', re.DOTALL)
ANCHOR_RE = re.compile(r'<a href="(?P<href>[^"]+)">(?P<label>.*?)</a>', re.DOTALL)
SPOTIFY_TRACK_ID_RE = re.compile(r"/track/(?P<spotify_id>[A-Za-z0-9]+)\.html")


class KworbSpotifyChartClient:
    def __init__(self, *, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds

    async def fetch_candidates(self, chart_urls: list[str], *, max_pages: int) -> list[ChartCandidate]:
        candidates: list[ChartCandidate] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for source_index, chart_url in enumerate(chart_urls[: max(1, max_pages)]):
                response = await client.get(chart_url)
                response.raise_for_status()
                candidates.extend(
                    parse_kworb_chart(
                        response.text,
                        chart_url=chart_url,
                        rank_offset=source_index * 1000,
                    )
                )
        return _dedupe_by_best_rank(candidates)


def parse_kworb_chart(markup: str, *, chart_url: str, rank_offset: int = 0) -> list[ChartCandidate]:
    candidates: list[ChartCandidate] = []
    chart_position = 0
    for match in KWORB_ROW_RE.finditer(markup):
        row = match.group("row")
        title_match = KWORB_TITLE_CELL_RE.search(row)
        if title_match is None:
            continue
        chart_position += 1
        rank_match = KWORB_DAILY_RANK_RE.search(row)
        chart_rank = int(rank_match.group("rank")) if rank_match else chart_position
        rank = rank_offset + chart_rank
        anchors = list(ANCHOR_RE.finditer(title_match.group("title_cell")))
        if len(anchors) < 2:
            continue
        track_anchor = anchors[-1]
        track_id_match = SPOTIFY_TRACK_ID_RE.search(track_anchor.group("href"))
        if not track_id_match:
            continue
        artists = [_clean_label(anchor.group("label")) for anchor in anchors[:-1]]
        artists = [artist for artist in artists if artist]
        title = _clean_label(track_anchor.group("label"))
        if not artists or not title:
            continue
        candidates.append(
            ChartCandidate(
                spotify_id=track_id_match.group("spotify_id"),
                title=title,
                artist=artists[0],
                artists=artists,
                popularity=max(1, 101 - min(chart_rank, 100)),
                playlist_source=urljoin(chart_url, chart_url),
                rank=rank,
            )
        )
    return candidates


def _dedupe_by_best_rank(candidates: list[ChartCandidate]) -> list[ChartCandidate]:
    best: dict[str, ChartCandidate] = {}
    for candidate in candidates:
        previous = best.get(candidate.spotify_id)
        if previous is None or candidate.rank < previous.rank:
            best[candidate.spotify_id] = candidate
    return sorted(best.values(), key=lambda candidate: (candidate.rank, -candidate.popularity, candidate.spotify_id))


def _clean_label(value: str) -> str:
    text = re.sub(r"<.*?>", "", value)
    return html.unescape(text).strip()
