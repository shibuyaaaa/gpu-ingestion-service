import tempfile
from pathlib import Path

from app.config import Settings
from app.crawler.kworb import parse_kworb_chart
from app.crawler.ops import JobTerminalState
from app.crawler.runner import CrawlerRunner
from app.crawler.spotify import _dedupe_and_sort
from app.crawler.store import CrawlerStore
from app.crawler.types import ChartCandidate


class FakeProvider:
    def __init__(self, candidates: list[ChartCandidate]):
        self.candidates = candidates
        self.calls = 0

    async def fetch_candidates(self, playlist_urls: list[str], *, max_pages: int) -> list[ChartCandidate]:
        self.calls += 1
        return list(self.candidates)


class FakePublisher:
    def __init__(self, *, fail_for: set[str] | None = None):
        self.fail_for = fail_for or set()
        self.payloads = []

    async def publish(self, payload: dict) -> str:
        if payload["source"] in self.fail_for:
            raise RuntimeError("publish failed")
        self.payloads.append(payload)
        return f"message-{len(self.payloads)}"


class FakeLibraryResult:
    def __init__(self, exists: bool):
        self.exists = exists


class FakeLibrary:
    def __init__(self, existing_ids: set[str] | None = None):
        self.existing_ids = existing_ids or set()

    async def lookup(self, metadata: dict):
        return FakeLibraryResult(metadata.get("spotify_id") in self.existing_ids)


class FakeOps:
    def __init__(self):
        self.states: dict[str, JobTerminalState] = {}

    async def job_state(self, job_id: str) -> JobTerminalState:
        return self.states.get(job_id, JobTerminalState(status="running", root_status="queued", child_summary={}))


def test_spotify_candidates_dedupe_and_sort_by_popularity_then_rank():
    candidates = [
        _candidate("a", popularity=60, rank=3),
        _candidate("b", popularity=90, rank=2),
        _candidate("c", popularity=90, rank=1),
        _candidate("a", popularity=80, rank=8),
    ]

    sorted_candidates = _dedupe_and_sort(candidates)

    assert [candidate.spotify_id for candidate in sorted_candidates] == ["c", "b", "a"]
    assert sorted_candidates[-1].popularity == 80


def test_kworb_chart_parser_extracts_spotify_track_ids():
    markup = """
    <tr><td class="np">5</td>
    <td class="np">+2</td>
    <td class="text mp"><div><a href="../artist/6BRxQ8cD3eqnrVj6WKDok8.html">Ella Langley</a> - <a href="../track/65DbTqJKhbwqYbZ1Okr0rc.html">Choosin' Texas</a></div></td>
    <td>243</td><td>1</td><td class="np mini text">(x42)</td>
    <td>1,546,826</td><td>+66,329</td><td class="smaller">12,361,813</td><td class="smaller">-195,954</td><td>299,187,347</td></tr>
    """

    candidates = parse_kworb_chart(markup, chart_url="https://kworb.net/spotify/country/us_daily.html")

    assert len(candidates) == 1
    assert candidates[0].spotify_id == "65DbTqJKhbwqYbZ1Okr0rc"
    assert candidates[0].title == "Choosin' Texas"
    assert candidates[0].artist == "Ella Langley"
    assert candidates[0].rank == 5


async def test_crawler_submits_batch_limit_and_skips_existing_library_songs():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(tmp, batch_size=3)
        store = CrawlerStore(Path(tmp) / "crawler.sqlite3")
        provider = FakeProvider([_candidate(str(idx), popularity=100 - idx, rank=idx) for idx in range(5)])
        publisher = FakePublisher()
        runner = CrawlerRunner(
            settings=settings,
            store=store,
            provider=provider,
            publisher=publisher,
            ops=FakeOps(),
            library=FakeLibrary(existing_ids={"1"}),
        )

        result = await runner.run_once()

        assert result["submitted_total"] == 3
        assert [payload["source"] for payload in publisher.payloads] == [
            "spotify:track:0",
            "spotify:track:2",
            "spotify:track:3",
        ]
        assert store.get_active_session()["status"] == "waiting"


async def test_crawler_does_not_resubmit_consumed_candidates_across_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(tmp, batch_size=2)
        store = CrawlerStore(Path(tmp) / "crawler.sqlite3")
        provider = FakeProvider([_candidate(str(idx), popularity=100 - idx, rank=idx) for idx in range(3)])
        ops = FakeOps()
        publisher = FakePublisher()
        runner = CrawlerRunner(
            settings=settings,
            store=store,
            provider=provider,
            publisher=publisher,
            ops=ops,
            library=FakeLibrary(),
        )

        await runner.run_once()
        first_session = store.get_active_session()["id"]
        for submission in store.submissions_for_session(first_session):
            ops.states[submission["job_id"]] = JobTerminalState(
                status="completed",
                root_status="completed",
                child_summary={"active": 0, "failed": 0},
            )
        await runner.run_once()
        await runner.run_once()

        assert [payload["source"] for payload in publisher.payloads] == [
            "spotify:track:0",
            "spotify:track:1",
            "spotify:track:2",
        ]


async def test_failed_terminal_jobs_allow_next_session_to_start():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(tmp, batch_size=1)
        store = CrawlerStore(Path(tmp) / "crawler.sqlite3")
        provider = FakeProvider([_candidate("a", popularity=99, rank=0), _candidate("b", popularity=98, rank=1)])
        ops = FakeOps()
        publisher = FakePublisher()
        runner = CrawlerRunner(
            settings=settings,
            store=store,
            provider=provider,
            publisher=publisher,
            ops=ops,
            library=FakeLibrary(),
        )

        await runner.run_once()
        first_job_id = publisher.payloads[0]["job_id"]
        ops.states[first_job_id] = JobTerminalState(
            status="failed",
            root_status="completed",
            child_summary={"active": 0, "failed": 1},
            error="child failed",
        )
        await runner.run_once()
        await runner.run_once()

        assert len(publisher.payloads) == 2
        assert publisher.payloads[1]["source"] == "spotify:track:b"


async def test_waiting_session_rechecks_failed_submission_until_session_closes():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(tmp, batch_size=2)
        store = CrawlerStore(Path(tmp) / "crawler.sqlite3")
        provider = FakeProvider([
            _candidate("a", popularity=99, rank=0),
            _candidate("b", popularity=98, rank=1),
        ])
        ops = FakeOps()
        publisher = FakePublisher()
        runner = CrawlerRunner(
            settings=settings,
            store=store,
            provider=provider,
            publisher=publisher,
            ops=ops,
            library=FakeLibrary(),
        )

        await runner.run_once()
        session_id = store.get_active_session()["id"]
        job_id = publisher.payloads[0]["job_id"]
        second_job_id = publisher.payloads[1]["job_id"]
        ops.states[job_id] = JobTerminalState(
            status="failed",
            root_status="failed",
            child_summary={"total": 1, "active": 1, "completed": 0, "failed": 0},
        )
        ops.states[second_job_id] = JobTerminalState(
            status="running",
            root_status="completed",
            child_summary={"total": 1, "active": 1, "completed": 0, "failed": 0},
        )
        await runner.run_once()
        ops.states[job_id] = JobTerminalState(
            status="completed",
            root_status="failed",
            child_summary={"total": 1, "active": 0, "completed": 1, "failed": 0},
        )
        await runner.run_once()

        detail = store.session_detail(session_id)
        assert detail["status"] == "waiting"
        assert detail["failed_count"] == 0
        assert detail["completed_count"] == 1


async def test_local_publish_failure_is_retryable_without_duplicating_successes():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(tmp, batch_size=2)
        store = CrawlerStore(Path(tmp) / "crawler.sqlite3")
        provider = FakeProvider([_candidate("a", popularity=99, rank=0), _candidate("b", popularity=98, rank=1)])
        publisher = FakePublisher(fail_for={"spotify:track:a"})
        runner = CrawlerRunner(
            settings=settings,
            store=store,
            provider=provider,
            publisher=publisher,
            ops=FakeOps(),
            library=FakeLibrary(),
        )

        await runner.run_once()
        first_success = publisher.payloads[0]["job_id"]
        runner.ops.states[first_success] = JobTerminalState(
            status="completed",
            root_status="completed",
            child_summary={"active": 0, "failed": 0},
        )
        publisher.fail_for = set()
        await runner.run_once()
        await runner.run_once()

        assert [payload["source"] for payload in publisher.payloads] == ["spotify:track:b", "spotify:track:a"]
        assert store.candidate_consumed("a") is True


def _candidate(spotify_id: str, *, popularity: int, rank: int) -> ChartCandidate:
    return ChartCandidate(
        spotify_id=spotify_id,
        title=f"Song {spotify_id}",
        artist="Artist",
        artists=["Artist"],
        popularity=popularity,
        playlist_source="spotify:playlist:charts",
        rank=rank,
    )


def _settings(tmp: str, *, batch_size: int) -> Settings:
    return Settings(
        crawler_enabled=True,
        crawler_batch_size=batch_size,
        crawler_poll_seconds=0.01,
        crawler_spotify_playlist_urls=["spotify:playlist:charts"],
        crawler_ingestion_url="http://127.0.0.1:8080",
        crawler_session_db_path=Path(tmp) / "crawler.sqlite3",
        library_precheck_enabled=True,
    )
