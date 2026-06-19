import asyncio
import logging
import signal
import time
from typing import Any

from dotenv import load_dotenv

from app.config import Settings, settings
from app.crawler.kworb import KworbSpotifyChartClient
from app.crawler.ops import IngestionOpsClient, IngestionOpsUnavailable
from app.crawler.publisher import LocalIngestionPublisher
from app.crawler.spotify import SpotifyChartPlaylistClient
from app.crawler.store import CrawlerStore
from app.crawler.types import ChartCandidate
from app.legacy.db import DBClient
from app.library_membership import LibraryMembershipChecker

logger = logging.getLogger(__name__)


class CrawlerRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        store: CrawlerStore,
        provider: Any,
        publisher: Any,
        ops: Any,
        library: Any,
        now: Any = time.time,
    ):
        self.settings = settings
        self.store = store
        self.provider = provider
        self.publisher = publisher
        self.ops = ops
        self.library = library
        self._now = now

    async def run_forever(self) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop.set)
            except NotImplementedError:
                pass

        while not stop.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("crawler tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.settings.crawler_poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def run_once(self) -> dict[str, Any]:
        if not self.settings.crawler_enabled:
            return {"enabled": False}

        session = self.store.get_active_session()
        if session and session["status"] == "waiting":
            refreshed = await self._refresh_waiting_session(session)
            if not refreshed.get("completed"):
                return refreshed
            session = None

        if session is None:
            session = self.store.create_session(batch_size=self.settings.crawler_batch_size)

        if session["status"] == "submitting":
            return await self._submit_session(session)
        return {"session": session, "status": "idle"}

    async def _submit_session(self, session: dict[str, Any]) -> dict[str, Any]:
        session_id = session["id"]
        existing_count = len(self.store.submissions_for_session(session_id))
        remaining = max(0, int(session["batch_size"]) - existing_count)
        if remaining <= 0:
            self.store.mark_session_waiting(
                session_id,
                next_poll_at=self._now() + self.settings.crawler_poll_seconds,
            )
            return {"session_id": session_id, "submitted": existing_count, "status": "waiting"}

        try:
            source_urls = self.settings.crawler_kworb_chart_urls or self.settings.crawler_spotify_playlist_urls
            candidates = await self.provider.fetch_candidates(
                source_urls,
                max_pages=self.settings.crawler_max_candidate_pages,
            )
        except Exception as exc:
            self.store.mark_session_error(
                session_id,
                next_poll_at=self._now() + self.settings.crawler_poll_seconds,
                error=str(exc)[:1000],
            )
            return {"session_id": session_id, "submitted": existing_count, "error": str(exc)[:1000]}

        submitted = 0
        skipped_library = 0
        publish_failures = 0
        for candidate in candidates:
            if submitted >= remaining:
                break
            if self.store.candidate_consumed(candidate.spotify_id):
                continue

            self.store.record_candidate(candidate, status="checking_library", session_id=session_id)
            library_result = await self.library.lookup(candidate.to_metadata())
            if library_result.exists:
                self.store.record_candidate(candidate, status="skipped_library", session_id=session_id)
                skipped_library += 1
                continue

            payload = _payload_for_candidate(session_id, candidate)
            try:
                await self.publisher.publish(payload)
            except Exception as exc:
                publish_failures += 1
                self.store.record_candidate(
                    candidate,
                    status="publish_failed",
                    session_id=session_id,
                    error=str(exc)[:1000],
                )
                continue

            self.store.record_candidate(candidate, status="submitted", session_id=session_id)
            self.store.record_submission(
                session_id=session_id,
                candidate=candidate,
                job_id=payload["job_id"],
                payload=payload,
            )
            submitted += 1

        total_submissions = len(self.store.submissions_for_session(session_id))
        if total_submissions > 0:
            self.store.mark_session_waiting(
                session_id,
                next_poll_at=self._now() + self.settings.crawler_poll_seconds,
                error="one or more publishes failed" if publish_failures else None,
            )
            status = "waiting"
        else:
            self.store.mark_session_error(
                session_id,
                next_poll_at=self._now() + self.settings.crawler_poll_seconds,
                error="no publishable candidates found",
            )
            status = "submitting"

        return {
            "session_id": session_id,
            "status": status,
            "submitted_this_tick": submitted,
            "submitted_total": total_submissions,
            "skipped_library": skipped_library,
            "publish_failures": publish_failures,
        }

    async def _refresh_waiting_session(self, session: dict[str, Any]) -> dict[str, Any]:
        session_id = session["id"]
        submissions = self.store.submissions_for_session(session_id)
        for submission in submissions:
            if submission["status"] == "completed":
                continue
            try:
                state = await self.ops.job_state(submission["job_id"])
            except IngestionOpsUnavailable as exc:
                logger.warning("ingestion ops unavailable; retrying crawler session later: %s", exc)
                self.store.mark_session_waiting(
                    session_id,
                    next_poll_at=self._now() + self.settings.crawler_poll_seconds,
                )
                return {"session_id": session_id, "completed": False, "status": "waiting", "ops_unavailable": True}
            self.store.update_submission_status(
                job_id=submission["job_id"],
                status=state.status,
                root_status=state.root_status,
                child_summary=state.child_summary,
                error=state.error,
            )

        if self.store.all_submissions_terminal(session_id):
            self.store.mark_session_completed(session_id)
            return {"session_id": session_id, "completed": True}
        self.store.mark_session_waiting(
            session_id,
            next_poll_at=self._now() + self.settings.crawler_poll_seconds,
        )
        return {"session_id": session_id, "completed": False, "status": "waiting"}


def _payload_for_candidate(session_id: str, candidate: ChartCandidate) -> dict[str, Any]:
    return {
        "job_id": f"crawler:{session_id}:{candidate.spotify_id}",
        "job_type": "bulk_dissect",
        "source": candidate.source,
        "crawler_session_id": session_id,
        "crawler_source": "spotify_chart_playlist",
        "crawler_rank": candidate.rank,
        "crawler_popularity": candidate.popularity,
        "spotify_metadata": candidate.to_metadata(),
    }


async def _main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    if not settings.crawler_enabled:
        logger.info("crawler disabled; set CRAWLER_ENABLED=true to run")
        return
    db = DBClient(min_size=settings.db_pool_min_size, max_size=settings.db_pool_max_size)
    library = LibraryMembershipChecker(db=db, settings=settings)
    runner = CrawlerRunner(
        settings=settings,
        store=CrawlerStore(settings.crawler_session_db_path),
        provider=(
            KworbSpotifyChartClient()
            if settings.crawler_kworb_chart_urls
            else SpotifyChartPlaylistClient()
        ),
        publisher=LocalIngestionPublisher(ingestion_url=settings.crawler_ingestion_url),
        ops=IngestionOpsClient(base_url=settings.crawler_ops_base_url),
        library=library,
    )
    try:
        await library.warmup()
        await runner.run_forever()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(_main())
