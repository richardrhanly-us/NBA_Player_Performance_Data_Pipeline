"""
Collect and persist raw historical player gamelogs for the seasons
configured in training.config.TRAINING_SEASONS.

This is the raw-data foundation for the (not-yet-built) training pipeline.
It does not engineer features, does not touch the legacy model, and does
not touch the production prediction path in any way.

Usage:
    python -m training.data.collect_gamelogs
    python -m training.data.collect_gamelogs --season 2025-26
    python -m training.data.collect_gamelogs --season 2023-24 2024-25 --max-players 50
    python -m training.data.collect_gamelogs --force-player 2544 1629029
    python -m training.data.collect_gamelogs --no-resume --season 2025-26

Resuming is the default behavior of a plain rerun: players already on disk
for a season (verified by storage.is_player_collected, not just the
manifest) are skipped automatically, so stopping the process at any point
-- Ctrl-C, a crash, a lost connection, a stats.nba.com throttle -- and
running the same command again continues rather than restarting.

This module consumes the src.data.basketball provider abstraction (a
BasketballDataProvider), not nba_api/stats.nba.com specifics directly --
see src/data/basketball/__init__.py. The active provider defaults to the
NBA development provider (get_basketball_provider()) but can be
overridden via the `provider` parameter, which is how tests inject a
fake, offline, in-memory provider instead of hitting the network.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from src.data.basketball import errors as basketball_errors
from src.data.basketball.provider import get_basketball_provider
from training import config
from training.data import storage

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_season(
    season,
    *,
    provider=None,
    max_players=None,
    force_player_ids=(),
    resume=True,
    sleep_func=time.sleep,
    request_delay=None,
):
    """
    Collect (or resume collecting) one season's raw player gamelogs.

    `provider` is a src.data.basketball.provider.BasketballDataProvider
    (defaults to get_basketball_provider(), the active NBA development
    provider). `sleep_func`/`request_delay` control only the polite
    pacing delay between players at this orchestration level -- retry
    backoff for a provider's own transient failures is that provider's
    internal concern (see src/data/basketball/providers/nba_api_provider.py).

    Returns a summary dict with the fields requested for operator
    reporting: season, players attempted/succeeded/failed (for THIS run),
    rows collected (for THIS run), and start/end time, plus a couple of
    cumulative totals useful when resuming across many runs.
    """
    provider = provider or get_basketball_provider()
    request_delay = (
        config.REQUEST_DELAY_SECONDS if request_delay is None else request_delay
    )
    force_player_ids = {int(pid) for pid in (force_player_ids or ())}

    run_start = _utc_now_iso()
    logger.info("=== Season %s: fetching roster ===", season)
    roster = provider.get_season_roster(season)
    total_in_roster = len(roster)
    logger.info("Season %s: %d players played this season", season, total_in_roster)

    manifest = storage.load_manifest(season)
    manifest["season"] = season
    if not manifest.get("start_time"):
        manifest["start_time"] = run_start
    manifest["last_resumed_at"] = run_start

    completed_ids = set(manifest.get("completed_player_ids", []))
    player_row_counts = dict(manifest.get("player_row_counts", {}))
    failed_players = dict(manifest.get("failed_players", {}))

    players_attempted = 0
    players_succeeded = 0
    players_failed = 0
    rows_collected = 0

    for i, player in enumerate(roster, start=1):
        player_id = int(player.player_id)
        player_name = str(player.player_name or "") or f"player_{player_id}"

        force_this_player = player_id in force_player_ids
        # Ground truth only: whether this player's file actually exists and
        # reads back cleanly. The manifest's completed_player_ids is a
        # reporting/checkpoint convenience and must never independently
        # certify completion -- a stale manifest entry for a player whose
        # file is missing or corrupt must not cause it to be skipped.
        already_done = (
            not force_this_player
            and resume
            and storage.is_player_collected(season, player_id)
        )
        if already_done:
            logger.debug(
                "Season %s: [%d/%d] player_id=%s (%s) already collected, skipping",
                season,
                i,
                total_in_roster,
                player_id,
                player_name,
            )
            continue

        if max_players is not None and players_attempted >= max_players:
            logger.info(
                "Season %s: reached --max-players=%d for this run, stopping early "
                "(rerun the same command to continue)",
                season,
                max_players,
            )
            break

        logger.info(
            "Season %s: [%d/%d] fetching player_id=%s (%s)",
            season,
            i,
            total_in_roster,
            player_id,
            player_name,
        )
        players_attempted += 1

        try:
            game_logs = provider.get_player_game_logs(
                player_id, season, player_name=player_name
            )
            normalized_df = storage.gamelogs_to_dataframe(game_logs)
            storage.save_player_gamelog(
                normalized_df, season=season, player_id=player_id
            )

            completed_ids.add(player_id)
            player_row_counts[str(player_id)] = len(normalized_df)
            failed_players.pop(str(player_id), None)

            players_succeeded += 1
            rows_collected += len(normalized_df)
            logger.info(
                "Season %s: player_id=%s (%s) OK -- %d rows",
                season,
                player_id,
                player_name,
                len(normalized_df),
            )

        except basketball_errors.ProviderError as exc:
            players_failed += 1
            # If a stale manifest still listed this player as completed (the
            # file was actually missing/corrupt, which is why we got here),
            # a failed repair attempt must not leave them marked complete.
            completed_ids.discard(player_id)
            player_row_counts.pop(str(player_id), None)
            failed_players[str(player_id)] = {
                "player_name": player_name,
                "error": str(exc),
                "last_attempt_at": _utc_now_iso(),
            }
            logger.error(
                "Season %s: player_id=%s (%s) FAILED, recorded for retry on next run: %s",
                season,
                player_id,
                player_name,
                exc,
            )

        # Persist progress after every single player, success or failure --
        # a crash on the *next* player must not lose this one's result.
        manifest["completed_player_ids"] = sorted(completed_ids)
        manifest["player_row_counts"] = player_row_counts
        manifest["failed_players"] = failed_players
        manifest["players_succeeded"] = len(completed_ids)
        manifest["players_failed"] = len(failed_players)
        manifest["rows_collected"] = sum(player_row_counts.values())
        storage.save_manifest(season, manifest)
        logger.debug(
            "Season %s: checkpoint written (%d completed, %d pending retry)",
            season,
            len(completed_ids),
            len(failed_players),
        )

        if request_delay:
            sleep_func(request_delay)

    run_end = _utc_now_iso()
    manifest["end_time"] = run_end
    storage.save_manifest(season, manifest)

    summary = {
        "season": season,
        "players_attempted": players_attempted,
        "players_succeeded": players_succeeded,
        "players_failed": players_failed,
        "rows_collected": rows_collected,
        "start_time": run_start,
        "end_time": run_end,
        "total_players_in_roster": total_in_roster,
        "total_completed_all_time": len(completed_ids),
        "total_pending_retry": len(failed_players),
    }
    logger.info("=== Season %s summary: %s ===", season, summary)
    return summary


def run_collection(
    seasons=None,
    *,
    provider=None,
    max_players=None,
    force_player_ids=(),
    resume=True,
    sleep_func=time.sleep,
    request_delay=None,
):
    """Run collect_season() over each of `seasons` (default: all configured
    seasons), reusing one provider instance across every season."""
    provider = provider or get_basketball_provider()
    seasons = list(seasons) if seasons else list(config.TRAINING_SEASONS)
    summaries = []

    for season in seasons:
        try:
            summaries.append(
                collect_season(
                    season,
                    provider=provider,
                    max_players=max_players,
                    force_player_ids=force_player_ids,
                    resume=resume,
                    sleep_func=sleep_func,
                    request_delay=request_delay,
                )
            )
        except basketball_errors.ProviderError as exc:
            # A season-level roster fetch failure (one request) is serious
            # but must not abort a multi-season run -- log it and move on;
            # rerunning will retry the roster fetch for this season.
            logger.error(
                "Season %s: roster fetch failed, skipping this season for now: %s",
                season,
                exc,
            )
            summaries.append(
                {
                    "season": season,
                    "error": str(exc),
                    "players_attempted": 0,
                    "players_succeeded": 0,
                    "players_failed": 0,
                    "rows_collected": 0,
                }
            )

    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.data.collect_gamelogs",
        description="Collect and persist raw historical NBA player gamelogs.",
    )
    parser.add_argument(
        "--season",
        nargs="+",
        default=None,
        metavar="SEASON",
        help=(
            "One or more seasons to collect (e.g. --season 2025-26). "
            f"Defaults to every season in training.config.TRAINING_SEASONS: "
            f"{', '.join(config.TRAINING_SEASONS)}."
        ),
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        metavar="N",
        help="Attempt at most N new players per season in this run, then stop (resumable).",
    )
    parser.add_argument(
        "--force-player",
        nargs="+",
        type=int,
        default=(),
        metavar="PLAYER_ID",
        help="Re-fetch these player IDs even if already marked collected.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore existing progress and refetch every player from scratch "
            "for the selected season(s). Resuming (skipping already-collected "
            "players) is the default behavior of a plain rerun."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    summaries = run_collection(
        seasons=args.season,
        max_players=args.max_players,
        force_player_ids=args.force_player,
        resume=not args.no_resume,
    )

    logger.info("=== Collection run complete ===")
    for summary in summaries:
        logger.info("%s", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
