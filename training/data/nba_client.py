"""
Resilient stats.nba.com access for the historical collector.

Two things live here:

1. A generic retry wrapper (`call_with_retries`) implementing exponential
   backoff with jitter, used by every network call this package makes.
2. Two thin, purpose-built fetch functions built on top of it:
   - fetch_season_roster(season): who actually played in a season, via
     LeagueDashPlayerStats -- see that function's docstring for why this,
     and specifically not CommonAllPlayers, is used for player enumeration.
   - fetch_player_gamelog(player_id, season): one player's raw game log
     for a season, via PlayerGameLog.

Both fetch functions accept injectable `sleep_func`/`rand_func` so tests can
exercise the real retry/backoff logic without actually sleeping.
"""

import logging
import random
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog

from training import config

logger = logging.getLogger(__name__)


class NbaApiError(RuntimeError):
    """Raised when a stats.nba.com request exhausts every retry attempt."""


def _backoff_delay(attempt, *, rand_func=random.random):
    """
    Exponential backoff with jitter for the given 1-indexed attempt number.
    attempt=1 -> base delay; each subsequent attempt roughly doubles, capped
    at BACKOFF_MAX_SECONDS, plus a small random jitter so retries from
    multiple runs (or a run restarted right after failing) don't all line
    up on the same schedule.
    """
    delay = min(
        config.BACKOFF_BASE_SECONDS * (config.BACKOFF_MULTIPLIER ** (attempt - 1)),
        config.BACKOFF_MAX_SECONDS,
    )
    jitter = rand_func() * config.BACKOFF_JITTER_SECONDS
    return delay + jitter


def call_with_retries(
    func, *, description, sleep_func=time.sleep, rand_func=random.random
):
    """
    Call the zero-argument callable `func` (expected to perform one network
    request and return its result), retrying on any exception with
    exponential backoff + jitter, up to config.MAX_ATTEMPTS times.

    Raises NbaApiError, chained from the last underlying exception, if every
    attempt fails. Never raises the raw underlying exception directly, so
    callers only need to handle one failure type.
    """
    last_exc = None

    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        try:
            result = func()
            if attempt > 1:
                logger.info(
                    "%s succeeded on attempt %d/%d",
                    description,
                    attempt,
                    config.MAX_ATTEMPTS,
                )
            return result
        except Exception as exc:  # noqa: BLE001 - intentionally broad: any
            # failure from an arbitrary nba_api/network call must be retried
            # here, not just a pre-enumerated subset of exception types.
            last_exc = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s: %s",
                description,
                attempt,
                config.MAX_ATTEMPTS,
                type(exc).__name__,
                exc,
            )
            if attempt < config.MAX_ATTEMPTS:
                delay = _backoff_delay(attempt, rand_func=rand_func)
                logger.debug("%s: backing off %.2fs before retry", description, delay)
                sleep_func(delay)

    logger.error(
        "%s: exhausted %d attempts, giving up", description, config.MAX_ATTEMPTS
    )
    raise NbaApiError(
        f"{description} failed after {config.MAX_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


def _played_mask(games_played: pd.Series) -> pd.Series:
    """Defensive belt-and-suspenders filter: keep only rows with GP > 0."""
    return pd.to_numeric(games_played, errors="coerce").fillna(0) > 0


def fetch_season_roster(season, *, sleep_func=time.sleep, rand_func=random.random):
    """
    Returns the players who actually appeared in a game during `season`
    (PLAYER_ID, PLAYER_NAME, GP, and the rest of LeagueDashPlayerStats'
    columns), regardless of whether they are on today's active-player list.
    One request per season.

    This uses LeagueDashPlayerStats rather than the more obviously-named
    CommonAllPlayers endpoint. CommonAllPlayers was tried first and rejected
    after live verification: `is_only_current_season=1` does NOT return that
    season's roster (empirically, for season="2023-24" it returned 141
    players who had all LEFT the league after 2023-24, and did not include
    LeBron James, who plainly played that season) -- its actual semantics
    appear to be "no longer on any roster as of today, last seen in this
    season", not "played in this season". `is_only_current_season=0`
    returns the full ~5,200-player all-time roster and ignores the `season`
    parameter for filtering purposes entirely.

    LeagueDashPlayerStats, in contrast, is a season-scoped stat aggregation:
    every row it returns is, by construction, a player with at least one
    game in that season (verified live: querying season="2023-24" returns
    exactly 572 players, includes LeBron James, and every row has GP >= 1).
    This is what actually solves the "include players who have since
    retired but did play in this historical season" requirement.
    """

    def _do_call():
        response = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            headers=config.NBA_STATS_HEADERS,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        return response.get_data_frames()[0]

    df = call_with_retries(
        _do_call,
        description=f"LeagueDashPlayerStats roster fetch (season={season})",
        sleep_func=sleep_func,
        rand_func=rand_func,
    )

    if df is None:
        return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "GP"])

    if "GP" in df.columns:
        before = len(df)
        df = df[_played_mask(df["GP"])].copy()
        if len(df) != before:
            logger.info("Season %s: %d/%d players had GP > 0", season, len(df), before)

    return df.reset_index(drop=True)


def fetch_player_gamelog(
    player_id, season, *, sleep_func=time.sleep, rand_func=random.random
):
    """One player's raw PlayerGameLog rows for a season. One request."""

    def _do_call():
        response = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            headers=config.NBA_STATS_HEADERS,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        return response.get_data_frames()[0]

    return call_with_retries(
        _do_call,
        description=f"PlayerGameLog fetch (player_id={player_id}, season={season})",
        sleep_func=sleep_func,
        rand_func=rand_func,
    )
