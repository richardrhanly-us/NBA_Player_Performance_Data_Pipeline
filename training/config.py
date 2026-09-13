"""
Central configuration for the historical data-collection layer.

This is the single place that defines which NBA seasons the raw-data
collector targets, where raw output is written, and how the collector
behaves when talking to stats.nba.com. Nothing under training/ should
hardcode a season string or a network-behavior constant outside of this
module -- change a value here, not in collect_gamelogs.py, to adjust scope.

Adding a future season (e.g. once 2026-27 has started) is a one-line change
to TRAINING_SEASONS below, not a code change anywhere else in this package.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Season scope
# ---------------------------------------------------------------------------

# Seasons the raw historical collector targets, in nba_api's own
# "YYYY-YY" season-string format.
TRAINING_SEASONS = ("2023-24", "2024-25", "2025-26")

# ---------------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------------

TRAINING_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = TRAINING_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = TRAINING_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# stats.nba.com request behavior
#
# This collector is an offline, unattended, hours-long batch job -- unlike
# the live Streamlit app's request path, it should be tuned to be patient
# (generous retries, real exponential backoff, a polite fixed delay between
# requests) rather than fast. Reliability over throughput.
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 30

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0
BACKOFF_MAX_SECONDS = 60.0
BACKOFF_JITTER_SECONDS = 1.0

# Polite fixed delay between successful requests (independent of retries),
# to avoid hammering stats.nba.com even when nothing is failing.
REQUEST_DELAY_SECONDS = 0.75

# Same spoofed browser headers already used by the production app's
# get_player_gamelog_df() in src/shared_app.py. Duplicated here rather than
# imported so that training/ has no dependency on the Streamlit application
# module (and the streamlit/gspread/nba_api-live import chain that pulls in).
NBA_STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}
