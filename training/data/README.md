# Raw historical gamelog data

This directory's `raw/` subdirectory holds the output of
[`collect_gamelogs.py`](collect_gamelogs.py): one Parquet file per player per
season, pulled from stats.nba.com via `nba_api`. It is the raw-data
foundation for the (not-yet-built) training pipeline — nothing in it has
been feature-engineered.

## Does this belong in Git?

**No.** `training/data/raw/` is gitignored. It is large, fully regenerable
from stats.nba.com by rerunning the collector, and specific to whoever ran
it (timestamps, exact retry history, etc. in the manifest files). Committing
it would bloat the repository the same way `models/points_regression.pkl`'s
git history already has (see the earlier technical audit) — don't repeat
that pattern for a dataset that's an order of magnitude larger and, unlike
the model artifact, trivially reproducible.

If you need to hand raw data to someone else (a teammate, CI, cloud
training), copy `training/data/raw/` directly (e.g. zip it) or re-run the
collector against their own machine — do not commit it.

## What gets generated

```
training/data/raw/
  2023-24/
    players/
      2544.parquet        # one player's deduplicated rows for this season
      1629029.parquet
      ...
    _manifest.json         # progress checkpoint + run summary for this season
  2024-25/
    players/
      ...
    _manifest.json
  2025-26/
    players/
      ...
    _manifest.json
```

Each `players/<player_id>.parquet` file holds every raw gamelog row nba_api
returned for that player in that season (native `PlayerGameLog` columns,
plus a few lossless derived columns — team/opponent/home-away parsed from
`MATCHUP`, and collection provenance). See `RAW_GAMELOG_COLUMNS` in
[`storage.py`](storage.py) for the exact, authoritative column list.

`_manifest.json` is the per-season checkpoint: which player IDs are done,
their row counts, which ones failed and why, and summary counts/timestamps.
It's a convenience and a fast resume-lookup — the real source of truth for
"is this player done" is always whether their Parquet file exists and reads
back cleanly (see `storage.is_player_collected`).

## Which seasons are present

Whatever's configured in `training.config.TRAINING_SEASONS` at collection
time — currently `("2023-24", "2024-25", "2025-26")`. Check
`training/data/raw/<season>/_manifest.json`'s `players_succeeded` /
`total_players_in_roster` (logged at the end of each run) to see how
complete a given season's collection actually is; a season directory
existing does not by itself mean collection finished.

## How to resume collection

Just rerun the same command:

```powershell
python -m training.data.collect_gamelogs
```

Players already on disk for a season are skipped automatically — this is
the default behavior of a plain rerun, not something you need to opt into.
It's safe to stop the process at any point (Ctrl-C, a crash, a lost
connection, stats.nba.com throttling) and pick up later with the same
command; it never re-fetches a player whose file is already present and
readable.

To collect in short, controlled bursts instead of one long run:

```powershell
python -m training.data.collect_gamelogs --max-players 50
```

Run it again as many times as needed; each run picks up the next 50
not-yet-collected players.

## How to refresh or add a season

- **Add a future season** (e.g. once 2026-27 has started): add it to
  `TRAINING_SEASONS` in [`training/config.py`](../config.py). Nothing else
  needs to change — that's the entire point of centralizing season config
  there rather than scattering season strings through the collector.
- **Refresh a specific player** whose data you suspect is stale or wrong:
  ```powershell
  python -m training.data.collect_gamelogs --season 2025-26 --force-player 2544
  ```
- **Refresh an entire season from scratch** (rare — e.g. after a schema
  change to `normalize_raw_gamelog`):
  ```powershell
  python -m training.data.collect_gamelogs --season 2025-26 --no-resume
  ```
  This re-fetches every player for that season, overwriting their existing
  files.

## Full run: what to expect

A full collection across all three configured seasons is a multi-hour,
several-thousand-request job (one `LeagueDashPlayerStats` roster call per
season, then one `PlayerGameLog` call per player who actually played that
season — roughly 500-600 players per season). It is deliberately paced to
be polite to stats.nba.com (a fixed delay between requests, generous
retries with exponential backoff) rather than fast — see
[`training/config.py`](../config.py) for the exact timing constants.

Run it, don't launch-and-forget for the first attempt: watch the logs for
a little while to confirm requests are succeeding before leaving it
unattended for hours.
