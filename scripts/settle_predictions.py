"""
Step 10: production entry point for "settle whatever can be settled
right now". Thin CLI wrapper around
src.services.orchestration.run_settlement_cycle -- see that module for
the actual decision-making (schema readiness, request deduping via
_CachingSettlementProvider, status classification).

Exit codes (Step 10 Phase 8): 0 for SUCCESS/NO_WORK/PARTIAL, 1 for
FAILED -- same convention as scripts/persist_prediction_board.py.

Requires DATABASE_URL. Uses the same BasketballDataProvider boundary as
the rest of the app (no direct nba_api access here). Safe to run
manually at any time for incident recovery -- see the Step 10 report's
manual-recovery section.
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.orchestration import CycleStatus, run_settlement_cycle

_NON_ALARMING = (CycleStatus.SUCCESS, CycleStatus.NO_WORK, CycleStatus.PARTIAL)


def log(msg):
    print(msg, flush=True)


def main() -> int:
    start_time = time.time()
    log("[SETTLEMENT CYCLE] ===== START =====")

    result = run_settlement_cycle()

    log(
        "[SETTLEMENT CYCLE] "
        f"status={result.status.value} "
        f"reason_code={result.reason_code} "
        f"pending_checked={result.pending_checked} "
        f"settled={result.settled} "
        f"still_pending={result.still_pending} "
        f"no_action={result.no_action} "
        f"unavailable={result.unavailable} "
        f"errors={result.errors} "
        f"scoreboard_calls={result.scoreboard_calls} "
        f"gamelog_calls={result.gamelog_calls} "
        f"duration_s={round(result.duration_seconds, 2)}"
    )
    log(f"[SETTLEMENT CYCLE] message: {result.message}")
    log(f"[SETTLEMENT CYCLE] Runtime: {round(time.time() - start_time, 2)} seconds")
    log("[SETTLEMENT CYCLE] ===== END =====")

    return 0 if result.status in _NON_ALARMING else 1


if __name__ == "__main__":
    sys.exit(main())
