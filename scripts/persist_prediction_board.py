"""
Step 10: production entry point for "generate and persist today's
board". Thin CLI wrapper around
src.services.orchestration.run_prediction_cycle -- see that module for
all the actual decision-making (schema readiness, FAILED-vs-NO_WORK
disambiguation, pregame cutoff filtering).

Exit codes (Step 10 Phase 8): 0 for SUCCESS/NO_WORK/PARTIAL (the job ran
correctly -- NO_WORK/PARTIAL are expected, non-alarming outcomes), 1 for
FAILED (a real operational problem). This is what a scheduler/workflow
should key off of, not the printed text.

Requires ODDS_API_KEY and DATABASE_URL (same as Step 9's version of
this script). Safe to run manually at any time for incident recovery --
see the Step 10 report's manual-recovery section.
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.orchestration import CycleStatus, run_prediction_cycle

_NON_ALARMING = (CycleStatus.SUCCESS, CycleStatus.NO_WORK, CycleStatus.PARTIAL)


def log(msg):
    print(msg, flush=True)


def main() -> int:
    start_time = time.time()
    log("[PREDICTION CYCLE] ===== START =====")

    result = run_prediction_cycle()

    log(
        "[PREDICTION CYCLE] "
        f"status={result.status.value} "
        f"reason_code={result.reason_code} "
        f"props_discovered={result.props_discovered} "
        f"predictions_generated={result.predictions_generated} "
        f"persisted={result.persisted_count} "
        f"excluded_live={result.excluded_live_count} "
        f"excluded_final={result.excluded_final_count} "
        f"excluded_postponed_or_canceled={result.excluded_postponed_or_canceled_count} "
        f"unmatched={result.unmatched_count} "
        f"unavailable={result.unavailable_count} "
        f"provider_calls={result.provider_calls} "
        f"run_id={result.run_id} "
        f"run_created={result.run_created} "
        f"duration_s={round(result.duration_seconds, 2)}"
    )
    log(f"[PREDICTION CYCLE] message: {result.message}")
    log(f"[PREDICTION CYCLE] Runtime: {round(time.time() - start_time, 2)} seconds")
    log("[PREDICTION CYCLE] ===== END =====")

    return 0 if result.status in _NON_ALARMING else 1


if __name__ == "__main__":
    sys.exit(main())
