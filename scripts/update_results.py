import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.shared_app import update_all_pending_sheet_results


if __name__ == "__main__":
    updated_count, checked_count, debug_info = update_all_pending_sheet_results(debug=False)
    print(f"Checked {checked_count} pending rows, updated {updated_count} rows.", flush=True)

    if debug_info:
        print("Debug summary:", flush=True)
        for key, value in debug_info.items():
            print(f"  {key}: {value}", flush=True)
