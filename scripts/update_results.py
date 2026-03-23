import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared_app import update_all_pending_sheet_results

def log(msg):
    print(msg, flush=True)

if __name__ == "__main__":
    result = update_all_pending_sheet_results(debug=False)

    if isinstance(result, tuple):
        if len(result) == 3:
            updated_count, checked_count, debug_info = result
            print(f"Checked {checked_count} rows, updated {updated_count} rows.", flush=True)
            print(debug_info, flush=True)
        elif len(result) == 2:
            updated_count, checked_count = result
            print(f"Checked {checked_count} rows, updated {updated_count} rows.", flush=True)
        else:
            print(result, flush=True)
    else:
        print(result, flush=True)
