import os
import sys

# Ensure the repository root is importable as `src.*` regardless of the
# directory pytest is invoked from -- mirrors the sys.path handling already
# used by apps/*.py and scripts/*.py in this repository.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
