import sys
from pathlib import Path

# Ensure the repo root is on sys.path so tests can import top-level packages like `models`, `retrieval`, `agents`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
