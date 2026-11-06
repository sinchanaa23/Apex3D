import sys
from pathlib import Path

# lets plain `pytest` (not only `python -m pytest`) import backend/ and notebooks/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
