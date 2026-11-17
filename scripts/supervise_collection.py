#!/usr/bin/env python3
"""Run the external VoxWM collection supervisor from the repository root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vae_data.supervisor import main


if __name__ == "__main__":
    raise SystemExit(main())
