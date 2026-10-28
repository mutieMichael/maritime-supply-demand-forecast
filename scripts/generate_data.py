#!/usr/bin/env python3
"""Write data/synthetic from the seeded generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandlery_forecast.pipeline import generate


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
