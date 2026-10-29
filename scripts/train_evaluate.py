#!/usr/bin/env python3
"""Fit models, run the rolling-origin backtest, and refresh reports/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chandlery_forecast.pipeline import train_and_evaluate


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
