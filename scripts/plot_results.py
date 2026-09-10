#!/usr/bin/env python3
"""Plot one or more sht_bench JSON result files."""

from __future__ import annotations

import sys

from sht_bench.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["plot", *sys.argv[1:]]))
