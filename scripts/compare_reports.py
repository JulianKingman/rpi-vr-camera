#!/usr/bin/env python3
"""Compare two metrics reports produced by the web client's Rec button.

Usage:
    python scripts/compare_reports.py reports/report-A.json reports/report-B.json

Prints a side-by-side table of each metric's p50/p95 with deltas (B - A).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "summary" not in data:
        raise SystemExit(f"{path}: no 'summary' key — not a recorder report?")
    return data


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    args = parser.parse_args()

    a = load(args.report_a)
    b = load(args.report_b)

    print(f"A: {args.report_a.name}  ({a['meta'].get('date', '?')})")
    print(f"B: {args.report_b.name}  ({b['meta'].get('date', '?')})")
    print()

    keys = sorted(set(a["summary"]) | set(b["summary"]))
    header = f"{'metric':<18} {'A p50':>8} {'B p50':>8} {'Δp50':>8}   {'A p95':>8} {'B p95':>8} {'Δp95':>8}"
    print(header)
    print("-" * len(header))
    for key in keys:
        sa = a["summary"].get(key) or {}
        sb = b["summary"].get(key) or {}
        d50 = (sb["p50"] - sa["p50"]) if ("p50" in sa and "p50" in sb) else None
        d95 = (sb["p95"] - sa["p95"]) if ("p95" in sa and "p95" in sb) else None
        print(
            f"{key:<18} {fmt(sa.get('p50')):>8} {fmt(sb.get('p50')):>8} {fmt(d50):>8}   "
            f"{fmt(sa.get('p95')):>8} {fmt(sb.get('p95')):>8} {fmt(d95):>8}"
        )


if __name__ == "__main__":
    main()
