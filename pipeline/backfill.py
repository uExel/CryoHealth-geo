"""CLI entrypoint for historical population: uv run python -m pipeline.backfill --start 2023-01-01

Separate from service.py's /run on purpose — a backfill is a long-running, bounded,
one-shot operation an operator drives from a terminal, not something a web request
should trigger (Process API quota + wall-clock time scale with the date range and lake
count; see docs/ai/PLAN.md for why the full historical run needs to stay a deliberate,
narrated action rather than a button).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from pipeline.batch import run_backfill
from pipeline.lakes import LAKES


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill historical Observation rows over a date range.")
    parser.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD, inclusive")
    parser.add_argument(
        "--end", type=_parse_date, default=None, help="YYYY-MM-DD, inclusive (default: today)"
    )
    parser.add_argument(
        "--lakes",
        nargs="+",
        choices=list(LAKES),
        default=None,
        help="Lake slugs to backfill (default: all seeded lakes)",
    )
    parser.add_argument(
        "--source",
        choices=["cdse", "planetary-computer"],
        default="cdse",
        help="Scene source (default: cdse, the production source)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    if args.source == "cdse":
        from pipeline.cdse_source import CdseSource

        source = CdseSource()
    else:
        from pipeline.stac_source import PlanetaryComputerSource

        source = PlanetaryComputerSource()

    end = args.end or date.today()
    logger.info(
        "backfill starting: lakes=%s range=%s..%s source=%s",
        args.lakes or list(LAKES),
        args.start,
        end,
        args.source,
    )

    results = run_backfill(source, args.start, end, args.lakes)

    total_written = sum(r.observations_written for r in results)
    total_checked = sum(r.scenes_checked for r in results)
    for r in results:
        logger.info(
            "  %-12s written=%-4d checked=%-4d stale=%-5s error=%s",
            r.slug, r.observations_written, r.scenes_checked, r.stale, r.error or "-",
        )
    logger.info(
        "backfill done: %d/%d scenes written across %d lake(s)", total_written, total_checked, len(results)
    )

    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
