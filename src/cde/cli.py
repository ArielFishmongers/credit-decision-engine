"""Command-line entry point. One subcommand per pipeline stage."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cde", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("schema", help="print the Release 47 column layout")
    args = parser.parse_args(argv)

    if args.stage == "schema":
        from cde.data.schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS, RELEASE

        print(f"Freddie Mac SFLD Release {RELEASE}")
        for label, cols in (
            ("ORIGINATION", ORIGINATION_COLUMNS),
            ("PERFORMANCE", PERFORMANCE_COLUMNS),
        ):
            print(f"\n{label} ({len(cols)} columns)")
            for i, (name, official) in enumerate(cols, 1):
                print(f"  {i:2d}  {name:<55} {official}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
