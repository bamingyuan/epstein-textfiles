from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import DB_PATH, initialize_search_objects  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SQLite indexes and FTS5 table for faster document search."
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="Path to sqlite database (default: %(default)s)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        raise SystemExit(f"Database not found: {db_path}")

    print(f"Initializing indexes and FTS on: {db_path}")
    initialize_search_objects(db_path)
    print("Done.")


if __name__ == "__main__":
    main()
