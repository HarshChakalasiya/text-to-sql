#!/usr/bin/env python3
"""Create tables and seed synthetic SCF data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.models import Base  # noqa: E402
from app.db.seed import seed_database  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402


def main() -> None:
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        counts = seed_database(db)
        print("Seeded:", counts)
    finally:
        db.close()
    print("Done.")


if __name__ == "__main__":
    main()