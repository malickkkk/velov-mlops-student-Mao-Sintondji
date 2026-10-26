"""Load the canonical CSV into PostgreSQL Bronze without creating duplicates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg


def ingest(csv_path: Path, database_url: str) -> int:
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"])
    required = {"station_id", "timestamp", "capacity", "bikes_available", "temperature", "is_raining"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {', '.join(sorted(missing))}")

    frame = frame.drop_duplicates(subset=["station_id", "timestamp"])
    url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS bronze")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bronze.velov_history (
                    station_id INTEGER NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    capacity INTEGER NOT NULL,
                    bikes_available INTEGER NOT NULL,
                    temperature DOUBLE PRECISION NOT NULL,
                    is_raining BOOLEAN NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (station_id, timestamp)
                )
                """
            )
            rows = [
                (
                    int(row.station_id),
                    row.timestamp.to_pydatetime(),
                    int(row.capacity),
                    int(row.bikes_available),
                    float(row.temperature),
                    bool(row.is_raining),
                )
                for row in frame.itertuples(index=False)
            ]
            cursor.executemany(
                """
                INSERT INTO bronze.velov_history
                    (station_id, timestamp, capacity, bikes_available, temperature, is_raining)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (station_id, timestamp) DO NOTHING
                """,
                rows,
            )
            cursor.execute("SELECT COUNT(*) FROM bronze.velov_history")
            return int(cursor.fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/velov_history.csv"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://velov:secret@localhost:5432/velov"))
    args = parser.parse_args()
    print(f"{ingest(args.csv, args.database_url)} lignes présentes dans bronze.velov_history")


if __name__ == "__main__":
    main()
