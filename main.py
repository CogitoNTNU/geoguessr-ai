import os
import sqlite3
import pandas as pd
from dataclasses import dataclass

DATABASE_PATH = os.path.join("sqlite", "dataset.sqlite")


@dataclass
class RunConfig:
    include_image: bool = os.getenv("INCLUDE_IMAGE") == "1"
    read_limit: int = int(os.getenv("READ_LIMIT", "1000"))
    read_where: str = os.getenv("READ_WHERE", "")


def get_database_tables(db_connection: sqlite3.Connection):
    try:
        print("Connected to SQLite")

        # Getting all tables from sqlite_master
        sql_query = """SELECT name FROM sqlite_master 
        WHERE type='table';"""

        # Creating cursor object using connection object
        cursor = db_connection.cursor()

        # executing our sql query
        cursor.execute(sql_query)
        print("List of tables\n")

        # printing all tables list
        print(cursor.fetchall())

    except sqlite3.Error as error:
        print("Failed to execute the above query", error)


def main(config: RunConfig) -> None:
    print(f"Using database at: {DATABASE_PATH}")
    # Open a connection and ensure it stays open while reading
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Make connection read-only at SQL level (prevents writes in this session)
        conn.execute("PRAGMA query_only = 1")

        # Show available tables
        get_database_tables(conn)

        # Detect available columns in 'samples'
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(samples)")
        available_cols = [row[1] for row in cur.fetchall()]

        # Performance-oriented pragmas (best-effort; ignore if unsupported)
        for pragma in (
            "mmap_size=268435456",  # 256MB memory map
            "temp_store=MEMORY",
            "cache_size=-200000",  # ~200MB cache (negative => KB units)
            "synchronous=OFF",  # safe for read-only
        ):
            try:
                conn.execute(f"PRAGMA {pragma}")
            except sqlite3.Error:
                pass

        requested_cols = [
            "location_id",
            "lat",
            "lon",
            "heading",
            "capture_date",
            "pano_id",
            "batch_date",
        ]

        if config.include_image:
            requested_cols.append("image")

        # Only select columns that exist; warn about missing ones
        selected = [c for c in requested_cols if c in available_cols]
        missing = [c for c in requested_cols if c not in available_cols]
        if missing:
            print(
                f"Warning: missing columns in 'samples': {missing}. Reading available columns only: {selected}"
            )
        if not selected:
            print("No requested columns are present in 'samples'. Aborting.")
            return

        # Optional: limit rows via env READ_LIMIT to avoid huge reads
        limit_clause = ""
        read_limit = os.getenv("READ_LIMIT")
        if read_limit:
            try:
                n = int(read_limit)
                if n > 0:
                    limit_clause = f" LIMIT {n}"
            except ValueError:
                pass

        query = f"SELECT {', '.join(selected)} FROM samples{limit_clause}"
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No data found in the 'samples' table.")
        return

    print(f"Loaded {len(df)} records from the 'samples' table.")
    print("Sample records:")
    print(df.head())
    print("Program Done!")


# Start the server
if __name__ == "__main__":
    config = RunConfig()
    main(config)
