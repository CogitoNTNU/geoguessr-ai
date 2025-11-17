import os
import pandas as pd
import sqlite3

DATABASE_PATH = os.path.join("sqlite", "dataset.sqlite")


def main() -> None:
    print(f"Using database at: {DATABASE_PATH}")
    database_path = DATABASE_PATH + "?mode=ro"
    # Connect to the SQLite database in read-only mode
    conn = sqlite3.connect(database_path, uri=True)
    conn.execute("PRAGMA query_only = 1")

    try:
        # Execute a query to fetch all records from the 'samples' table
        df = pd.read_sql_query(
            """
            SELECT
              location_id,
              lat,
              lon,
              heading,
              capture_date,
              pano_id,
              batch_date,
              image
            FROM samples
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("No data found in the 'samples' table.")
        return

    print(f"Loaded {len(df)} records from the 'samples' table.")
    print("Sample records:")
    print(df.head())


# Start the server
if __name__ == "__main__":
    main()
