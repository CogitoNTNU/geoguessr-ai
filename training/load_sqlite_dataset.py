import os
import sqlite3
import tempfile
from typing import Dict, Iterator, Optional

import pandas as pd
from pathlib import Path

# Reuse S3 client and helpers from backend.s3bucket
from backend.s3bucket import (
    BUCKET,
    DATASET_SQLITE_PREFIX,
    get_json,
    s3,
)


def _resolve_sqlite_local_path(sqlite_path: Optional[str] = None) -> str:
    """
    Resolve a local filesystem path to the dataset SQLite file.
    If `sqlite_path` starts with 's3://', download it to a temp file and return that path.
    If `sqlite_path` is None, use the S3 pointer dataset_sqlite/_latest.json and download.
    """
    if sqlite_path and not sqlite_path.startswith("s3://"):
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")
        return sqlite_path

    if sqlite_path and sqlite_path.startswith("s3://"):
        bucket, key = sqlite_path[5:].split("/", 1)
    else:
        ptr = get_json(BUCKET, f"{DATASET_SQLITE_PREFIX}/_latest.json")
        if not ptr or "s3" not in ptr:
            raise FileNotFoundError("Latest SQLite pointer not found in S3.")
        base = ptr["s3"]
        if not base.endswith("/"):
            base = base + "/"
        bucket, key = base[5:].split("/", 1)
        key = key + "dataset.sqlite"

    td = tempfile.mkdtemp(prefix="sqlite_ds_")
    local_path = os.path.join(td, "dataset.sqlite")
    s3.download_file(bucket, key, local_path)
    return local_path


def _connect_readonly(local_db_path: str) -> sqlite3.Connection:
    """
    Open the SQLite database strictly read-only to avoid creating -wal/-shm files.
    """
    ro_uri = f"{Path(local_db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(ro_uri, uri=True)
    conn.execute("PRAGMA query_only = 1")
    return conn


def load_sqlite_dataset(
    sqlite_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the SQLite dataset produced by backend.s3bucket.create_and_upload_sqlite_from_latest_snapshot.
    Returns a pandas DataFrame containing the entire 'samples' table.

    Args:
      sqlite_path: Optional path to the SQLite file. Can be:
        - local path (/.../dataset.sqlite)
        - s3 URL (s3://bucket/path/to/dataset.sqlite)
        - None to auto-resolve the latest from S3 pointer
    """
    local_db_path = _resolve_sqlite_local_path(sqlite_path)

    conn = _connect_readonly(local_db_path)
    try:
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

    # Normalize potential memoryview in BLOB column to bytes
    if "image" in df.columns:
        df["image"] = df["image"].apply(lambda x: bytes(x) if isinstance(x, memoryview) else x)
    return df


__all__ = ["load_sqlite_dataset"]


