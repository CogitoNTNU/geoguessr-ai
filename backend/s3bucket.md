# S3 Bucket Code Documentation

Lines 453 onwards from `s3bucket.py`

______________________________________________________________________

## Functions Overview

This document covers the main functions starting from line 453:

- `create_and_upload_sqlite_from_latest_snapshot()`
- `create_and_upload_sqlite_clip_embeddings_from_latest_snapshot()`
- `create_and_upload_sqlite_tinyvit_embeddings_from_latest_snapshot()`
- `main()`

______________________________________________________________________

## `create_and_upload_sqlite_from_latest_snapshot()`

**Purpose:** Builds SQLite database from latest snapshot and uploads to S3. Stores raw JPEG bytes in the database.

**Parameters:**

- `max_rows` - Limits number of rows processed (None = all rows)
- `commit_interval` - Number of rows before committing to DB (default 1000)
- `num_workers` - Number of parallel threads for downloading images (default 64)
- `writer_batch_size` - Number of rows to batch before inserting to DB (default 1000)
- `fetch_window_size` - Number of rows to process per window to control memory (default 10000)

**Returns:** Dict with `sqlite_key`, `local_sqlite_path`, `rows`, `run_id`

______________________________________________________________________
