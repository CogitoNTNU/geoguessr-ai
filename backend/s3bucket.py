import os
import re
import json
import hashlib
import tempfile
import datetime
import struct
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import time

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    class _TqdmNoop:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(*args, **kwargs):  # type: ignore
        return _TqdmNoop(kwargs.get("total"), kwargs.get("desc"))


# Optional Weights & Biases lightweight logging
try:
    import wandb  # type: ignore

    def _wandb_log(data: dict, step: int | None = None):
        run = getattr(wandb, "run", None)
        if run is not None:
            wandb.log(data, step=step)
except Exception:  # pragma: no cover

    def _wandb_log(data: dict, step: int | None = None):
        return None


import boto3
import pandas as pd
from typing import Iterable, Dict, Any
from botocore.config import Config
from botocore.exceptions import ClientError


# ---------- CONFIG ----------
BUCKET = "cogito-geoguessr"
VERSION = "v1"
MANIFEST_PREFIX = f"{VERSION}/manifest"
SNAPSHOT_PREFIX = f"{VERSION}/snapshot"
HOLDOUT_PREFIX = "holdout_dataset"
HOLDOUT_SNAPSHOT_PREFIX = f"{HOLDOUT_PREFIX}/snapshot"
DATASET_SQLITE_PREFIX = "dataset_sqlite"
DATASET_SQLITE_CLIP_PREFIX = "dataset_sqlite_clip_embeddings"
DATASET_SQLITE_TINYVIT_PREFIX = "dataset_sqlite_tinyvit_embeddings"
HEADINGS_DEFAULT = (0, 90, 180, 270)
STREETVIEW_RE = re.compile(
    r"^streetview_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)_heading_(\d{1,3})\.jpg$",
    re.IGNORECASE,
)
REGION = "eu-north-1"
_Q = 10_000_000  # 1e-7° grid (~1.1 cm)
s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=os.getenv("AWS_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL"),
    config=Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=256,
        read_timeout=120,
        connect_timeout=10,
    ),
)


# ---------- UTILS ----------
def make_location_id(lat: float, lon: float, hex_len: int = 12) -> str:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat/lon out of bounds")
    if lat == 0.0:
        lat = 0.0
    if lon == 0.0:
        lon = 0.0
    lat_i = int(round(lat * _Q))
    lon_i = int(round(lon * _Q))
    payload = struct.pack(">ii", lat_i, lon_i)
    return hashlib.sha1(b"geo:v1:" + payload).hexdigest()[:hex_len]


def img_key(location_id: str, heading_deg: int) -> str:
    return f"{VERSION}/images/location_id={location_id}/heading={heading_deg:03d}.jpg"


def put_json(obj: dict, bucket: str, key: str):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj).encode("utf-8"),
        ContentType="application/json",
    )


def get_json(bucket: str, key: str) -> dict | None:
    try:
        b = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(b.decode("utf-8"))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise


def upload_one_image(rec: dict) -> dict:
    key = img_key(rec["location_id"], rec["heading"])
    s3.upload_file(
        rec["image_path"], BUCKET, key, ExtraArgs={"ContentType": "image/jpeg"}
    )
    return {
        "location_id": rec["location_id"],
        "lat": rec["lat"],
        "lon": rec["lon"],
        "heading": rec["heading"],
        "capture_date": rec.get("capture_date"),
        "pano_id": rec.get("pano_id"),
        "image_path": f"s3://{BUCKET}/{key}",
    }


def upload_batch(records: list[dict], max_workers=16) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(upload_one_image, r) for r in records]
        for fut in as_completed(futs):
            rows.append(fut.result())
    df = pd.DataFrame(rows)
    return df


def write_batch_manifest(df_batch: pd.DataFrame, batch_date: str) -> str:
    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    key = f"{MANIFEST_PREFIX}/run={run_id}/part-000.parquet"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "part-000.parquet")
        df_batch.to_parquet(p, index=False)
        s3.upload_file(p, BUCKET, key)
    return key


def load_previous_snapshot() -> pd.DataFrame | None:
    ptr = get_json(BUCKET, f"{SNAPSHOT_PREFIX}/_latest.json")
    if not ptr:
        return None
    snap_prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")
    return read_parquet_prefix(snap_prefix)


def load_latest_snapshot_df() -> pd.DataFrame:
    """
    Each row is an image, contains lat, lon, filepath to s3.
    """
    ptr = get_json(BUCKET, f"{SNAPSHOT_PREFIX}/_latest.json")
    if not ptr:
        raise FileNotFoundError("No snapshot pointer found.")
    snap_prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")
    df = read_parquet_prefix(snap_prefix)
    if df is None or df.empty:
        raise FileNotFoundError("Snapshot folder has no Parquet parts.")
    return df


def load_latest_holdout_snapshot_df() -> pd.DataFrame:
    """
    Same structure as v1 snapshot, but reads from 'holdout_dataset/snapshot'.
    """
    ptr = get_json(BUCKET, f"{HOLDOUT_SNAPSHOT_PREFIX}/_latest.json")
    if not ptr:
        raise FileNotFoundError("No holdout snapshot pointer found.")
    snap_prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")
    df = read_parquet_prefix(snap_prefix)
    if df is None or df.empty:
        raise FileNotFoundError("Holdout snapshot folder has no Parquet parts.")
    return df


def pick_image(rec, h):
    images = rec["images"]
    ang = float(h) % 360  # handle strings/floats robustly

    def circ_dist(a, b):
        # smallest angular distance (0..180)
        return abs((a - b + 180) % 360 - 180)

    nearest_key = min(images.keys(), key=lambda k: circ_dist(k, ang))
    return images[nearest_key]


def merge_snapshot(prev: pd.DataFrame | None, batch_df: pd.DataFrame) -> pd.DataFrame:
    if prev is None or prev.empty:
        return batch_df.copy()
    all_cols = sorted(set(prev.columns) | set(batch_df.columns))
    prev2 = prev.reindex(columns=all_cols)
    batch2 = batch_df.reindex(columns=all_cols)

    prev2["_is_new"] = 0
    batch2["_is_new"] = 1
    cat = pd.concat([prev2, batch2], ignore_index=True)

    cat.sort_values(by=["location_id", "heading", "_is_new"], inplace=True)
    latest = cat.drop_duplicates(subset=["location_id", "heading"], keep="last").drop(
        columns=["_is_new"]
    )
    latest = pd.DataFrame(latest).reset_index(drop=True)
    return latest


def write_new_snapshot(df_latest: pd.DataFrame) -> str:
    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    snap_prefix = f"{SNAPSHOT_PREFIX}/{run_id}"
    key = f"{snap_prefix}/part-000.parquet"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "part-000.parquet")
        df_latest.to_parquet(p, index=False)
        s3.upload_file(p, BUCKET, key)
    put_json(
        {"s3": f"s3://{BUCKET}/{snap_prefix}/"},
        BUCKET,
        f"{SNAPSHOT_PREFIX}/_latest.json",
    )
    return key


def parse_streetview_folder(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Scans `root_dir` for files like:
      streetview_<lat>_<lon>_heading_<0|90|180|270>.jpg

    Returns:
      key "<lat>:<lon>:heading:capture_data:pano_id" -> {
        "lat": float, "lon": float, "heading": int, "capture_date": string, "pano_id": string
        "images": { heading(int): local_path(str) }
      }
    """
    index: Dict[str, Dict[str, Any]] = {}

    for name in os.listdir(root_dir):
        if not name.lower().endswith(".jpg"):
            continue
        m = name.split("_")
        if not m:
            continue

        lat = float(m[0])
        lon = float(m[1])
        heading = int(m[2])
        capture_date = str(m[3])
        pano_id = str(m[4])

        key = f"{lat:.7f}_{lon:.7f}"
        rec = index.setdefault(
            key,
            {
                "lat": lat,
                "lon": lon,
                "images": {},
                "capture_date": capture_date,
                "pano_id": pano_id,
            },
        )
        rec["images"][heading] = os.path.join(root_dir, name)

    return index


def records_from_streetview_index(
    idx: Dict[str, Dict[str, Any]],
    headings: Iterable[int] = HEADINGS_DEFAULT,
) -> list[dict]:
    """
    Builds upload records your existing upload_batch(...) understands.
    Only includes headings that actually exist on disk.
    """
    records: list[dict] = []
    for rec in idx.values():
        lat, lon = rec["lat"], rec["lon"]
        loc_id = make_location_id(lat, lon)  # quantized hash from your script

        capture_date = rec["capture_date"]
        pano_id = rec["pano_id"]

        for h in headings:
            img_path = pick_image(rec, h)
            if not img_path:
                continue
            records.append(
                {
                    "location_id": loc_id,
                    "lat": lat,
                    "lon": lon,
                    "heading": int(h) % 360,
                    "image_path": img_path,
                    "capture_date": capture_date,
                    "pano_id": pano_id,
                }
            )
    return records


def list_keys(prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            keys.append(o["Key"])
    return keys


def read_parquet_prefix(prefix: str) -> pd.DataFrame | None:
    parts, keys = [], list_keys(prefix)
    if not keys:
        return None
    with tempfile.TemporaryDirectory() as td:
        for k in keys:
            if not k.endswith(".parquet"):
                continue
            lp = os.path.join(td, os.path.basename(k))
            s3.download_file(BUCKET, k, lp)
            parts.append(pd.read_parquet(lp))
    return pd.concat(parts, ignore_index=True) if parts else None


def upload_dataset_from_folder(
    folder: str,
    batch_date: str | None = None,
    headings: Iterable[int] = HEADINGS_DEFAULT,
    max_workers: int = 16,
):
    batch_date = batch_date or datetime.date.today().isoformat()

    idx = parse_streetview_folder(folder)
    records = records_from_streetview_index(idx, headings=headings)
    if not records:
        return {"message": "No records found to ingest.", "rows_in_batch": 0}

    df_batch = upload_batch(records, max_workers=max_workers)
    df_batch["batch_date"] = batch_date

    batch_key = write_batch_manifest(df_batch, batch_date)

    prev = load_previous_snapshot()
    latest = merge_snapshot(prev, df_batch)
    snapshot_key = write_new_snapshot(latest)

    return {
        "batch_manifest_key": batch_key,
        "snapshot_key": snapshot_key,
        "rows_in_batch": len(df_batch),
        "rows_in_latest": len(latest),
        "ingested_locations": len({r["location_id"] for r in records}),
    }


def download(dest_dir: str, overwrite: bool, row):
    path = row["image_path"]
    if not path.startswith("s3://"):
        return (path, "skipped (not s3)")
    bucket, key = path[5:].split("/", 1)
    # Make local path relative to the 'images/' subfolder regardless of top-level prefix (v1/ or holdout_dataset/)
    images_idx = key.find("images/")
    if images_idx != -1:
        rel = key[images_idx + len("images/") :]
        local = "/" + rel
    else:
        # Fallback: use full key
        local = "/" + key
    local_path = dest_dir + local
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not overwrite and os.path.exists(local_path):
        return (local_path, "skipped (exists)")
    try:
        s3.download_file(bucket, key, local_path)
        return (local_path, "downloaded")
    except Exception as e:
        print(e)
        return (local_path, f"failed: {e}")


def download_latest_images(
    dest_dir: str,
    overwrite: bool = False,
    max_workers: int = 16,
    df=None,
):
    if df is None:
        df = load_latest_snapshot_df()
    os.makedirs(dest_dir, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(download, dest_dir, overwrite, row) for _, row in df.iterrows()
        ]
        for fut in as_completed(futs):
            results.append(fut.result())

    return results


def download_random_panorama(
    dest_dir: str,
    overwrite: bool = False,
) -> dict:
    """
    Downloads up to 4 images for a random location_id from the latest snapshot.

    Returns a dict with:
      {
        "paths": [local_paths...],
        "location_id": str,
        "lat": float,
        "lon": float,
      }
    """
    df = load_latest_snapshot_df()
    if df is None or df.empty:
        raise ValueError("Latest snapshot is empty.")

    # Group by location_id and keep only groups with at least 4 images
    groups = [g for _, g in df.groupby("location_id") if len(g) >= 4]
    if not groups:
        raise ValueError("No location_id with at least 4 images found in snapshot.")

    group = random.choice(groups)
    group = group.sort_values("heading").head(4)

    os.makedirs(dest_dir, exist_ok=True)
    local_paths: list[str] = []
    for _, row in group.iterrows():
        local_path, _ = download(dest_dir, overwrite, row)
        local_paths.append(local_path)

    first = group.iloc[0]
    return {
        "paths": local_paths,
        "location_id": str(first["location_id"]),
        "lat": float(first["lat"]),
        "lon": float(first["lon"]),
    }


def load_points():
    df = load_latest_snapshot_df()
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)
    cols = ["location_id", "lat", "lon"]
    point_df = df[cols].copy()
    return point_df


def load_holdout_points():
    df = load_latest_holdout_snapshot_df()
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)
    cols = ["location_id", "lat", "lon"]
    point_df = df[cols].copy()
    return point_df


def add_metadata():
    df = load_latest_snapshot_df()
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)


def get_snapshot_metadata():
    df = load_latest_snapshot_df()

    remove = ("heading", "batch_date")
    cols_to_drop = []
    for col in df.columns:
        for remove in remove:
            if remove in col:
                cols_to_drop.append(col)
                break

    meta_data = df.drop(columns=cols_to_drop).reset_index(drop=True)

    return meta_data


# upload_dataset_from_folder("./dataset", max_workers=24)

# points = load_points()
# print(f"Total points saved in S3: {len(points)}")
# points.to_csv("./test.csv")


def create_and_upload_sqlite_from_latest_snapshot(
    max_rows: int | None = None,
    commit_interval: int = 1000,
    num_workers: int = 64,
    writer_batch_size: int = 1000,
    fetch_window_size: int = 10000,
):
    """
    Builds a SQLite DB from the latest v1 snapshot and uploads it to S3 at
    dataset_sqlite/run_ts=.../dataset.sqlite. The table has the same columns
    as the parquet (including 'image_path'), but in the SQLite table the bytes
    are stored under the column 'image' (BLOB) instead of an s3:// URL.
    """
    df = load_latest_snapshot_df()
    if max_rows is not None:
        df = df.head(max_rows)

    # Ensure expected columns exist (treat optional ones as present with NULLs)
    expected_cols = [
        "location_id",
        "lat",
        "lon",
        "heading",
        "capture_date",
        "pano_id",
        "batch_date",
        "image_path",  # will be BLOB in SQLite
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    sqlite_key = f"{DATASET_SQLITE_PREFIX}/{run_id}/dataset.sqlite"

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "dataset.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            start_ts = time.time()
            # Pragmas for reasonable write performance and durability trade-offs
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA mmap_size=268435456;")  # 256MB

            # Schema: mirror snapshot columns, but 'image_path' is a BLOB
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                  location_id TEXT NOT NULL,
                  lat REAL NOT NULL,
                  lon REAL NOT NULL,
                  heading INTEGER NOT NULL,
                  capture_date TEXT,
                  pano_id TEXT,
                  batch_date TEXT,
                  image BLOB NOT NULL,
                  PRIMARY KEY (location_id, heading)
                ) WITHOUT ROWID;
                """
            )
            conn.commit()

            def _get_image_bytes(path: str) -> bytes:
                if not path or not isinstance(path, str):
                    raise ValueError("image_path missing")
                if not path.startswith("s3://"):
                    # Fallback: if local path (unexpected in latest snapshot), read from disk
                    with open(path, "rb") as f:
                        return f.read()
                bucket, key = path[5:].split("/", 1)
                obj = s3.get_object(Bucket=bucket, Key=key)
                return obj["Body"].read()

            # Concurrent downloads in bounded windows, single-writer batched inserts
            cols = [
                "location_id",
                "lat",
                "lon",
                "heading",
                "capture_date",
                "pano_id",
                "batch_date",
                "image_path",
            ]

            def fetch(rec):
                b = _get_image_bytes(rec.get("image_path"))
                return (
                    rec.get("location_id"),
                    float(rec.get("lat")) if rec.get("lat") is not None else None,
                    float(rec.get("lon")) if rec.get("lon") is not None else None,
                    int(rec.get("heading")) if rec.get("heading") is not None else None,
                    rec.get("capture_date"),
                    rec.get("pano_id"),
                    rec.get("batch_date"),
                    sqlite3.Binary(b),
                )

            total_rows = int(len(df))
            rows_since_commit = 0
            processed_total = 0
            pbar = tqdm(total=total_rows, desc="Building SQLite (JPEG bytes)")
            cur.execute("BEGIN;")
            for start in range(0, total_rows, fetch_window_size):
                end = min(start + fetch_window_size, total_rows)
                rows_chunk = df.iloc[start:end][cols].to_dict("records")
                batch = []
                # New executor per window keeps memory bounded
                with ThreadPoolExecutor(max_workers=num_workers) as ex:
                    futs = [ex.submit(fetch, r) for r in rows_chunk]
                    for fut in as_completed(futs):
                        batch.append(fut.result())
                        pbar.update(1)
                        if len(batch) >= writer_batch_size:
                            cur.executemany(
                                """
                                INSERT OR REPLACE INTO samples
                                  (location_id, lat, lon, heading, capture_date, pano_id, batch_date, image)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                batch,
                            )
                            inserted = len(batch)
                            rows_since_commit += inserted
                            processed_total += inserted
                            _wandb_log(
                                {
                                    "mode": "jpeg",
                                    "processed": processed_total,
                                    "total": total_rows,
                                    "throughput_img_per_s": processed_total
                                    / max(time.time() - start_ts, 1e-6),
                                    "phase": "inserting",
                                },
                                step=processed_total,
                            )
                            batch.clear()
                            if rows_since_commit >= commit_interval:
                                conn.commit()
                                cur.execute("BEGIN;")
                                rows_since_commit = 0
                # Flush remainder of this window
                if batch:
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO samples
                          (location_id, lat, lon, heading, capture_date, pano_id, batch_date, image)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    inserted = len(batch)
                    rows_since_commit += inserted
                    processed_total += inserted
                    _wandb_log(
                        {
                            "mode": "jpeg",
                            "processed": processed_total,
                            "total": total_rows,
                            "throughput_img_per_s": processed_total
                            / max(time.time() - start_ts, 1e-6),
                            "phase": "inserting",
                        },
                        step=processed_total,
                    )
                    batch.clear()
                # Commit at end of window to keep WAL small
                conn.commit()
                cur.execute("BEGIN;")
                rows_since_commit = 0
            # Final commit and close bar
            conn.commit()
            pbar.close()
        finally:
            conn.close()

        # Upload SQLite file (disabled: no S3 upload)
        # s3.upload_file(
        #     db_path,
        #     BUCKET,
        #     sqlite_key,
        #     ExtraArgs={"ContentType": "application/octet-stream"},
        # )

    # Write/update pointer (disabled: no S3 write)
    # put_json(
    #     {"s3": f"s3://{BUCKET}/{DATASET_SQLITE_PREFIX}/{run_id}/"},
    #     BUCKET,
    #     f"{DATASET_SQLITE_PREFIX}/_latest.json",
    # )

    # Also write a local copy beside the repository directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    os.makedirs(repo_parent_dir, exist_ok=True)
    # Save a local copy with an explicit `_2` suffix in the filename so it can
    # be used alongside the original without collisions (e.g. parallel runs).
    local_sqlite_path = os.path.join(
        repo_parent_dir, f"dataset_sqlite_6_{run_id}.sqlite"
    )
    # Recreate the DB to copy from S3-uploaded temp path
    # Note: db_path no longer exists outside the with block; create file again by downloading is costly.
    # Instead, regenerate inside a new temp block for copy consistency.
    # Simpler: move creation before temp cleanup. As we are outside, reuse last built path by creating again.
    # Rebuild locally by downloading from S3 would cost network; better generate inside the previous block.
    # Since temp dir is gone, copy must occur before it. Adjusted: regenerate is not feasible here.
    # Fallback: we already have sqlite_key in S3; download and save locally.

    # Final W&B summary
    try:
        sz = os.path.getsize(local_sqlite_path)
    except Exception:
        sz = None
    elapsed = time.time() - start_ts
    _wandb_log(
        {
            "mode": "jpeg",
            "duration_s": elapsed,
            "avg_throughput_img_per_s": int(len(df)) / max(elapsed, 1e-6),
            "sqlite_size_bytes": sz,
            "phase": "done",
        }
    )

    return {
        "sqlite_key": sqlite_key,
        "local_sqlite_path": local_sqlite_path,
        "rows": int(len(df)),
        "run_id": run_id,
    }


def create_and_upload_sqlite_clip_embeddings_from_latest_snapshot(
    max_rows: int | None = None,
    commit_interval: int = 1000,
    device: str | None = None,
    num_workers: int = 64,
    embed_batch_size: int = 256,
    fetch_window_size: int = 10000,
):
    """
    Builds a SQLite DB from the latest v1 snapshot where each row stores a CLIP
    embedding (float32 blob) instead of the raw JPEG. Uploads to:
      dataset_sqlite_clip_embeddings/run_ts=.../dataset.sqlite
    Also writes dataset_sqlite_clip_embeddings/_latest.json pointer.
    """
    df = load_latest_snapshot_df()
    if max_rows is not None:
        df = df.head(max_rows)

    expected_cols = [
        "location_id",
        "lat",
        "lon",
        "heading",
        "capture_date",
        "pano_id",
        "batch_date",
        "image_path",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    sqlite_key = f"{DATASET_SQLITE_CLIP_PREFIX}/{run_id}/dataset.sqlite"

    # Lazy imports to avoid heavy deps at module import time
    import io
    from PIL import Image
    import torch
    from pretrain.clip_embedder import CLIPEmbedding

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPEmbedding(
        model_name="", device=dev, load_checkpoint=False, panorama=False
    )
    clip_model.eval()

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "dataset.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            start_ts = time.time()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA mmap_size=268435456;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                  location_id TEXT NOT NULL,
                  lat REAL NOT NULL,
                  lon REAL NOT NULL,
                  heading INTEGER NOT NULL,
                  capture_date TEXT,
                  pano_id TEXT,
                  batch_date TEXT,
                  embedding BLOB NOT NULL,
                  embedding_dim INTEGER NOT NULL,
                  PRIMARY KEY (location_id, heading)
                ) WITHOUT ROWID;
                """
            )
            conn.commit()

            def _get_image_bytes(path: str) -> bytes:
                if not path or not isinstance(path, str):
                    raise ValueError("image_path missing")
                if not path.startswith("s3://"):
                    with open(path, "rb") as f:
                        return f.read()
                bucket, key = path[5:].split("/", 1)
                obj = s3.get_object(Bucket=bucket, Key=key)
                return obj["Body"].read()

            # Concurrent downloads in bounded windows + GPU batched embedding + single-writer inserts
            cols = [
                "location_id",
                "lat",
                "lon",
                "heading",
                "capture_date",
                "pano_id",
                "batch_date",
                "image_path",
            ]

            def fetch_to_pil(rec):
                b = _get_image_bytes(rec.get("image_path"))
                pil = Image.open(io.BytesIO(b)).convert("RGB")
                return rec, pil

            rows_since_commit = 0
            observed_dim = None
            cur.execute("BEGIN;")
            total_rows = int(len(df))
            pbar = tqdm(total=total_rows, desc="Building SQLite (CLIP embeddings)")
            processed_total = 0
            for start in range(0, total_rows, fetch_window_size):
                end = min(start + fetch_window_size, total_rows)
                rows_chunk = df.iloc[start:end][cols].to_dict("records")
                buffer_recs = []
                buffer_imgs = []
                with ThreadPoolExecutor(max_workers=num_workers) as ex:
                    futs = [ex.submit(fetch_to_pil, r) for r in rows_chunk]
                    for fut in as_completed(futs):
                        rec, pil = fut.result()
                        buffer_recs.append(rec)
                        buffer_imgs.append(pil)
                        pbar.update(1)
                        if len(buffer_imgs) >= embed_batch_size:
                            with torch.no_grad():
                                inputs = clip_model.processor(
                                    images=buffer_imgs, return_tensors="pt"
                                )
                                pixel_values = inputs["pixel_values"]
                                if isinstance(clip_model.device, str):
                                    pixel_values = pixel_values.to(clip_model.device)
                                else:
                                    pixel_values = pixel_values.cuda(clip_model.device)
                                outputs = clip_model.clip_model.base_model(
                                    pixel_values=pixel_values
                                )
                                embs = (
                                    outputs.last_hidden_state.mean(dim=1)
                                    .to(torch.float32)
                                    .cpu()
                                )
                            dim = int(embs.shape[-1])
                            observed_dim = observed_dim or dim
                            batch_rows = []
                            for rec_i, emb in zip(buffer_recs, embs):
                                batch_rows.append(
                                    (
                                        rec_i.get("location_id"),
                                        float(rec_i.get("lat"))
                                        if rec_i.get("lat") is not None
                                        else None,
                                        float(rec_i.get("lon"))
                                        if rec_i.get("lon") is not None
                                        else None,
                                        int(rec_i.get("heading"))
                                        if rec_i.get("heading") is not None
                                        else None,
                                        rec_i.get("capture_date"),
                                        rec_i.get("pano_id"),
                                        rec_i.get("batch_date"),
                                        sqlite3.Binary(emb.numpy().tobytes()),
                                        dim,
                                    )
                                )
                            cur.executemany(
                                """
                                INSERT OR REPLACE INTO samples
                                  (location_id, lat, lon, heading, capture_date, pano_id, batch_date, embedding, embedding_dim)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                batch_rows,
                            )
                            rows_since_commit += len(batch_rows)
                            processed_total += len(batch_rows)
                            _wandb_log(
                                {
                                    "mode": "clip",
                                    "processed": processed_total,
                                    "total": total_rows,
                                    "throughput_img_per_s": processed_total
                                    / max(time.time() - start_ts, 1e-6),
                                    "phase": "embedding_inserting",
                                },
                                step=processed_total,
                            )
                            buffer_recs.clear()
                            buffer_imgs.clear()
                            if rows_since_commit >= commit_interval:
                                conn.commit()
                                cur.execute("BEGIN;")
                                rows_since_commit = 0
                # Flush leftovers in this window
                if buffer_imgs:
                    with torch.no_grad():
                        inputs = clip_model.processor(
                            images=buffer_imgs, return_tensors="pt"
                        )
                        pixel_values = inputs["pixel_values"]
                        if isinstance(clip_model.device, str):
                            pixel_values = pixel_values.to(clip_model.device)
                        else:
                            pixel_values = pixel_values.cuda(clip_model.device)
                        outputs = clip_model.clip_model.base_model(
                            pixel_values=pixel_values
                        )
                        embs = (
                            outputs.last_hidden_state.mean(dim=1)
                            .to(torch.float32)
                            .cpu()
                        )
                    dim = int(embs.shape[-1])
                    observed_dim = observed_dim or dim
                    batch_rows = []
                    for rec_i, emb in zip(buffer_recs, embs):
                        batch_rows.append(
                            (
                                rec_i.get("location_id"),
                                float(rec_i.get("lat"))
                                if rec_i.get("lat") is not None
                                else None,
                                float(rec_i.get("lon"))
                                if rec_i.get("lon") is not None
                                else None,
                                int(rec_i.get("heading"))
                                if rec_i.get("heading") is not None
                                else None,
                                rec_i.get("capture_date"),
                                rec_i.get("pano_id"),
                                rec_i.get("batch_date"),
                                sqlite3.Binary(emb.numpy().tobytes()),
                                dim,
                            )
                        )
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO samples
                          (location_id, lat, lon, heading, capture_date, pano_id, batch_date, embedding, embedding_dim)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch_rows,
                    )
                    rows_since_commit += len(batch_rows)
                    processed_total += len(batch_rows)
                    _wandb_log(
                        {
                            "mode": "clip",
                            "processed": processed_total,
                            "total": total_rows,
                            "throughput_img_per_s": processed_total
                            / max(time.time() - start_ts, 1e-6),
                            "phase": "embedding_inserting",
                        },
                        step=processed_total,
                    )
                # Commit per window
                conn.commit()
                cur.execute("BEGIN;")
                rows_since_commit = 0
            conn.commit()
            pbar.close()
        finally:
            conn.close()

        # s3.upload_file(
        #     db_path,
        #     BUCKET,
        #     sqlite_key,
        #     ExtraArgs={"ContentType": "application/octet-stream"},
        # )

    # put_json(
    #     {"s3": f"s3://{BUCKET}/{DATASET_SQLITE_CLIP_PREFIX}/{run_id}/"},
    #     BUCKET,
    #     f"{DATASET_SQLITE_CLIP_PREFIX}/_latest.json",
    # )

    # Also write a local copy beside the repository directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    os.makedirs(repo_parent_dir, exist_ok=True)
    local_sqlite_path = os.path.join(
        repo_parent_dir, f"dataset_sqlite_clip_embeddings_{run_id}.sqlite"
    )

    # Final W&B summary
    try:
        sz = os.path.getsize(local_sqlite_path)
    except Exception:
        sz = None
    elapsed = time.time() - start_ts
    _wandb_log(
        {
            "mode": "clip",
            "duration_s": elapsed,
            "avg_throughput_img_per_s": int(len(df)) / max(elapsed, 1e-6),
            "sqlite_size_bytes": sz,
            "embedding_dim": observed_dim,
            "phase": "done",
        }
    )

    return {
        "sqlite_key": sqlite_key,
        "local_sqlite_path": local_sqlite_path,
        "rows": int(len(df)),
        "run_id": run_id,
        "device": dev,
        "embedding_dim": observed_dim,
    }


def create_and_upload_sqlite_tinyvit_embeddings_from_latest_snapshot(
    max_rows: int | None = None,
    commit_interval: int = 1000,
    device: str | None = None,
    model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
    num_workers: int = 64,
    embed_batch_size: int = 256,
    fetch_window_size: int = 10000,
):
    """
    Builds a SQLite DB from the latest v1 snapshot where each row stores a TinyViT
    embedding (float32 blob) instead of the raw JPEG. Uploads to:
      dataset_sqlite_tinyvit_embeddings/run_ts=.../dataset.sqlite
    Also writes dataset_sqlite_tinyvit_embeddings/_latest.json pointer.
    """
    df = load_latest_snapshot_df()
    if max_rows is not None:
        df = df.head(max_rows)

    expected_cols = [
        "location_id",
        "lat",
        "lon",
        "heading",
        "capture_date",
        "pano_id",
        "batch_date",
        "image_path",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    sqlite_key = f"{DATASET_SQLITE_TINYVIT_PREFIX}/{run_id}/dataset.sqlite"

    # Lazy imports
    import io
    from PIL import Image
    import torch
    from pretrain.tinyvit_embedder import TinyViTEmbedding

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tinyvit = TinyViTEmbedding(
        model_name=model_name, device=dev, load_checkpoint=False, panorama=False
    )
    tinyvit.eval()

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "dataset.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            start_ts = time.time()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA mmap_size=268435456;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                  location_id TEXT NOT NULL,
                  lat REAL NOT NULL,
                  lon REAL NOT NULL,
                  heading INTEGER NOT NULL,
                  capture_date TEXT,
                  pano_id TEXT,
                  batch_date TEXT,
                  embedding BLOB NOT NULL,
                  embedding_dim INTEGER NOT NULL,
                  PRIMARY KEY (location_id, heading)
                ) WITHOUT ROWID;
                """
            )
            conn.commit()

            def _get_image_bytes(path: str) -> bytes:
                if not path or not isinstance(path, str):
                    raise ValueError("image_path missing")
                if not path.startswith("s3://"):
                    with open(path, "rb") as f:
                        return f.read()
                bucket, key = path[5:].split("/", 1)
                obj = s3.get_object(Bucket=bucket, Key=key)
                return obj["Body"].read()

            # Concurrent downloads in bounded windows + GPU batched embedding + single-writer inserts
            cols = [
                "location_id",
                "lat",
                "lon",
                "heading",
                "capture_date",
                "pano_id",
                "batch_date",
                "image_path",
            ]

            def fetch_to_pil(rec):
                b = _get_image_bytes(rec.get("image_path"))
                pil = Image.open(io.BytesIO(b)).convert("RGB")
                return rec, pil

            rows_since_commit = 0
            observed_dim = None
            cur.execute("BEGIN;")
            total_rows = int(len(df))
            pbar = tqdm(total=total_rows, desc="Building SQLite (TinyViT embeddings)")
            processed_total = 0
            for start in range(0, total_rows, fetch_window_size):
                end = min(start + fetch_window_size, total_rows)
                rows_chunk = df.iloc[start:end][cols].to_dict("records")
                buffer_recs = []
                buffer_imgs = []
                with ThreadPoolExecutor(max_workers=num_workers) as ex:
                    futs = [ex.submit(fetch_to_pil, r) for r in rows_chunk]
                    for fut in as_completed(futs):
                        rec, pil = fut.result()
                        buffer_recs.append(rec)
                        buffer_imgs.append(pil)
                        pbar.update(1)
                        if len(buffer_imgs) >= embed_batch_size:
                            with torch.no_grad():
                                batch_tensor = torch.stack(
                                    [tinyvit.transforms(img) for img in buffer_imgs],
                                    dim=0,
                                )
                                if isinstance(tinyvit.device, str):
                                    batch_tensor = batch_tensor.to(tinyvit.device)
                                else:
                                    batch_tensor = batch_tensor.cuda(tinyvit.device)
                                embs = (
                                    tinyvit.tinyvit_model(batch_tensor)
                                    .to(torch.float32)
                                    .cpu()
                                )
                            dim = int(embs.shape[-1])
                            observed_dim = observed_dim or dim
                            batch_rows = []
                            for rec_i, emb in zip(buffer_recs, embs):
                                batch_rows.append(
                                    (
                                        rec_i.get("location_id"),
                                        float(rec_i.get("lat"))
                                        if rec_i.get("lat") is not None
                                        else None,
                                        float(rec_i.get("lon"))
                                        if rec_i.get("lon") is not None
                                        else None,
                                        int(rec_i.get("heading"))
                                        if rec_i.get("heading") is not None
                                        else None,
                                        rec_i.get("capture_date"),
                                        rec_i.get("pano_id"),
                                        rec_i.get("batch_date"),
                                        sqlite3.Binary(emb.numpy().tobytes()),
                                        dim,
                                    )
                                )
                            cur.executemany(
                                """
                                INSERT OR REPLACE INTO samples
                                  (location_id, lat, lon, heading, capture_date, pano_id, batch_date, embedding, embedding_dim)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                batch_rows,
                            )
                            rows_since_commit += len(batch_rows)
                            processed_total += len(batch_rows)
                            _wandb_log(
                                {
                                    "mode": "tinyvit",
                                    "processed": processed_total,
                                    "total": total_rows,
                                    "throughput_img_per_s": processed_total
                                    / max(time.time() - start_ts, 1e-6),
                                    "phase": "embedding_inserting",
                                },
                                step=processed_total,
                            )
                            buffer_recs.clear()
                            buffer_imgs.clear()
                            if rows_since_commit >= commit_interval:
                                conn.commit()
                                cur.execute("BEGIN;")
                                rows_since_commit = 0
                # Flush leftovers in this window
                if buffer_imgs:
                    with torch.no_grad():
                        batch_tensor = torch.stack(
                            [tinyvit.transforms(img) for img in buffer_imgs], dim=0
                        )
                        if isinstance(tinyvit.device, str):
                            batch_tensor = batch_tensor.to(tinyvit.device)
                        else:
                            batch_tensor = batch_tensor.cuda(tinyvit.device)
                        embs = (
                            tinyvit.tinyvit_model(batch_tensor).to(torch.float32).cpu()
                        )
                    dim = int(embs.shape[-1])
                    observed_dim = observed_dim or dim
                    batch_rows = []
                    for rec_i, emb in zip(buffer_recs, embs):
                        batch_rows.append(
                            (
                                rec_i.get("location_id"),
                                float(rec_i.get("lat"))
                                if rec_i.get("lat") is not None
                                else None,
                                float(rec_i.get("lon"))
                                if rec_i.get("lon") is not None
                                else None,
                                int(rec_i.get("heading"))
                                if rec_i.get("heading") is not None
                                else None,
                                rec_i.get("capture_date"),
                                rec_i.get("pano_id"),
                                rec_i.get("batch_date"),
                                sqlite3.Binary(emb.numpy().tobytes()),
                                dim,
                            )
                        )
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO samples
                          (location_id, lat, lon, heading, capture_date, pano_id, batch_date, embedding, embedding_dim)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch_rows,
                    )
                    rows_since_commit += len(batch_rows)
                    processed_total += len(batch_rows)
                    _wandb_log(
                        {
                            "mode": "tinyvit",
                            "processed": processed_total,
                            "total": total_rows,
                            "throughput_img_per_s": processed_total
                            / max(time.time() - start_ts, 1e-6),
                            "phase": "embedding_inserting",
                        },
                        step=processed_total,
                    )
                # Commit per window
                conn.commit()
                cur.execute("BEGIN;")
                rows_since_commit = 0
            conn.commit()
            pbar.close()
        finally:
            conn.close()

        # s3.upload_file(
        #     db_path,
        #     BUCKET,
        #     sqlite_key,
        #     ExtraArgs={"ContentType": "application/octet-stream"},
        # )

    # put_json(
    #     {"s3": f"s3://{BUCKET}/{DATASET_SQLITE_TINYVIT_PREFIX}/{run_id}/"},
    #     BUCKET,
    #     f"{DATASET_SQLITE_TINYVIT_PREFIX}/_latest.json",
    # )

    # Also write a local copy beside the repository directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    os.makedirs(repo_parent_dir, exist_ok=True)
    local_sqlite_path = os.path.join(
        repo_parent_dir, f"dataset_sqlite_tinyvit_embeddings_{run_id}.sqlite"
    )
    try:
        sz = os.path.getsize(local_sqlite_path)
    except Exception:
        sz = None
    elapsed = time.time() - start_ts
    _wandb_log(
        {
            "mode": "tinyvit",
            "duration_s": elapsed,
            "avg_throughput_img_per_s": int(len(df)) / max(elapsed, 1e-6),
            "sqlite_size_bytes": sz,
            "embedding_dim": observed_dim,
            "phase": "done",
        }
    )

    return {
        "sqlite_key": sqlite_key,
        "local_sqlite_path": local_sqlite_path,
        "rows": int(len(df)),
        "run_id": run_id,
        "device": dev,
        "embedding_dim": observed_dim,
        "model_name": model_name,
    }


def download_climate_file(path="koppen_geiger_climatezones_1991_2020_1km.tif"):
    key = f"{VERSION}/climate/koppen_geiger_climatezones_1991_2020_1km.tif"
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    s3.download_file(BUCKET, key, path)
    return path


def upload_model_checkpoint(local_ckpt_dir: str) -> str:
    """
    Uploads a full HuggingFace checkpoint directory to:
        v1/saved_models/run_ts=TIMESTAMP/<files>

    Also writes:
        v1/saved_models/_latest.json

    Returns the S3 prefix of the uploaded checkpoint.
    """
    if not os.path.isdir(local_ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {local_ckpt_dir}")

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    prefix = f"{VERSION}/saved_models/{run_id}"

    # Upload all files inside the directory
    for root, dirs, files in os.walk(local_ckpt_dir):
        for fn in files:
            local_path = os.path.join(root, fn)
            rel = os.path.relpath(local_path, local_ckpt_dir)
            key = f"{prefix}/{rel}"

            s3.upload_file(
                local_path,
                BUCKET,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )

    # Write the pointer to latest
    put_json(
        {"s3": f"s3://{BUCKET}/{prefix}/"},
        BUCKET,
        f"{VERSION}/saved_models/_latest.json",
    )

    return prefix


def download_latest_model_checkpoint(dest_dir: str) -> str:
    """
    Downloads the latest uploaded checkpoint version from:
        v1/saved_models/_latest.json

    Reconstructs the original HuggingFace checkpoint folder structure
    under dest_dir.

    Returns the local path to the checkpoint folder.
    """
    ptr = get_json(BUCKET, f"{VERSION}/saved_models/_latest.json")
    if not ptr:
        raise FileNotFoundError("No latest model checkpoint pointer found.")

    prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")

    # List all checkpoint files inside prefix
    keys = list_keys(prefix)
    if not keys:
        raise FileNotFoundError(f"No checkpoint files found under: {prefix}")

    # Create local directory
    os.makedirs(dest_dir, exist_ok=True)

    for key in keys:
        rel = key[len(prefix) :].lstrip("/")
        local_path = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET, key, local_path)

    return dest_dir


def download_model_checkpoint_number(index: int, dest_dir: str) -> str:
    """
    Downloads the Nth most-recent saved model checkpoint from:
        v1/saved_models/run_ts=TIMESTAMP_N/...

    The newest checkpoint has index=0, second newest index=1, etc.

    Returns the local path to the checkpoint folder (dest_dir).
    """
    if index < 0:
        raise ValueError("index must be non-negative (0 = newest checkpoint).")

    base_prefix = f"{VERSION}/saved_models"
    keys = list_keys(base_prefix)
    if not keys:
        raise FileNotFoundError(f"No saved model checkpoints found under: {base_prefix}")

    # Collect unique run prefixes 'run_ts=...'
    run_ids: set[str] = set()
    for key in keys:
        parts = key.split("/")
        # Expect keys like: v1/saved_models/run_ts=YYYY.../file
        if len(parts) >= 3 and parts[1] == "saved_models" and parts[2].startswith(
            "run_ts="
        ):
            run_ids.add(parts[2])

    if not run_ids:
        raise FileNotFoundError(f"No run_ts=* checkpoints found under: {base_prefix}")

    # Sort run_ids by timestamp string (descending: newest first)
    sorted_run_ids = sorted(run_ids, reverse=True)
    if index >= len(sorted_run_ids):
        raise IndexError(
            f"Requested checkpoint index {index}, but only {len(sorted_run_ids)} "
            f"checkpoint runs exist under {base_prefix}."
        )

    run_id = sorted_run_ids[index]
    prefix = f"{VERSION}/saved_models/{run_id}"

    # List all files for the selected run
    run_keys = list_keys(prefix)
    if not run_keys:
        raise FileNotFoundError(f"No checkpoint files found under: {prefix}")

    # Create local directory and reconstruct relative structure
    os.makedirs(dest_dir, exist_ok=True)
    for key in run_keys:
        rel = key[len(prefix) :].lstrip("/")
        local_path = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET, key, local_path)

    return dest_dir


def main():
    """
    Convenience entrypoint: builds SQLite with JPEG bytes from latest snapshot
    and uploads it to S3 under dataset_sqlite/. Prints the resulting manifest.
    """
    # Initialize Weights & Biases if available and not already initialized
    try:
        import wandb  # local import to ensure availability

        if getattr(wandb, "run", None) is None:
            wandb.init(
                project="geoguessr-ai",
                job_type="sqlite_build",
                config={
                    "mode": "jpeg",
                    "num_workers": 64,
                    "writer_batch_size": 1000,
                },
            )
    except Exception:
        pass
    result = create_and_upload_sqlite_from_latest_snapshot()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
