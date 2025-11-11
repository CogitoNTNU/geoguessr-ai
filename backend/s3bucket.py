import os
import re
import json
import hashlib
import tempfile
import datetime
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import shutil

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
        max_pool_connections=50,
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

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%M%SZ")
    sqlite_key = f"{DATASET_SQLITE_PREFIX}/{run_id}/dataset.sqlite"

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "dataset.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
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

            # Single transaction with periodic commits to avoid huge WAL
            rows_since_commit = 0
            cur.execute("BEGIN;")
            for _, row in df.iterrows():
                img_bytes = _get_image_bytes(row.get("image_path"))
                cur.execute(
                    """
                    INSERT OR REPLACE INTO samples
                      (location_id, lat, lon, heading, capture_date, pano_id, batch_date, image)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("location_id"),
                        float(row.get("lat")) if row.get("lat") is not None else None,
                        float(row.get("lon")) if row.get("lon") is not None else None,
                        int(row.get("heading")) if row.get("heading") is not None else None,
                        row.get("capture_date"),
                        row.get("pano_id"),
                        row.get("batch_date"),
                        sqlite3.Binary(img_bytes),
                    ),
                )
                rows_since_commit += 1
                if rows_since_commit >= commit_interval:
                    conn.commit()
                    cur.execute("BEGIN;")
                    rows_since_commit = 0
            conn.commit()
        finally:
            conn.close()

        # Upload SQLite file
        s3.upload_file(
            db_path,
            BUCKET,
            sqlite_key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )

    # Write/update pointer
    put_json({"s3": f"s3://{BUCKET}/{DATASET_SQLITE_PREFIX}/{run_id}/"}, BUCKET, f"{DATASET_SQLITE_PREFIX}/_latest.json")

    # Also write a local copy beside the repository directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    os.makedirs(repo_parent_dir, exist_ok=True)
    local_sqlite_path = os.path.join(repo_parent_dir, f"dataset_sqlite_{run_id}.sqlite")
    # Recreate the DB to copy from S3-uploaded temp path
    # Note: db_path no longer exists outside the with block; create file again by downloading is costly.
    # Instead, regenerate inside a new temp block for copy consistency.
    # Simpler: move creation before temp cleanup. As we are outside, reuse last built path by creating again.
    # Rebuild locally by downloading from S3 would cost network; better generate inside the previous block.
    # Since temp dir is gone, copy must occur before it. Adjusted: regenerate is not feasible here.
    # Fallback: we already have sqlite_key in S3; download and save locally.
    bucket_key = sqlite_key
    with tempfile.TemporaryDirectory() as td2:
        tmp_local = os.path.join(td2, "dataset.sqlite")
        s3.download_file(BUCKET, bucket_key, tmp_local)
        shutil.copyfile(tmp_local, local_sqlite_path)

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

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%M%SZ")
    sqlite_key = f"{DATASET_SQLITE_CLIP_PREFIX}/{run_id}/dataset.sqlite"

    # Lazy imports to avoid heavy deps at module import time
    import io
    from PIL import Image
    import torch
    from pretrain.clip_embedder import CLIPEmbedding

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPEmbedding(model_name="", device=dev, load_checkpoint=False, panorama=False)
    clip_model.eval()

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "dataset.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
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

            rows_since_commit = 0
            observed_dim = None
            cur.execute("BEGIN;")
            for _, row in df.iterrows():
                img_bytes = _get_image_bytes(row.get("image_path"))
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                with torch.no_grad():
                    emb = clip_model(img)
                    if emb.ndim == 2 and emb.shape[0] == 1:
                        emb = emb[0]
                    emb = emb.detach().cpu().to(torch.float32)
                dim = int(emb.shape[-1])
                observed_dim = observed_dim or dim
                emb_bytes = emb.numpy().tobytes()

                cur.execute(
                    """
                    INSERT OR REPLACE INTO samples
                      (location_id, lat, lon, heading, capture_date, pano_id, batch_date, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("location_id"),
                        float(row.get("lat")) if row.get("lat") is not None else None,
                        float(row.get("lon")) if row.get("lon") is not None else None,
                        int(row.get("heading")) if row.get("heading") is not None else None,
                        row.get("capture_date"),
                        row.get("pano_id"),
                        row.get("batch_date"),
                        sqlite3.Binary(emb_bytes),
                        dim,
                    ),
                )
                rows_since_commit += 1
                if rows_since_commit >= commit_interval:
                    conn.commit()
                    cur.execute("BEGIN;")
                    rows_since_commit = 0
            conn.commit()
        finally:
            conn.close()

        s3.upload_file(
            db_path,
            BUCKET,
            sqlite_key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )

    put_json({"s3": f"s3://{BUCKET}/{DATASET_SQLITE_CLIP_PREFIX}/{run_id}/"}, BUCKET, f"{DATASET_SQLITE_CLIP_PREFIX}/_latest.json")

    # Also write a local copy beside the repository directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    os.makedirs(repo_parent_dir, exist_ok=True)
    local_sqlite_path = os.path.join(repo_parent_dir, f"dataset_sqlite_clip_embeddings_{run_id}.sqlite")
    with tempfile.TemporaryDirectory() as td2:
        tmp_local = os.path.join(td2, "dataset.sqlite")
        s3.download_file(BUCKET, sqlite_key, tmp_local)
        shutil.copyfile(tmp_local, local_sqlite_path)

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

    run_id = "run_ts=" + datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%M%SZ")
    sqlite_key = f"{DATASET_SQLITE_TINYVIT_PREFIX}/{run_id}/dataset.sqlite"

    # Lazy imports
    import io
    from PIL import Image
    import torch
    from pretrain.tinyvit_embedder import TinyViTEmbedding

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tinyvit = TinyViTEmbedding(model_name=model_name, device=dev, load_checkpoint=False, panorama=False)
    tinyvit.eval()

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "dataset.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
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

            rows_since_commit = 0
            observed_dim = None
            cur.execute("BEGIN;")
            for _, row in df.iterrows():
                img_bytes = _get_image_bytes(row.get("image_path"))
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                with torch.no_grad():
                    emb = tinyvit(img)
                    if emb.ndim == 2 and emb.shape[0] == 1:
                        emb = emb[0]
                    emb = emb.detach().cpu().to(torch.float32)
                dim = int(emb.shape[-1])
                observed_dim = observed_dim or dim
                emb_bytes = emb.numpy().tobytes()

                cur.execute(
                    """
                    INSERT OR REPLACE INTO samples
                      (location_id, lat, lon, heading, capture_date, pano_id, batch_date, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("location_id"),
                        float(row.get("lat")) if row.get("lat") is not None else None,
                        float(row.get("lon")) if row.get("lon") is not None else None,
                        int(row.get("heading")) if row.get("heading") is not None else None,
                        row.get("capture_date"),
                        row.get("pano_id"),
                        row.get("batch_date"),
                        sqlite3.Binary(emb_bytes),
                        dim,
                    ),
                )
                rows_since_commit += 1
                if rows_since_commit >= commit_interval:
                    conn.commit()
                    cur.execute("BEGIN;")
                    rows_since_commit = 0
            conn.commit()
        finally:
            conn.close()

        s3.upload_file(
            db_path,
            BUCKET,
            sqlite_key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )

    put_json({"s3": f"s3://{BUCKET}/{DATASET_SQLITE_TINYVIT_PREFIX}/{run_id}/"}, BUCKET, f"{DATASET_SQLITE_TINYVIT_PREFIX}/_latest.json")

    # Also write a local copy beside the repository directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
    os.makedirs(repo_parent_dir, exist_ok=True)
    local_sqlite_path = os.path.join(repo_parent_dir, f"dataset_sqlite_tinyvit_embeddings_{run_id}.sqlite")
    with tempfile.TemporaryDirectory() as td2:
        tmp_local = os.path.join(td2, "dataset.sqlite")
        s3.download_file(BUCKET, sqlite_key, tmp_local)
        shutil.copyfile(tmp_local, local_sqlite_path)

    return {
        "sqlite_key": sqlite_key,
        "local_sqlite_path": local_sqlite_path,
        "rows": int(len(df)),
        "run_id": run_id,
        "device": dev,
        "embedding_dim": observed_dim,
        "model_name": model_name,
    }


def main():
    """
    Convenience entrypoint: builds SQLite with JPEG bytes from latest snapshot
    and uploads it to S3 under dataset_sqlite/. Prints the resulting manifest.
    """
    result = create_and_upload_sqlite_from_latest_snapshot()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
