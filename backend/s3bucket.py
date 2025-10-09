import os
import re
import json
import hashlib
import tempfile
import datetime
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ptr = get_json(BUCKET, f"{SNAPSHOT_PREFIX}/_latest.json")
    if not ptr:
        raise FileNotFoundError("No snapshot pointer found.")
    snap_prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")
    df = read_parquet_prefix(snap_prefix)
    if df is None or df.empty:
        raise FileNotFoundError("Snapshot folder has no Parquet parts.")
    return df


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
      key "<lat>_<lon>" -> {
        "lat": float, "lon": float,
        "images": { heading(int): local_path(str) }
      }
    """
    index: Dict[str, Dict[str, Any]] = {}

    for name in os.listdir(root_dir):
        if not name.lower().endswith(".jpg"):
            continue
        m = STREETVIEW_RE.match(name)
        if not m:
            continue

        lat = float(m.group(1))
        lon = float(m.group(2))
        heading = int(m.group(3)) % 360

        key = f"{lat:.7f}_{lon:.7f}"  # stable key per coordinate
        rec = index.setdefault(key, {"lat": lat, "lon": lon, "images": {}})
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
        for h in headings:
            img_path = rec["images"].get(int(h) % 360)
            if not img_path:
                continue
            records.append(
                {
                    "location_id": loc_id,
                    "lat": lat,
                    "lon": lon,
                    "heading": int(h) % 360,
                    "image_path": img_path,
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
    local_path = os.path.join(dest_dir, os.path.basename(key))
    if not overwrite and os.path.exists(local_path):
        return (local_path, "skipped (exists)")
    try:
        s3.download_file(bucket, key, local_path)
        return (local_path, "downloaded")
    except Exception as e:
        return (local_path, f"failed: {e}")


def download_latest_images(
    dest_dir: str, overwrite: bool = False, max_workers: int = 16
):
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


# upload_dataset_from_folder("./dataset", max_workers=24)
