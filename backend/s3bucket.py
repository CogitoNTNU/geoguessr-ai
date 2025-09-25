import os
import json
import hashlib
import tempfile
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from botocore.config import Config

# ---------- CONFIG ----------
BUCKET = "cogito-geoguessr"
BASE_PREFIX = "geoguessr-ai"
VERSION = "v1"
MANIFEST_PREFIX = f"{BASE_PREFIX}/manifest/{VERSION}"
SNAPSHOT_PREFIX = f"{BASE_PREFIX}/manifest_snapshot/{VERSION}"
REGION = "eu-north-1"
s3 = boto3.client("s3", region_name=REGION, config=Config(max_pool_connections=50))


# ---------- UTILS ----------
def ensure_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure geometry is Point in EPSG:4326. If not Point, use centroid."""
    if gdf.crs is None:
        raise ValueError(
            "Input GeoDataFrame has no CRS. Set gdf.set_crs('EPSG:XXXX') first."
        )
    gdf2 = gdf.to_crs(4326)
    if not all(gdf2.geometry.geom_type == "Point"):
        gdf2 = gdf2.copy()
        gdf2["geometry"] = gdf2.geometry.centroid
    return gdf2


def loc_id_from_geometry(geom: Point) -> str:
    """
    Stable id from WKB; trimming keeps it short but collision-resistant for practical use.
    """
    wkb = geom.wkb
    return hashlib.sha1(wkb).hexdigest()[:10]


def img_key(location_id: str, heading_deg: int) -> str:
    return f"{BASE_PREFIX}/images/{VERSION}/location_id={location_id}/heading={heading_deg:03d}.jpg"


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
    except s3.exceptions.NoSuchKey:
        return None


def build_records_from_gdf(
    gdf: gpd.GeoDataFrame,
    headings=(0, 90, 180, 270),
    image_path_resolver=None,
    pitch: float | None = None,
):
    """
    Expands each row in gdf to N= len(headings) records.
    - gdf must have a geometry column (Point) and be convertible to EPSG:4326
    - image_path_resolver(loc_id, heading, row) -> local path to the JPG for that (loc, heading)
    """
    gdf2 = ensure_epsg4326(gdf)
    gdf2 = gdf2.copy()
    gdf2["location_id"] = gdf2.geometry.apply(loc_id_from_geometry)
    gdf2["lon"] = gdf2.geometry.x
    gdf2["lat"] = gdf2.geometry.y

    if image_path_resolver is None:

        def image_path_resolver(loc_id, heading, row):
            return f"./images/{loc_id}_{heading:03d}.jpg"

    records = []
    for _, row in gdf2.iterrows():
        for h in headings:
            records.append(
                {
                    "location_id": row["location_id"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "heading": int(h),
                    "pitch": pitch,
                    "row_ctx": row,
                    "image_path": image_path_resolver(row["location_id"], int(h), row),
                    "geometry": row.geometry,
                }
            )
    return records


def upload_one_image(rec: dict) -> dict:
    """
    rec must include: location_id, lat, lon, heading, image_path, geometry (Point)
    Returns one manifest row (dict).
    """
    key = img_key(rec["location_id"], rec["heading"])
    with open(rec["image_path"], "rb") as f:
        data = f.read()
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=data,
        ContentType="image/jpeg",
    )
    return {
        "location_id": rec["location_id"],
        "lat": rec["lat"],
        "lon": rec["lon"],
        "heading": rec["heading"],
        "pitch": rec.get("pitch"),
        "pano_id": rec.get("pano_id"),
        "capture_date": rec.get("capture_date"),
        "s3_uri": f"s3://{BUCKET}/{key}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "license_source": "Google Street View",
        "version": VERSION,
        "geometry": rec["geometry"],
    }


def upload_batch(records: list[dict], max_workers=16) -> gpd.GeoDataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(upload_one_image, r) for r in records]
        for fut in as_completed(futs):
            rows.append(fut.result())
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def write_batch_manifest(gdf_batch: gpd.GeoDataFrame, batch_date: str) -> str:
    key = f"{MANIFEST_PREFIX}/batch_date={batch_date}/part-000.parquet"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "part-000.parquet")
        gdf_batch.to_parquet(p, index=False)
        s3.upload_file(p, BUCKET, key)
    return key


def load_previous_snapshot() -> gpd.GeoDataFrame | None:
    pointer_key = f"{SNAPSHOT_PREFIX}/_latest.json"
    ptr = get_json(BUCKET, pointer_key)
    if not ptr:
        return None
    snap_prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=snap_prefix).get("Contents", [])
    parts = []
    with tempfile.TemporaryDirectory() as td:
        for o in objs or []:
            if o["Key"].endswith(".parquet"):
                lp = os.path.join(td, os.path.basename(o["Key"]))
                s3.download_file(BUCKET, o["Key"], lp)
                parts.append(gpd.read_parquet(lp))
    if not parts:
        return None
    parts = [ensure_epsg4326(g) for g in parts]
    return gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )


def merge_snapshot(
    prev: gpd.GeoDataFrame | None, batch_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    if prev is None or prev.empty:
        return batch_gdf.copy()
    all_cols = sorted(set(prev.columns) | set(batch_gdf.columns))
    prev2 = prev.reindex(columns=all_cols)
    batch2 = batch_gdf.reindex(columns=all_cols)

    prev2["_is_new"] = 0
    batch2["_is_new"] = 1
    cat = pd.concat([prev2, batch2], ignore_index=True)

    cat.sort_values(by=["location_id", "heading", "_is_new"], inplace=True)
    latest = cat.drop_duplicates(subset=["location_id", "heading"], keep="last").drop(
        columns=["_is_new"]
    )
    latest = gpd.GeoDataFrame(latest, geometry="geometry", crs="EPSG:4326").reset_index(
        drop=True
    )
    return latest


def write_new_snapshot(gdf_latest: gpd.GeoDataFrame) -> str:
    run_id = "run_ts=" + datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    snap_prefix = f"{SNAPSHOT_PREFIX}/{run_id}"
    key = f"{snap_prefix}/part-000.parquet"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "part-000.parquet")
        gdf_latest.to_parquet(p, index=False)
        s3.upload_file(p, BUCKET, key)
    put_json(
        {"s3": f"s3://{BUCKET}/{snap_prefix}/"},
        BUCKET,
        f"{SNAPSHOT_PREFIX}/_latest.json",
    )
    return key


def ingest_geo_batch(
    gdf_locations: gpd.GeoDataFrame,
    batch_date: str | None = None,
    max_workers=16,
    headings=(0, 90, 180, 270),
    image_path_resolver=None,
    pitch=None,
):
    batch_date = batch_date or datetime.date.today().isoformat()

    records = build_records_from_gdf(
        gdf_locations,
        headings=headings,
        image_path_resolver=image_path_resolver,
        pitch=pitch,
    )

    gdf_batch = upload_batch(records, max_workers=max_workers)
    gdf_batch["batch_date"] = batch_date

    batch_key = write_batch_manifest(gdf_batch, batch_date)

    prev = load_previous_snapshot()
    latest = merge_snapshot(prev, gdf_batch)
    snapshot_key = write_new_snapshot(latest)

    return {
        "batch_manifest_key": batch_key,
        "snapshot_key": snapshot_key,
        "rows_in_batch": len(gdf_batch),
        "rows_in_latest": len(latest),
    }


def load_latest_snapshot_gdf() -> gpd.GeoDataFrame:
    pointer_key = f"{SNAPSHOT_PREFIX}/_latest.json"
    ptr = get_json(BUCKET, pointer_key)
    if not ptr:
        raise FileNotFoundError("No snapshot pointer found.")
    snap_prefix = ptr["s3"].replace(f"s3://{BUCKET}/", "")
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=snap_prefix).get("Contents", [])
    parts = []
    with tempfile.TemporaryDirectory() as td:
        for o in objs or []:
            if o["Key"].endswith(".parquet"):
                lp = os.path.join(td, os.path.basename(o["Key"]))
                s3.download_file(BUCKET, o["Key"], lp)
                parts.append(gpd.read_parquet(lp))
    if not parts:
        raise FileNotFoundError("Snapshot folder has no Parquet parts.")
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    return gdf
