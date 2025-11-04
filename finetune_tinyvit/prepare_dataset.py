"""
Prepare a local classification dataset from the latest S3 snapshot.

- Downloads images to a local folder using backend.s3bucket.download_latest_images.
- Assigns a country label per (lat, lon) via GADM country polygons.
- Writes a manifest CSV with: filepath, location_id, lat, lon, country.
- Splits into train/val CSVs.

Usage (optional docs; you can run via `python -m finetune_tinyvit.prepare_dataset`):
  - See README in this folder.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Tuple, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from backend.s3bucket import (
    load_latest_snapshot_df,
    download_latest_images,
)


def load_gadm_countries(gadm_dir: str) -> gpd.GeoDataFrame:
    """Load and concatenate GADM country GeoJSONs.

    Tries to standardize a `country` column from available fields.
    """
    gadm_path = Path(gadm_dir)
    files = sorted(gadm_path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No GADM country files found in {gadm_path}")

    gdfs: List[gpd.GeoDataFrame] = []
    for fp in files:
        gdf = gpd.read_file(fp)
        # standardize column name for country
        country_col = None
        for cand in [
            "COUNTRY",
            "NAME_0",
            "CNTRY_NAME",
            "ADM0_NAME",
            "name",
        ]:
            if cand in gdf.columns:
                country_col = cand
                break
        if country_col is None:
            # fallback: derive from filename (e.g., gadm41_NOR_0.json -> NOR)
            stem = fp.stem
            part = stem.split("_")
            code = part[1] if len(part) > 1 else stem
            gdf["country"] = code
        else:
            gdf["country"] = gdf[country_col]

        gdfs.append(gdf[["country", "geometry"]].copy())

    countries = pd.concat(gdfs, ignore_index=True)
    countries = gpd.GeoDataFrame(countries, geometry="geometry", crs="EPSG:4326")
    return countries


def build_manifest(
    out_dir: str,
    limit: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a manifest DataFrame with labels and local file paths.

    Returns the manifest (filepath, location_id, lat, lon, country).
    """
    df = load_latest_snapshot_df()
    if limit is not None and limit > 0:
        df = df.sample(n=min(limit, len(df)), random_state=seed)

    # Ensure images are downloaded
    out_dir = str(Path(out_dir).expanduser().absolute())
    download_latest_images(out_dir, overwrite=False, max_workers=16)

    # Convert s3 paths to local paths (mirrors backend.s3bucket logic)
    def s3_to_local(p: str) -> str:
        if not p.startswith("s3://"):
            return p
        return (out_dir + p.replace("s3://cogito-geoguessr/v1/images", "")).strip()

    df["filepath"] = df["image_path"].map(s3_to_local)
    # Drop any missing files (failed downloads)
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    # Label by country using point-in-polygon join
    countries = load_gadm_countries(
        "data/GADM_data/GADM_country"
    )  # EPSG:4326
    pts = gpd.GeoDataFrame(
        df[["location_id", "lat", "lon", "filepath"]].copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, countries, how="left", predicate="within")

    manifest = joined[["filepath", "location_id", "lat", "lon", "country"]].copy()
    manifest["country"].fillna("UNKNOWN", inplace=True)
    manifest.dropna(subset=["filepath"], inplace=True)
    manifest = manifest.reset_index(drop=True)
    return manifest


def split_train_val(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Stratify by country when possible
    rng = random.Random(seed)
    splits = []
    for country, grp in df.groupby("country"):
        idx = list(grp.index)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx = set(idx[:n_val])
        train_idx = [i for i in idx if i not in val_idx]
        splits.append((df.loc[train_idx], df.loc[list(val_idx)]))
    train = pd.concat([s[0] for s in splits], ignore_index=True)
    val = pd.concat([s[1] for s in splits], ignore_index=True)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./out", help="Local image cache root")
    ap.add_argument("--limit", type=int, default=2000, help="Limit total images for a lightweight run")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--manifest_dir", type=str, default="finetune_tinyvit/manifests")
    args = ap.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)

    manifest = build_manifest(args.out_dir, limit=args.limit)
    train_df, val_df = split_train_val(manifest, val_frac=args.val_frac)

    all_csv = Path(args.manifest_dir) / "all.csv"
    train_csv = Path(args.manifest_dir) / "train.csv"
    val_csv = Path(args.manifest_dir) / "val.csv"

    manifest.to_csv(all_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"Wrote {len(train_df)} train and {len(val_df)} val rows")
    print(f"Manifests: {train_csv} | {val_csv}")


if __name__ == "__main__":
    main()
