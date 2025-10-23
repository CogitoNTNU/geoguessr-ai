import pandas as pd
from typing import Optional
from PIL import Image

# Reuse the S3 helpers already present in this repo
from s3bucket import load_latest_snapshot_df, download


def add_file_paths_to_df(
    dest_dir: str = "out",
    overwrite: bool = False,
    load_images: bool = True,
) -> pd.DataFrame:
    """
    Load the latest snapshot metadata from S3 and ensure each image exists locally.

    For each row in the snapshot DataFrame:
    - Download the image locally (if needed) into `dest_dir` keeping the S3 folder structure
    - Add a new column `local_image_path` with the local filesystem path
    - Optionally load the image into memory and place it in a new `image` column (PIL.Image)

    Parameters:
        dest_dir: Local destination directory for downloaded images (default: "out").
        overwrite: If True, re-download even if the file already exists.
        load_images: If True, load each image into memory (can be heavy for large datasets).

    Returns:
        A DataFrame with added columns: `local_image_path` and, if requested, `image`.
    """
    df = load_latest_snapshot_df().copy()

    local_paths: list[str] = []
    images: list[Optional[Image.Image]] = []

    for _, row in df.iterrows():
        local_path, _status = download(dest_dir, overwrite, row)
        local_paths.append(local_path)

        if load_images:
            try:
                with Image.open(local_path) as im:
                    # copy() to detach from context manager and file handle
                    images.append(im.copy())
            except Exception:
                images.append(None)

    df["local_image_path"] = local_paths
    if load_images:
        df["image"] = images

    return df


