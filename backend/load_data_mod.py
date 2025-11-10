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
    """
    print("Loading latest snapshot DataFrame from S3...")
    df_full = load_latest_snapshot_df()
    print(f"Successfully loaded DataFrame with {len(df_full)} total rows.")

    # --- MODIFICATION: Use.head(10) to get only the first 10 rows for testing ---
    df = df_full.head(10).copy()
    print(f"--- RUNNING IN TEST MODE: Processing only the first {len(df)} images. ---")
    # --------------------------------------------------------------------------

    # --- FIX: Initialize the lists ---
    local_paths: list[str]
    images: list[Optional[Image.Image]]
    # -------------------------------
    
    total_rows = len(df)
    print(f"Starting to process {total_rows} images...")

    for i, (_, row) in enumerate(df.iterrows()):
        # Add a progress indicator for clarity
        print(f"Processing image {i + 1} / {total_rows}...")

        local_path, _status = download(dest_dir, overwrite, row)
        local_paths.append(local_path)

        if load_images:
            try:
                with Image.open(local_path) as im:
                    images.append(im.copy())
            except Exception:
                images.append(None)

    df["local_image_path"] = local_paths
    if load_images:
        df["image"] = images

    print("--- Processing complete! ---")
    return df

# You also need to make sure you are calling this function at the end of your script.
# Add this to the bottom of your load_data_mod.py file if it's not there.
if __name__ == "__main__":
    add_file_paths_to_df()