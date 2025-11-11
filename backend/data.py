from backend.s3bucket import load_latest_snapshot_df


import io
from typing import Iterator, Optional, Dict, Any
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image
import fsspec
from torchvision import transforms as T

# ---- Your existing helpers (assumed available) ----
# from backend.s3bucket import load_latest_snapshot_df


def _ensure_s3_uri(path: str, bucket: Optional[str] = None) -> str:
    # Accept either s3://bucket/key or just key with provided bucket
    if path.startswith("s3://"):
        return path
    if bucket is None:
        raise ValueError(
            f"image_path='{path}' is not an s3:// URI and no bucket provided."
        )
    return f"s3://{bucket}/{path.lstrip('/')}"


def _build_fs(
    cache_dir: Optional[str] = None,
    anon: bool = False,
    s3_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Make a per-worker filesystem. If cache_dir is set, wrap S3 with a local file cache.
    """
    s3_kwargs = s3_kwargs or {}
    base = fsspec.filesystem(
        "s3",
        anon=anon,
        **s3_kwargs,  # e.g. {"client_kwargs": {"endpoint_url": "..."}}
    )
    if cache_dir:
        # Use filecache layer to avoid re-downloading across epochs
        fs = fsspec.filesystem(
            "filecache",
            target_protocol="s3",
            target_options={"anon": anon, **s3_kwargs},
            cache_storage=cache_dir,
            cache_check=600,  # seconds between cache validity checks
            same_names=True,
        )
        # filecache returns an FS directly usable with open(s3://...)
        return fs
    return base


class GeoImageIterableDataset(IterableDataset):
    """
    Streams images from S3 on-demand from a snapshot DataFrame.

    df columns expected:
      - image_path (s3 uri or key)
      - lat (float), lon (float)
      - heading (int)
      - location_id (str)
      - ... (other metadata allowed)
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        transform: Optional[torch.nn.Module] = None,
        required_size: Optional[int] = 336,
    ):
        super().__init__()
        self.df = df if df is not None else load_latest_snapshot_df()
        if self.df is None or len(self.df) == 0:
            raise RuntimeError("Empty snapshot DataFrame.")
        self.bucket = "cogito-geoguessr"
        self.transform = transform
        self.cache_dir = "./.s3cache"
        self.s3_anon = False
        self.s3_options = {}
        self.required_size = required_size

        # Default transform to make collate feasible: resize + tensor
        if self.transform is None:
            ops = []
            if self.required_size:
                ops.append(T.Resize(self.required_size))
            ops += [T.ToTensor()]
            self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.df)

    def _iter_shard(self) -> Iterator[int]:
        """
        Split rows across dataloader workers.
        """
        worker = get_worker_info()
        n = len(self.df)
        if worker is None:
            # single-process
            yield from range(n)
        else:
            # even sharding by index
            wid, wnum = worker.id, worker.num_workers
            # round-robin shard
            for i in range(wid, n, wnum):
                yield i

    def _open_image(self, fs, uri: str) -> Image.Image:
        # Read and decode
        with fs.open(uri, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            return img.convert("RGB")

    def __iter__(self):
        """
        Yields (image_tensor, target_dict) tuples.
        Target-dict is a dictionary of all the metadata connected to the picture,
        which is fetched from the connected parquet-file.
        """
        # Build a per-worker filesystem (thread/process-safe)
        fs = _build_fs(
            cache_dir=self.cache_dir, anon=self.s3_anon, s3_kwargs=self.s3_options
        )

        for idx in self._iter_shard():
            row = self.df.iloc[idx]
            uri = _ensure_s3_uri(str(row["image_path"]), bucket=self.bucket)

            # Robustness: basic retry loop
            img = None
            last_exc = None
            for attempt in range(3):
                try:
                    img = self._open_image(fs, uri)
                    break
                except Exception as e:
                    last_exc = e
                    continue

            # FIX: Return a placeholder instead of skipping (which causes None in batch)
            if img is None:
                # Log warning and yield a black placeholder image
                print(
                    f"Warning: Failed to load image at index {idx} (uri={uri}): {last_exc}"
                )
                # Create a placeholder tensor of expected size
                if self.required_size:
                    placeholder = torch.zeros(3, self.required_size, self.required_size)
                else:
                    placeholder = torch.zeros(3, 224, 224)
                tensor = placeholder
            else:
                # Apply transform to get a tensor (C,H,W)
                tensor = self.transform(img) if self.transform else img

            # Build label/target dict (adjust to your needs)
            # Note: Avoid None values in targets; default_collate can't handle NoneType.
            # Use empty string for missing optional fields to keep types consistent across the batch.
            target = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "location_id": str(row["location_id"]),
                "heading": int(row.get("heading", 0)),
                # Optional fields (always strings, never None)
                # "pano_id": "" if pd.isna(row.get("pano_id", None)) else str(row["pano_id"]),
                "capture_date": ""
                if pd.isna(row.get("capture_date", None))
                else str(row["capture_date"]),
                # "batch_date": str(row.get("batch_date", "")),
                # "image_path": uri,
            }

            yield tensor, target


class PanoramaIterableDataset(IterableDataset):
    """
    Yields (image_stack, target) where:
      - image_stack: (4, C, H, W)
      - target: dict (shared metadata for the panorama)
    Assumes df contains 'pano_id' and each pano has >= tiles_per_pano tiles.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[torch.nn.Module] = None,
        tiles_per_pano: int = 4,
        order_cols=("tile_idx", "yaw"),
    ):
        super().__init__()
        # Provide a safe default transform that converts PIL -> Tensor if none supplied.
        # Users can still pass their own (e.g., with Resize/Normalize). If they pass an identity
        # lambda returning a PIL Image, we will convert it just before stacking to avoid TypeError.
        self.transform = transform
        self.tiles_per_pano = tiles_per_pano
        self.bucket = "cogito-geoguessr"

        if "pano_id" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'pano_id' column for panorama grouping."
            )

        # Pre-group & sort deterministically so workers shard by group, not by row
        groups = []
        for pid, g in df.groupby("pano_id"):
            g = g.copy()
            for col in order_cols:
                if col in g.columns:
                    g = g.sort_values(col)
                    break
            rows = list(g.itertuples(index=False))
            if len(rows) >= tiles_per_pano:
                groups.append(rows[:tiles_per_pano])  # exactly 4
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def _iter_group_shard(self):
        """Shard by panorama groups so a pano never crosses workers."""
        worker = get_worker_info()
        n = len(self.groups)
        if worker is None:
            yield from range(n)
        else:
            wid, wnum = worker.id, worker.num_workers
            for i in range(wid, n, wnum):
                yield i

    def _load_image_from_row(self, fs, row):
        """Open image from S3 using a shared filesystem."""
        uri = _ensure_s3_uri(str(row.image_path), bucket=self.bucket)
        with fs.open(uri, "rb") as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def __iter__(self):
        # Build FS ONCE per worker
        fs = _build_fs(cache_dir="./.s3cache", anon=False, s3_kwargs={})

        for gi in self._iter_group_shard():
            rows = self.groups[gi]
            imgs = []
            target = {
                "pano_id": getattr(rows[0], "pano_id", None),
                "lat": float(getattr(rows[0], "lat", 0.0)),
                "lon": float(getattr(rows[0], "lon", 0.0)),
                "location_id": str(getattr(rows[0], "location_id", "")),
                "image_paths": [],
            }
            for r in rows:
                pil = self._load_image_from_row(fs, r)  # PIL.Image
                img_obj = pil
                # Apply user transform if provided
                if self.transform is not None:
                    try:
                        img_obj = self.transform(pil)
                    except Exception as e:
                        # Fallback: attempt basic ToTensor if user transform fails
                        print(
                            f"Warning: transform failed for pano_id={getattr(r, 'pano_id', 'NA')} - {e}. Using raw image."
                        )
                        img_obj = pil
                # Ensure we have a tensor (C,H,W); if still PIL, convert.
                if isinstance(img_obj, Image.Image):
                    # Minimal default: resize to 336 if large dimension differs? Keep original for now.
                    to_tensor = T.ToTensor()
                    img_obj = to_tensor(img_obj)
                if not torch.is_tensor(img_obj):
                    raise TypeError(
                        f"Transform must return a Tensor, got {type(img_obj)} for pano_id={getattr(r, 'pano_id', 'NA')}"
                    )
                imgs.append(img_obj)
                if hasattr(r, "image_path"):
                    target["image_paths"].append(str(r.image_path))
            yield torch.stack(imgs, dim=0), target  # (4,C,H,W)
