import json
import math
import os
import random
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from transformers import CLIPVisionModel

from backend.s3bucket import download_model_checkpoint_number
from config import CLIP_MODEL
from inference import _find_default_checkpoint, _load_geocell_metadata
from main_coordinator_idun_s3 import LocalGeoMapDataset
from models.super_guessr import SuperGuessr
from training.load_sqlite_dataset import load_sqlite_panorama_dataset


_BASE_DIR = os.path.dirname(__file__)


def _build_clip_transform() -> torch.nn.Module:
    """Preprocessing pipeline matching CLIP settings used in inference.py."""
    size = 336
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return T.Compose(
        [T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)]
    )


def _load_clip_backbone_from_index(
    index: int, dest_dir: str, device: torch.device
) -> CLIPVisionModel:
    """
    Load a CLIP vision backbone from the Nth most recent saved_models checkpoint.

    Falls back to CLIP_MODEL if no indexed checkpoint is found.
    """
    try:
        ckpt_dir = download_model_checkpoint_number(index=index, dest_dir=dest_dir)
    except (FileNotFoundError, IndexError, ValueError):
        ckpt_dir = CLIP_MODEL
    model = CLIPVisionModel.from_pretrained(ckpt_dir)
    return model.to(device)


def _haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two points on Earth (km)."""
    r_km = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return r_km * c


def run_benchmark(
    num_samples: int = 100,
    clip_checkpoint_index: int = 1,
    sqlite_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run SuperGuessr inference on random panoramas from the test set and
    write per-sample metrics to data/out/inference_results.json.

    - Test set = last 10% of rows from load_sqlite_panorama_dataset(sqlite_path).
    - CLIP backbone from the Nth most-recent saved_models checkpoint (index=1 by default).
    - SuperGuessr weights loaded from the default inference checkpoint if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load panorama dataset from SQLite (S3-aware loader).
    df = load_sqlite_panorama_dataset(sqlite_path)
    n_total = len(df)
    if n_total == 0:
        raise ValueError("Panorama dataset is empty.")

    train_fraction = 0.9
    num_training_samples = int(n_total * train_fraction)
    df_test = df.iloc[num_training_samples:].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(
            "Test split is empty (no rows in the last 10% of the dataset)."
        )

    # Dataset reusing LocalGeoMapDataset logic, but with CLIP-style transform.
    transform = _build_clip_transform()
    test_dataset = LocalGeoMapDataset(df_test, required_size=336, transform=transform)

    # Sample indices for evaluation (100 panoramas or as many as available).
    n_eval = min(num_samples, len(test_dataset))
    if n_eval <= 0:
        raise ValueError("No samples available for benchmarking.")
    indices = random.sample(range(len(test_dataset)), k=n_eval)

    # Build CLIP backbone and SuperGuessr model in serving panorama mode.
    clip_dest = os.path.join(_BASE_DIR, "checkpoints", "clip_backbone_s3_index1")
    backbone_model = _load_clip_backbone_from_index(
        index=clip_checkpoint_index, dest_dir=clip_dest, device=device
    )
    model = SuperGuessr(
        base_model=backbone_model,
        panorama=True,
        serving=True,
        should_smooth_labels=False,
    ).to(device)

    # Load SuperGuessr checkpoint parameters if default checkpoint exists.
    checkpoint = _find_default_checkpoint()
    if checkpoint:
        raw_state = torch.load(checkpoint, map_location=device)
        state_dict = (
            raw_state.get("model_state_dict", raw_state)
            if isinstance(raw_state, dict)
            else raw_state
        )
        model_state = model.state_dict()
        filtered_state: Dict[str, torch.Tensor] = {}
        for name, param in state_dict.items():
            target = model_state.get(name, None)
            if target is None:
                continue
            if target.shape != param.shape:
                continue
            filtered_state[name] = param
        model.load_state_dict(filtered_state, strict=False)

    model.eval()

    # Metadata: geocell_index -> (country, admin1)
    meta = _load_geocell_metadata()

    # Geoguessr scoring parameters in kilometers (size = 14,916.862 m).
    size_km = 14916.862 / 1000.0

    results: List[Dict[str, Any]] = []

    for idx in tqdm(indices, desc="Benchmarking", unit="sample"):
        images, target = test_dataset[idx]

        # images: (V, C, H, W) for panoramas
        if images.dim() == 3:
            # Single image fallback.
            pixel_values = images.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        elif images.dim() == 4:
            pixel_values = images.unsqueeze(0)  # (1, V, C, H, W)
        else:
            raise ValueError(
                f"Unexpected image tensor shape {tuple(images.shape)} for index {idx}."
            )

        pixel_values = pixel_values.to(device)
        dummy_labels = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            pred_llh, topk, _ = model(
                pixel_values=pixel_values,
                labels_clf=dummy_labels,
            )

        # SuperGuessr returns (lon, lat); convert to (lat, lon).
        pred_lon, pred_lat = pred_llh[0].tolist()
        gt_lat = float(target["lat"])
        gt_lon = float(target["lon"])

        distance_km = _haversine_distance_km(gt_lat, gt_lon, pred_lat, pred_lon)
        score = 5000.0 * math.exp(-10.0 * distance_km / size_km)

        top_ids = topk.indices[0].detach().cpu().tolist()
        top_probs = topk.values[0].detach().cpu().tolist()

        top5_cells: List[Dict[str, Any]] = []
        for geocell_id, prob in zip(top_ids, top_probs):
            country, admin1 = meta.get(geocell_id, ("?", "?"))
            top5_cells.append(
                {
                    "geocell_index": int(geocell_id),
                    "probability": float(prob),
                    "country": country,
                    "admin1": admin1,
                }
            )

        results.append(
            {
                "ground_truth": {"lat": gt_lat, "lon": gt_lon},
                "prediction": {"lat": pred_lat, "lon": pred_lon},
                "distance_km": distance_km,
                "score": score,
                "top5_geocells": top5_cells,
            }
        )

    # Write results to JSON in data/out.
    if output_path is None:
        output_path = os.path.join(_BASE_DIR, "data", "out", "inference_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} benchmark samples to {output_path}")
    return {
        "num_samples": len(results),
        "output_path": output_path,
        "clip_checkpoint_index": clip_checkpoint_index,
        "device": str(device),
    }


if __name__ == "__main__":
    run_benchmark()


