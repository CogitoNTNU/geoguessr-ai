import json
import math
import os
import random
from typing import List, Dict, Any, Optional

import numpy as np
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


rad_np = 6371000.0  # Earth radius in meters for numpy haversine


def haversine_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the haversine distance between two sets of points

    Args:
        x (np.ndarray): points 1 (lon, lat)
        y (np.ndarray): points 2 (lon, lat)

    Returns:
        np.ndarray: haversine distance in km
    """
    x_rad, y_rad = map(np.radians, [x, y])

    delta = y_rad - x_rad

    a = np.sin(delta[:, 1] / 2) ** 2 + np.cos(x_rad[:, 1]) * np.cos(y_rad[:, 1]) * np.sin(
        delta[:, 0] / 2
    ) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = (rad_np * c) / 1000
    return km


def geoguessr_score_from_distance(
    distance_km: float, decay_km: float = 1492.7
) -> int:
    """
    GeoGuessr-style (World, distance scoring) points from distance in km,
    using the exponential decay model:

        points = 5000 * exp(-d / decay_km)

    Result is clamped to [0, 5000] and rounded to nearest integer.
    """
    if distance_km < 0:
        distance_km = 0.0
    points = 5000.0 * math.exp(-(distance_km / decay_km))
    points = max(0.0, min(5000.0, points))
    return int(round(points))


def _compute_summary_from_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics over a list of inference result samples.

    Each sample is expected to have:
      - "distance_km"
      - "score"
      - "top5_geocells" (list, where [0]["probability"] is the top-1 prob)
    """
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty list of samples for summary computation")

    total_distance = 0.0
    total_top_prob = 0.0
    total_score = 0.0
    n = 0
    distances: List[float] = []

    for sample in data:
        # distance_km
        dist = float(sample.get("distance_km", 0.0))
        total_distance += dist
        distances.append(dist)

        # score
        score = float(sample.get("score", 0.0))
        total_score += score

        # top geocell probability, if available
        top5 = sample.get("top5_geocells") or []
        if top5:
            top_prob = float(top5[0].get("probability", 0.0))
        else:
            top_prob = 0.0
        total_top_prob += top_prob

        n += 1

    avg_distance = total_distance / n
    avg_top_prob = total_top_prob / n
    avg_score = total_score / n
    median_distance = float(np.median(distances))

    return {
        "num_samples": n,
        "avg_distance_km": avg_distance,
        "median_distance_km": median_distance,
        "avg_top1_prob": avg_top_prob,
        "avg_score": avg_score,
    }


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

        # Use numpy haversine with (lon, lat) ordering
        x = np.array([[gt_lon, gt_lat]], dtype=float)
        y = np.array([[pred_lon, pred_lat]], dtype=float)
        distance_km = float(haversine_np(x, y)[0])
        score = geoguessr_score_from_distance(distance_km)

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

    # Append summary statistics as the last element.
    summary = _compute_summary_from_data(results)
    results.append(
        {
            "summary": True,
            "num_samples": summary["num_samples"],
            "avg_distance_km": summary["avg_distance_km"],
            "median_distance_km": summary["median_distance_km"],
            "avg_top1_prob": summary["avg_top1_prob"],
            "avg_score": summary["avg_score"],
        }
    )

    # Write results (including summary) to JSON in data/out.
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


