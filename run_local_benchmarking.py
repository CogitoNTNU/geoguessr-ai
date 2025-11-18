import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from inference import _load_geocell_metadata
from run_local_inference import (
    _geoguessr_score_from_distance,
    _haversine_distance_km,
    _run_random_panorama_inference,
)


_BASE_DIR = os.path.dirname(__file__)


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run local SuperGuessr benchmarking using random panoramas from the "
            "SQLite test set or S3 holdout, and write results to data/out."
        )
    )
    parser.add_argument(
        "--sqlitePath",
        type=str,
        default=None,
        help=(
            "Optional path to the SQLite dataset file "
            "(local path or s3:// URL). "
            "If omitted, uses the S3 holdout snapshot."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--numSamples",
        type=int,
        default=100,
        help="Number of random panoramas to evaluate (default: 100).",
    )
    parser.add_argument(
        "--outputPath",
        type=str,
        default=None,
        help=(
            "Optional output JSON path. "
            "Defaults to data/out/inference_results_local.json."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    num_samples: int = max(1, int(args.numSamples))

    # Metadata: geocell_index -> (country, admin1)
    meta = _load_geocell_metadata()

    results: List[Dict[str, Any]] = []

    for _ in tqdm(
        range(num_samples),
        desc="Running local benchmarking",
        unit="sample",
    ):
        (
            pred_lat,
            pred_lon,
            top_ids,
            top_probs,
            gt_lat,
            gt_lon,
        ) = _run_random_panorama_inference(device=device, sqlite_path=args.sqlitePath)

        # Compute distance and GeoGuessr-style score using (lon, lat) ordering.
        distance_km = _haversine_distance_km(
            lon1=gt_lon, lat1=gt_lat, lon2=pred_lon, lat2=pred_lat
        )
        score = _geoguessr_score_from_distance(distance_km)

        # Top-5 geocells enriched with (country, admin1) metadata.
        top5_cells: List[Dict[str, Any]] = []
        for geocell_id, prob in list(zip(top_ids, top_probs))[:5]:
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
                "ground_truth": {"lat": float(gt_lat), "lon": float(gt_lon)},
                "prediction": {"lat": float(pred_lat), "lon": float(pred_lon)},
                "distance_km": float(distance_km),
                "score": int(score),
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
    output_path: Optional[str] = args.outputPath
    if output_path is None:
        output_path = os.path.join(
            _BASE_DIR, "data", "out", "inference_results_local.json"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"Wrote {len(results)} local benchmark samples (including summary) to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()


