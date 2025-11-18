import argparse
import math
import os
import random
from pathlib import Path
from typing import Optional
import webbrowser

import pydeck as pdk
import torch
from tqdm import tqdm

from inference import _build_transform, _load_backbone
from main_coordinator_idun_s3 import LocalGeoMapDataset
from backend.s3bucket import download_random_holdout_panorama
from models.super_guessr import SuperGuessr
from training.load_sqlite_dataset import load_sqlite_panorama_dataset
from PIL import Image


_BASE_DIR = os.path.dirname(__file__)
_EARTH_RADIUS_M = 6371000.0  # Earth radius in meters used for haversine


def _find_tinyvit_panorama_checkpoint() -> str:
    """
    Return the path to the (single) TinyViT panorama checkpoint in
    `inference/checkpoints/tinyvit_panorama`.

    Assumes there is exactly one `.pt` file in that directory.
    """
    ckpt_dir = Path(_BASE_DIR) / "inference" / "checkpoints" / "tinyvit_panorama"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}. "
            "Make sure the TinyViT panorama checkpoint is placed there."
        )

    pt_files = list(ckpt_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt checkpoint files found in {ckpt_dir}. "
            "Expected a single TinyViT panorama checkpoint."
        )
    if len(pt_files) > 1:
        raise RuntimeError(
            f"Expected exactly one .pt file in {ckpt_dir}, found {len(pt_files)}."
        )
    return str(pt_files[0])


def _haversine_distance_km(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """
    Compute haversine distance between two (lon, lat) points in kilometers.

    This matches the scalar behavior of haversine_np used in run_benchmark.py.
    """
    # Convert degrees to radians
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
        math.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(a))
    km = (_EARTH_RADIUS_M * c) / 1000.0
    return km


def _geoguessr_score_from_distance(
    distance_km: float, decay_km: float = 1492.7
) -> int:
    """
    GeoGuessr-style (World) distance score from distance in km, using:

        points = 5000 * exp(-d / decay_km)

    Result is clamped to [0, 5000] and rounded to nearest integer.
    """
    if distance_km < 0:
        distance_km = 0.0
    points = 5000.0 * math.exp(-(distance_km / decay_km))
    points = max(0.0, min(5000.0, points))
    return int(round(points))


def _build_single_guess_deck(
    gt_lat: float,
    gt_lon: float,
    pred_lat: float,
    pred_lon: float,
    distance_km: float,
    score: float,
) -> pdk.Deck:
    """
    Build a pydeck globe visualizing a single guess as arcs + points.

    This mirrors the style used in data/geocells/visualize_guesses_pydeck.py.
    """
    # Prepare data for layers
    gt_pos = [gt_lon, gt_lat]
    pred_pos = [pred_lon, pred_lat]

    arc_data = [
        {
            "from": gt_pos,
            "to": pred_pos,
            "distance_km": distance_km,
            "score": score,
        }
    ]
    gt_points = [
        {
            "position": gt_pos,
            "distance_km": distance_km,
            "score": score,
        }
    ]
    pred_points = [
        {
            "position": pred_pos,
            "distance_km": distance_km,
            "score": score,
        }
    ]

    # Base globe view and initial camera
    initial_view_state = pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=0.5,
        min_zoom=0.5,
        max_zoom=10,
        pitch=30,
        bearing=0,
    )

    globe_view = pdk.View(type="_GlobeView", controller=True, width=1000, height=700)

    # Base map layer for countries
    countries_url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_scale_rank.geojson"
    countries_layer = pdk.Layer(
        "GeoJsonLayer",
        data=countries_url,
        stroked=True,
        filled=True,
        get_fill_color=[200, 200, 200, 40],
        get_line_color=[255, 255, 255, 160],
        get_line_width=30,
        pickable=False,
    )

    # Arc from ground truth to prediction, colored by distance
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[220, 220, 220, 180],
        get_target_color=[
            "distance_km * 0.5",
            "255 - distance_km * 0.5",
            120,
            220,
        ],
        get_width=2.5,
        width_min_pixels=1,
        pickable=False,
    )

    # Ground truth as small red crosses (via TextLayer)
    gt_text_layer = pdk.Layer(
        "TextLayer",
        data=gt_points,
        get_position="position",
        get_text="×",
        get_color=[255, 60, 60, 255],
        get_size=24,
        size_units="pixels",
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        billboard=True,
    )

    # Ground truth as red scatterplot points
    gt_point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=gt_points,
        get_position="position",
        get_fill_color=[255, 60, 60, 255],
        stroked=False,
        pickable=False,
        radius_min_pixels=4,
        radius_max_pixels=12,
        get_radius=30000,
    )

    # Prediction as green scatterplot point
    pred_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pred_points,
        get_position="position",
        get_fill_color=[60, 220, 120, 255],
        stroked=False,
        pickable=False,
        radius_min_pixels=4,
        radius_max_pixels=12,
        get_radius=40000,
    )

    tooltip = {
        "html": "<b>Distance:</b> {distance_km} km<br/>"
        "<b>Score:</b> {score}",
        "style": {"backgroundColor": "black", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[countries_layer, arc_layer, gt_text_layer, gt_point_layer, pred_layer],
        views=[globe_view],
        initial_view_state=initial_view_state,
        tooltip=tooltip,
        map_provider=None,
        parameters={"cull": True},
        width="100%",
        height="100%",
    )

    return deck


def _visualize_single_guess_pydeck(
    gt_lat: float,
    gt_lon: float,
    pred_lat: float,
    pred_lon: float,
    distance_km: float,
    score: float,
) -> None:
    """
    Render a pydeck HTML globe for a single guess and open it in the browser.
    """
    deck = _build_single_guess_deck(
        gt_lat=gt_lat,
        gt_lon=gt_lon,
        pred_lat=pred_lat,
        pred_lon=pred_lon,
        distance_km=distance_km,
        score=score,
    )

    html_content = deck.to_html(as_string=True)
    centered_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Single Inference Guess Globe (pydeck)</title>
    <style>
        body {{ margin: 0; padding: 0; height: 100vh; display: flex; justify-content: center; align-items: center; background-color: black; }}
        .deck-container {{ width: 100%; height: 100%; max-width: 1200px; max-height: 800px; }}
    </style>
</head>
<body>
    <div class="deck-container">
        {html_content}
    </div>
</body>
</html>"""

    out_html = os.path.join(os.path.dirname(__file__), "single_guess_globe_pydeck.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(centered_html)

    webbrowser.open(f"file://{out_html}")


def _sample_random_test_panorama(
    sqlite_path: Optional[str] = None, required_size: int = 512
) -> tuple[torch.Tensor, dict]:
    """
    Load the panorama dataset from a SQLite snapshot, take the last 10% as a
    test split, and return a single random panorama tensor + its target dict.

    This is only used when an explicit --sqlitePath is provided; otherwise
    we prefer sampling from the holdout dataset stored in S3.
    """
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

    # Use TinyViT-style transform as in inference.py
    transform = _build_transform("tinyvit")
    test_dataset = LocalGeoMapDataset(
        df_test, required_size=required_size, transform=transform
    )

    idx = random.randrange(len(test_dataset))
    images, target = test_dataset[idx]
    return images, target


def _sample_random_holdout_panorama(
    required_size: int = 512,
) -> tuple[torch.Tensor, dict]:
    """
    Sample a single random panorama from the holdout dataset snapshot stored in S3.

    Uses backend.s3bucket.download_random_holdout_panorama() to fetch up to 4
    local JPEGs for a random location_id, then applies the TinyViT transform
    and returns a (V, C, H, W) tensor plus its (lat, lon) target dict.
    """
    # Use same temp directory convention as inference.py
    dest_dir = os.path.join(_BASE_DIR, "inference", "tmp_panorama")
    pano = download_random_holdout_panorama(dest_dir=dest_dir, overwrite=False)

    paths = pano.get("paths") or []
    if not paths:
        raise ValueError("download_random_holdout_panorama() returned no image paths.")

    transform = _build_transform("tinyvit")
    tensors = []
    for p in paths:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                tensors.append(transform(img))
        finally:
            # Best-effort cleanup of downloaded JPEGs after we've loaded them.
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                # Ignore filesystem errors during cleanup.
                pass

    images = torch.stack(tensors, dim=0)  # (V, C, H, W)
    target = {"lat": float(pano["lat"]), "lon": float(pano["lon"])}
    return images, target


def _run_random_panorama_inference(
    device: torch.device, sqlite_path: Optional[str] = None
) -> tuple[float, float, list[int], list[float], float, float]:
    """
    Run SuperGuessr inference on a single random panorama.

    If `sqlite_path` is provided, we sample from the SQLite test split.
    Otherwise, we sample from the *holdout* dataset snapshot stored in S3.

    Uses TinyViT backbone and the local TinyViT panorama checkpoint.

    Returns:
      pred_lat, pred_lon, top_geocell_ids, top_geocell_probs, gt_lat, gt_lon
    """
    if sqlite_path is not None:
        images, target = _sample_random_test_panorama(
            sqlite_path=sqlite_path, required_size=512
        )
    else:
        images, target = _sample_random_holdout_panorama(required_size=512)

    # images: (V, C, H, W) for panorama, or (C, H, W) for single image
    if images.dim() == 3:
        # Single image: (C, H, W) -> (1, 1, C, H, W)
        pixel_values = images.unsqueeze(0).unsqueeze(0)
        panorama = False
    elif images.dim() == 4:
        # Panorama: (V, C, H, W) -> (1, V, C, H, W)
        pixel_values = images.unsqueeze(0)
        panorama = True
    else:
        raise ValueError(f"Unexpected image tensor shape {tuple(images.shape)}.")

    pixel_values = pixel_values.to(device)

    backbone_model = _load_backbone("tinyvit").to(device)
    model = SuperGuessr(
        base_model=backbone_model,
        panorama=panorama,
        serving=True,
        should_smooth_labels=False,
    ).to(device)

    # Load TinyViT panorama checkpoint
    checkpoint_path = _find_tinyvit_panorama_checkpoint()
    raw_state = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()

    # Unwrap common training format {"model_state_dict": ..., ...}
    state_dict = (
        raw_state.get("model_state_dict", raw_state)
        if isinstance(raw_state, dict)
        else raw_state
    )

    filtered_state: dict[str, torch.Tensor] = {}
    for name, param in tqdm(
        state_dict.items(), desc="Filtering checkpoint parameters", unit="param"
    ):
        target_param = model_state.get(name)
        if target_param is None:
            continue
        if target_param.shape != param.shape:
            continue
        filtered_state[name] = param

    if filtered_state:
        model.load_state_dict(filtered_state, strict=False)

    model.eval()

    with torch.no_grad():
        dummy_labels = torch.zeros(1, dtype=torch.long, device=device)
        pred_llh, topk, _ = model(pixel_values=pixel_values, labels_clf=dummy_labels)

    # SuperGuessr returns (lon, lat); convert to (lat, lon)
    pred_lon, pred_lat = pred_llh[0].tolist()

    top_indices = topk.indices[0].detach().cpu().tolist()
    top_probs = topk.values[0].detach().cpu().tolist()

    gt_lat = float(target["lat"])
    gt_lon = float(target["lon"])

    return pred_lat, pred_lon, top_indices, top_probs, gt_lat, gt_lon


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run local SuperGuessr inference using a random panorama from the "
            "SQLite test set and the TinyViT panorama checkpoint in "
            "`inference/checkpoints/tinyvit_panorama`."
        )
    )
    parser.add_argument(
        "--sqlitePath",
        type=str,
        default=None,
        help=(
            "Optional path to the SQLite dataset file "
            "(local path or s3:// URL). "
            "If omitted, uses the latest-from-S3 pointer."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

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

    print(f"The model guessed: {pred_lat:.6f},{pred_lon:.6f}")
    print(f"The ground truth is: {gt_lat:.6f},{gt_lon:.6f}")
    print(f"Distance: {distance_km:.2f} km")
    print(f"GeoGuessr score: {score}")
    print("Top geocells (id, probability):")
    for gid, prob in zip(top_ids, top_probs):
        print(f"{gid}\t{prob:.4f}")

    # Visualize the single guess on a pydeck globe.
    _visualize_single_guess_pydeck(
        gt_lat=gt_lat,
        gt_lon=gt_lon,
        pred_lat=pred_lat,
        pred_lon=pred_lon,
        distance_km=distance_km,
        score=score,
    )


if __name__ == "__main__":
    main()
