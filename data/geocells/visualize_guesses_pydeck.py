import json
import os
import webbrowser
from typing import Any, Dict, List

import pydeck as pdk


def _load_guesses(results_path: str) -> List[Dict[str, Any]]:
    """Load per-sample guesses from inference_results.json (skip summary rows)."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    guesses: List[Dict[str, Any]] = []
    for entry in data:
        if entry.get("summary"):
            # Skip the summary record at the end of the file
            continue

        gt = entry.get("ground_truth") or {}
        pred = entry.get("prediction") or {}

        try:
            gt_lat = float(gt["lat"])
            gt_lon = float(gt["lon"])
            pred_lat = float(pred["lat"])
            pred_lon = float(pred["lon"])
        except (KeyError, TypeError, ValueError):
            # Skip malformed entries
            continue

        distance_km = float(entry.get("distance_km", 0.0) or 0.0)
        score = float(entry.get("score", 0.0) or 0.0)

        guesses.append(
            {
                "ground_truth": {"lat": gt_lat, "lon": gt_lon},
                "prediction": {"lat": pred_lat, "lon": pred_lon},
                "distance_km": distance_km,
                "score": score,
            }
        )

    if not guesses:
        raise ValueError("No valid guess entries found in inference_results.json")

    return guesses


def _build_deck(guesses: List[Dict[str, Any]]) -> pdk.Deck:
    """Build a pydeck globe visualizing guesses as arcs + points."""

    # Prepare data for layers
    arc_data: List[Dict[str, Any]] = []
    gt_points: List[Dict[str, Any]] = []
    pred_points: List[Dict[str, Any]] = []

    for g in guesses:
        gt = g["ground_truth"]
        pred = g["prediction"]
        distance = float(g["distance_km"])
        score = float(g["score"])

        gt_pos = [gt["lon"], gt["lat"]]
        pred_pos = [pred["lon"], pred["lat"]]

        arc_data.append(
            {
                "from": gt_pos,
                "to": pred_pos,
                "distance_km": distance,
                "score": score,
            }
        )
        gt_points.append({"position": gt_pos, "distance_km": distance, "score": score})
        pred_points.append(
            {"position": pred_pos, "distance_km": distance, "score": score}
        )

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

    # Arcs from ground truth to prediction, colored by distance
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

    # Predictions as green scatterplot points
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


def main() -> None:
    # Repo root two levels above this file
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(repo_root, "data")
    results_path = os.path.join(data_root, "out", "inference_results.json")

    guesses = _load_guesses(results_path)
    deck = _build_deck(guesses)

    # Center the deck HTML in a full-page black background, similar to admin_visualizer
    html_content = deck.to_html(as_string=True)
    centered_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Inference Guesses Globe (pydeck)</title>
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

    out_html = os.path.join(os.path.dirname(__file__), "guesses_globe_pydeck.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(centered_html)

    webbrowser.open(f"file://{out_html}")


if __name__ == "__main__":
    main()


