import os
import ast
import webbrowser
from typing import Dict, List, Tuple
import colorsys

import numpy as np
import pandas as pd
import pydeck as pdk
import geopandas as gpd
from tqdm.auto import tqdm


def _load_sv_points(points_txt_path: str) -> np.ndarray:
    # Robustly parse lines like "lat, lng"; skip blanks/malformed
    latlng: List[Tuple[float, float]] = []
    with open(points_txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) != 2:
                # Try whitespace delimiter as fallback
                parts = s.split()
            if len(parts) != 2:
                continue
            try:
                lat = float(parts[0])
                lng = float(parts[1])
                latlng.append((lat, lng))
            except Exception:
                continue
    if not latlng:
        raise ValueError("No valid coordinates parsed from sv_points_all_latlong.txt")
    return np.asarray(latlng, dtype=float)


def _parse_indices_column(indices_value: str) -> List[int]:
    # indices can be stored with/without quotes in CSV; use literal_eval robustly
    if isinstance(indices_value, list):
        return [int(i) for i in indices_value]
    if pd.isna(indices_value):
        return []
    try:
        parsed = ast.literal_eval(indices_value)
        if isinstance(parsed, (list, tuple)):
            return [int(i) for i in parsed]
        # Some singletons may be written without list brackets
        return [int(parsed)]
    except Exception:
        # Fallback: attempt simple split stripping brackets
        s = str(indices_value).strip().strip("[]")
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",")]


def _cluster_id_to_rgba(cluster_id: int) -> List[int]:
    # Distinct, stable colors per cluster id. -1 (noise) gets gray.
    if cluster_id == -1:
        return [150, 150, 150, 180]
    # Use a low-discrepancy sequence on hue
    hue = ((cluster_id * 137) % 360) / 360.0
    sat = 0.65
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return [int(r * 255), int(g * 255), int(b * 255), 200]


def _resolve_gpkg_path(repo_root: str) -> str:
    gadm_dir = os.path.join(repo_root, "data", "GADM_data")
    primary = os.path.join(gadm_dir, "gadm_world_all_levels.filtered_noadm345.gpkg")
    fallback = os.path.join(
        gadm_dir, "gadm_world_all_levels.filtered_noadm345.adm0_sorted.gpkg"
    )
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError("GADM geopackage not found in data/GADM_data")


def _load_adm2_geojson(gpkg_path: str) -> Dict:
    gdf = gpd.read_file(gpkg_path, layer="ADM_2")
    # Keep a small set of useful attributes to minimize payload
    keep_cols = [
        c for c in ["GID_0", "COUNTRY", "GID_1", "NAME_1", "GID_2", "NAME_2"] if c in gdf.columns
    ]
    if keep_cols:
        gdf = gdf[keep_cols + ["geometry"]]
    return gdf.__geo_interface__


def _build_points_from_proto(proto_csv_path: str, sv_points: np.ndarray) -> List[Dict]:
    """
    Build ScatterplotLayer-compatible point dicts, coloring by cluster,
    using proto_df indices into sv_points_all_latlong.
    """
    df = pd.read_csv(proto_csv_path)
    out_points: List[Dict] = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Building cluster points", unit="row", leave=False
    ):
        cluster_id = int(row["cluster_id"])
        color = _cluster_id_to_rgba(cluster_id)
        indices = _parse_indices_column(row["indices"])
        for idx in indices:
            if idx < 0 or idx >= len(sv_points):
                continue
            lat, lng = float(sv_points[idx, 0]), float(sv_points[idx, 1])
            out_points.append(
                {
                    "position": [lng, lat],
                    "properties": {
                        "cluster_id": cluster_id,
                        "geocell_id": int(row["geocell_id"]),
                        "country": row.get("country", None),
                    },
                    "color": color,
                }
            )
    return out_points


def create_deck(adm2_geojson: Dict, point_data: List[Dict]) -> pdk.Deck:
    # Initial view roughly centered on globe
    initial_view_state = pdk.ViewState(
        latitude=20,
        longitude=0,
        zoom=1.5,
        min_zoom=1.5,
        pitch=0,
        bearing=0,
    )

    globe_view = pdk.View(type="_GlobeView", controller=True, width=1200, height=800)

    countries_url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_scale_rank.geojson"
    countries_layer = pdk.Layer(
        "GeoJsonLayer",
        data=countries_url,
        stroked=True,
        filled=True,
        get_fill_color=[200, 200, 200, 40],
        get_line_color=[255, 255, 255, 160],
        get_line_width=40,
        pickable=False,
    )

    adm2_layer = pdk.Layer(
        "GeoJsonLayer",
        data=adm2_geojson,
        stroked=True,
        filled=False,
        get_line_color=[255, 255, 255, 180],
        get_line_width=20,
        lineWidthMinPixels=0.5,
        pickable=False,
        parameters={"depthTest": False},
    )

    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=point_data,
        get_position="position",
        get_fill_color="color",
        stroked=False,
        pickable=True,
        radiusMinPixels=1,
        radiusMaxPixels=6,
        get_radius=20000,
        parameters={"depthTest": False},
    )

    tooltip = {
        "html": "<b>Cluster:</b> {properties.cluster_id}<br/>"
        "<b>Geocell:</b> {properties.geocell_id}<br/>"
        "<b>Country:</b> {properties.country}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[countries_layer, adm2_layer, points_layer],
        views=[globe_view],
        initial_view_state=initial_view_state,
        tooltip=tooltip,
        map_provider=None,
        parameters={"cull": True},
        width="100%",
        height="100%",
    )
    return deck


def show(deck: pdk.Deck, out_html: str) -> None:
    html_content = deck.to_html(as_string=True)
    centered_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Finished Geocells Globe</title>
    <style>
        body {{ margin: 0; padding: 0; height: 100vh; display: flex; justify-content: center; align-items: center; background-color: black; }}
        .deck-container {{ width: 100%; height: 100%; max-width: 1400px; max-height: 900px; }}
    </style>
</head>
<body>
    <div class="deck-container">
        {html_content}
    </div>
</body>
</html>"""
    with open(out_html, "w") as f:
        f.write(centered_html)
    webbrowser.open(f"file://{out_html}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(repo_root, "data")
    proto_csv_path = os.path.join(data_root, "geocells", "proto_df.csv")
    sv_points_txt_path = os.path.join(data_root, "out", "sv_points_all_latlong.txt")
    gpkg_path = _resolve_gpkg_path(repo_root)

    # High-level progress bar for HTML creation workflow
    with tqdm(total=5, desc="Creating globe HTML", unit="step") as pbar:
        # 1) Resolve and load ADM_2 boundaries
        adm2_geojson = _load_adm2_geojson(gpkg_path)
        pbar.update(1)

        # 2) Load Street View points
        sv_points = _load_sv_points(sv_points_txt_path)
        pbar.update(1)

        # 3) Build colored cluster points
        point_data = _build_points_from_proto(proto_csv_path, sv_points)
        pbar.update(1)

        # 4) Create deck
        deck = create_deck(adm2_geojson, point_data)
        pbar.update(1)

        # 5) Write and open HTML
        out_html = os.path.join(os.path.dirname(__file__), "finished_geocells_globe.html")
        show(deck, out_html)
        pbar.update(1)


if __name__ == "__main__":
    main()


