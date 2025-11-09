import os
import ast
import webbrowser
from typing import Dict, List, Tuple
import colorsys
import json

import numpy as np
import pandas as pd
import pydeck as pdk
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


def _build_cluster_metadata(df: pd.DataFrame, sv_points: np.ndarray) -> Dict[Tuple[int, int], Dict]:
    """
    Returns a mapping: (geocell_id, cluster_id) -> {
        'centroid': [lat, lng],
        'color': [r,g,b,a]
    }
    Colors are assigned so that clusters within the same geocell have distinct hues.
    """
    meta: Dict[Tuple[int, int], Dict] = {}
    # Assign colors per geocell to ensure adjacent clusters are different
    geocell_to_clusters: Dict[int, List[int]] = (
        df.groupby("geocell_id")["cluster_id"].apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
    )
    # Compute centroid per (geocell, cluster) from indices
    for geocell_id, clusters in tqdm(geocell_to_clusters.items(), desc="Preparing cluster colors", unit="geocell", leave=False):
        n = max(1, len(clusters))
        cluster_index_map = {cid: i for i, cid in enumerate(clusters)}
        for cid in clusters:
            # Assign distinct hue per cluster index within the geocell
            hue = (cluster_index_map[cid] / n) % 1.0
            sat = 0.70
            val = 0.95
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            color = [int(r * 255), int(g * 255), int(b * 255), 200]
            meta[(int(geocell_id), int(cid))] = {"centroid": None, "color": color}
    # Compute centroids
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing cluster centroids", unit="row", leave=False):
        geocell_id = int(row["geocell_id"])
        cluster_id = int(row["cluster_id"])
        key = (geocell_id, cluster_id)
        idxs = _parse_indices_column(row["indices"])
        if not idxs:
            continue
        valid = [i for i in idxs if 0 <= i < len(sv_points)]
        if not valid:
            continue
        lats = [float(sv_points[i, 0]) for i in valid]
        lngs = [float(sv_points[i, 1]) for i in valid]
        meta[key]["centroid"] = [float(np.mean(lats)), float(np.mean(lngs))]
    return meta


def _build_points_from_proto(proto_csv_path: str, sv_points: np.ndarray) -> List[Dict]:
    """
    Build ScatterplotLayer-compatible point dicts, coloring by cluster,
    using proto_df indices into sv_points_all_latlong.
    """
    df = pd.read_csv(proto_csv_path)
    out_points: List[Dict] = []

    cluster_meta = _build_cluster_metadata(df, sv_points)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building cluster points", unit="row", leave=False):
        geocell_id = int(row["geocell_id"])
        cluster_id = int(row["cluster_id"])
        color = cluster_meta.get((geocell_id, cluster_id), {}).get("color", _cluster_id_to_rgba(cluster_id))
        country_val = row.get("country", None)
        country_str = None if pd.isna(country_val) else str(country_val)
        indices = _parse_indices_column(row["indices"])
        for idx in indices:
            if idx < 0 or idx >= len(sv_points):
                continue
            lat, lng = float(sv_points[idx, 0]), float(sv_points[idx, 1])
            out_points.append(
                {
                    "position": [lng, lat],
                    "country": country_str,
                    "properties": {
                        "cluster_id": cluster_id,
                        "geocell_id": geocell_id,
                        "country": country_str,
                    },
                    "color": color,
                }
            )
    return out_points


def _build_arrows_from_proto(proto_csv_path: str, sv_points: np.ndarray) -> List[Dict]:
    """
    Build ArcLayer-compatible records with source at point and target at cluster centroid.
    """
    df = pd.read_csv(proto_csv_path)
    arrows: List[Dict] = []
    cluster_meta = _build_cluster_metadata(df, sv_points)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building arrows to centroids", unit="row", leave=False):
        geocell_id = int(row["geocell_id"])
        cluster_id = int(row["cluster_id"])
        meta = cluster_meta.get((geocell_id, cluster_id))
        if not meta or not meta.get("centroid"):
            continue
        centroid_lat, centroid_lng = meta["centroid"][0], meta["centroid"][1]
        color = meta["color"]
        country_val = row.get("country", None)
        country_str = None if pd.isna(country_val) else str(country_val)
        indices = _parse_indices_column(row["indices"])
        for idx in indices:
            if idx < 0 or idx >= len(sv_points):
                continue
            lat, lng = float(sv_points[idx, 0]), float(sv_points[idx, 1])
            arrows.append(
                {
                    "source": [lng, lat],
                    "target": [centroid_lng, centroid_lat],
                    "color": color,
                    "country": country_str,
                }
            )
    return arrows


def _build_interactive_html(point_data: List[Dict], arrow_data: List[Dict]) -> str:
    # Prepare JSON payloads
    points_json = json.dumps(point_data, separators=(",", ":"))
    arrows_json = json.dumps(arrow_data, separators=(",", ":"))
    # Extract countries
    countries = sorted({(p.get("country") or "Unknown") for p in point_data})
    countries_json = json.dumps(countries, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Finished Geocells Globe</title>
    <style>
        html, body {{
            margin: 0; padding: 0; height: 100%; width: 100%; background: black;
            font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        }}
        #app {{
            position: absolute; inset: 0; display: flex; justify-content: center; align-items: center;
        }}
        #deck-container {{
            width: 100%; height: 100%; max-width: 1400px; max-height: 900px;
        }}
        #controls {{
            position: absolute; top: 16px; left: 16px; width: 280px; max-height: 80vh; overflow: hidden;
            background: rgba(20, 20, 20, 0.9); color: #fff; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.08);
        }}
        #controls header {{
            padding: 10px 12px; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        #search {{
            width: calc(100% - 24px); margin: 10px 12px; padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.08); color: #fff; outline: none;
        }}
        #country-list {{
            overflow: auto; max-height: calc(80vh - 120px); padding: 6px 8px 10px 8px;
        }}
        .country-item {{
            display: flex; align-items: center; gap: 8px; padding: 6px 6px; border-radius: 6px;
            cursor: pointer; user-select: none;
        }}
        .country-item:hover {{
            background: rgba(255,255,255,0.06);
        }}
        .country-item input {{
            cursor: pointer;
        }}
        .hint {{
            font-size: 12px; opacity: 0.8; padding: 0 12px 8px 12px;
        }}
    </style>
    <script src="https://unpkg.com/deck.gl@8.9.34/dist.min.js"></script>
    <script>
    // Embedded datasets
    const ALL_POINTS = {points_json};
    const ALL_ARROWS = {arrows_json};
    const COUNTRIES = {countries_json};

    let selected = new Set(); // default: none selected

    function filterData() {{
        if (selected.size === 0) return {{points: [], arrows: []}};
        const points = ALL_POINTS.filter(p => selected.has(p.country || "Unknown"));
        const arrows = ALL_ARROWS.filter(a => selected.has(a.country || "Unknown"));
        return {{points, arrows}};
    }}

    function createDeck() {{
        const viewState = {{longitude: 0, latitude: 20, zoom: 1.5, minZoom: 1.5, pitch: 0, bearing: 0}};
        const globe = new deck._GlobeView({{controller: true}});
        const countriesLayer = new deck.GeoJsonLayer({{
            id: 'countries',
            data: 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_scale_rank.geojson',
            stroked: true,
            filled: true,
            getFillColor: [200,200,200,40],
            getLineColor: [255,255,255,160],
            getLineWidth: 40,
            pickable: false,
            parameters: {{depthTest: false}}
        }});
        const filtered = filterData();
        const arrowsLayer = new deck.ArcLayer({{
            id: 'arrows',
            data: filtered.arrows,
            getSourcePosition: d => d.source,
            getTargetPosition: d => d.target,
            getSourceColor: d => d.color,
            getTargetColor: d => d.color,
            getWidth: 2,
            widthMinPixels: 0.2,
            parameters: {{depthTest: false}}
        }});
        const pointsLayer = new deck.ScatterplotLayer({{
            id: 'points',
            data: filtered.points,
            getPosition: d => d.position,
            getFillColor: d => d.color,
            stroked: false,
            pickable: true,
            radiusMinPixels: 3,
            radiusMaxPixels: 8,
            getRadius: 40000,
            parameters: {{depthTest: false}}
        }});
        const deckgl = new deck.Deck({{
            views: [globe],
            initialViewState: viewState,
            layers: [countriesLayer, arrowsLayer, pointsLayer],
            parent: document.getElementById('deck-container'),
            parameters: {{cull: true}},
            getTooltip: info => {{
                const p = info && info.object && info.object.properties;
                if (!p) return null;
                return {{
                    html: `<b>Cluster:</b> ${{p.cluster_id}}<br/><b>Geocell:</b> ${{p.geocell_id}}<br/><b>Country:</b> ${{p.country}}`,
                    style: {{backgroundColor: 'steelblue', color: 'white'}}
                }};
            }}
        }});
        return deckgl;
    }}

    let deckInstance = null;
    function updateDeck() {{
        const filtered = filterData();
        deckInstance.setProps({{
            layers: [
                deckInstance.props.layers[0],
                new deck.ArcLayer({{
                    id: 'arrows',
                    data: filtered.arrows,
                    getSourcePosition: d => d.source,
                    getTargetPosition: d => d.target,
                    getSourceColor: d => d.color,
                    getTargetColor: d => d.color,
                    getWidth: 2,
                    widthMinPixels: 0.2,
                    parameters: {{depthTest: false}}
                }}),
                new deck.ScatterplotLayer({{
                    id: 'points',
                    data: filtered.points,
                    getPosition: d => d.position,
                    getFillColor: d => d.color,
                    stroked: false,
                    pickable: true,
                    radiusMinPixels: 3,
                    radiusMaxPixels: 8,
                    getRadius: 40000,
                    parameters: {{depthTest: false}}
                }})
            ]
        }});
    }}

    function renderCountryList() {{
        const list = document.getElementById('country-list');
        list.innerHTML = '';
        const q = document.getElementById('search').value.toLowerCase();
        const filtered = COUNTRIES.filter(c => c.toLowerCase().includes(q));
        for (const c of filtered) {{
            const id = 'chk-' + c.replace(/\\W+/g, '_');
            const item = document.createElement('div');
            item.className = 'country-item';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.id = id;
            cb.checked = selected.has(c);
            cb.addEventListener('change', () => {{
                if (cb.checked) selected.add(c); else selected.delete(c);
                updateDeck();
            }});
            const label = document.createElement('label');
            label.htmlFor = id;
            label.textContent = c;
            item.appendChild(cb);
            item.appendChild(label);
            list.appendChild(item);
        }}
    }}

    window.addEventListener('DOMContentLoaded', () => {{
        deckInstance = createDeck();
        renderCountryList();
        document.getElementById('search').addEventListener('input', () => {{
            renderCountryList();
        }});
    }});
    </script>
</head>
<body>
    <div id="app">
        <div id="deck-container"></div>
        <div id="controls">
            <header>Filter Countries</header>
            <div class="hint">Search and select countries to display (default: none)</div>
            <input id="search" type="text" placeholder="Search countries..." />
            <div id="country-list"></div>
        </div>
    </div>
</body>
</html>"""
    return html


def show_html(html: str, out_html: str) -> None:
    with open(out_html, "w") as f:
        f.write(html)
    webbrowser.open(f"file://{out_html}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(repo_root, "data")
    proto_csv_path = os.path.join(data_root, "geocells", "proto_df.csv")
    sv_points_txt_path = os.path.join(data_root, "out", "sv_points_all_latlong.txt")

    # High-level progress bar for HTML creation workflow (no GADM)
    with tqdm(total=4, desc="Creating globe HTML", unit="step") as pbar:

        # 1) Load Street View points
        sv_points = _load_sv_points(sv_points_txt_path)
        pbar.update(1)

        # 2) Build colored cluster points
        point_data = _build_points_from_proto(proto_csv_path, sv_points)
        pbar.update(1)

        # 3) Build arrows from points to cluster centroids
        arrow_data = _build_arrows_from_proto(proto_csv_path, sv_points)
        pbar.update(1)

        # 4) Create interactive HTML and write/open
        html = _build_interactive_html(point_data, arrow_data)
        out_html = os.path.join(os.path.dirname(__file__), "finished_geocells_globe.html")
        show_html(html, out_html)
        pbar.update(1)


if __name__ == "__main__":
    main()


