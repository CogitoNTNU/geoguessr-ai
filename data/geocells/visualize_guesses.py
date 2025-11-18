import json
import os
import webbrowser
from statistics import mean, median
from typing import Any, Dict, List


def _load_inference_results(results_path: str) -> Dict[str, Any]:
    """Load and sanitize inference results from JSON, including summary and best/worst."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    guesses: List[Dict[str, Any]] = []
    summary: Dict[str, Any] | None = None

    for entry in data:
        if entry.get("summary"):
            # Capture the summary record at the end of the file
            summary = {
                "num_samples": int(entry.get("num_samples", 0) or 0),
                "avg_distance_km": float(entry.get("avg_distance_km", 0.0) or 0.0),
                "median_distance_km": float(entry.get("median_distance_km", 0.0) or 0.0),
                "avg_top1_prob": float(entry.get("avg_top1_prob", 0.0) or 0.0),
                "avg_score": float(entry.get("avg_score", 0.0) or 0.0),
            }
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

    # If the JSON did not contain a summary row, compute basic summary stats.
    if summary is None:
        distances = [g["distance_km"] for g in guesses]
        scores = [g["score"] for g in guesses]
        summary = {
            "num_samples": len(guesses),
            "avg_distance_km": float(mean(distances)) if distances else 0.0,
            "median_distance_km": float(median(distances)) if distances else 0.0,
            "avg_top1_prob": 0.0,
            "avg_score": float(mean(scores)) if scores else 0.0,
        }

    # Compute best and worst guesses by distance
    best_idx, best_guess = min(
        enumerate(guesses), key=lambda x: x[1]["distance_km"]
    )
    worst_idx, worst_guess = max(
        enumerate(guesses), key=lambda x: x[1]["distance_km"]
    )

    summary.update(
        {
            "best_index": best_idx + 1,  # 1-based for display
            "best_distance_km": best_guess["distance_km"],
            "best_score": best_guess["score"],
            "worst_index": worst_idx + 1,
            "worst_distance_km": worst_guess["distance_km"],
            "worst_score": worst_guess["score"],
        }
    )

    return {"guesses": guesses, "summary": summary}


def _build_html(guesses: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """Build a Leaflet-based HTML map visualizing all guesses, similar to visualize_sv_points."""
    num_samples = int(summary.get("num_samples", len(guesses)) or len(guesses))
    avg_distance_km = float(summary.get("avg_distance_km", 0.0) or 0.0)
    median_distance_km = float(summary.get("median_distance_km", 0.0) or 0.0)
    avg_top1_prob = float(summary.get("avg_top1_prob", 0.0) or 0.0)
    avg_score = float(summary.get("avg_score", 0.0) or 0.0)

    best_index = int(summary.get("best_index", 1) or 1)
    best_distance_km = float(summary.get("best_distance_km", 0.0) or 0.0)
    best_score = float(summary.get("best_score", 0.0) or 0.0)
    worst_index = int(summary.get("worst_index", 1) or 1)
    worst_distance_km = float(summary.get("worst_distance_km", 0.0) or 0.0)
    worst_score = float(summary.get("worst_score", 0.0) or 0.0)

    # Nicely formatted strings for display
    avg_distance_str = f"{avg_distance_km:.1f} km"
    median_distance_str = f"{median_distance_km:.1f} km"
    avg_top1_prob_str = f"{avg_top1_prob:.3f}"
    avg_score_str = f"{avg_score:.0f}"
    best_distance_str = f"{best_distance_km:.1f} km"
    worst_distance_str = f"{worst_distance_km:.1f} km"
    best_score_str = f"{best_score:.0f}"
    worst_score_str = f"{worst_score:.0f}"

    # Compute a reasonable initial center using ground truth coordinates
    gt_lats = [g["ground_truth"]["lat"] for g in guesses]
    gt_lons = [g["ground_truth"]["lon"] for g in guesses]
    avg_lat = sum(gt_lats) / len(gt_lats)
    avg_lon = sum(gt_lons) / len(gt_lons)

    # Prepare compact data for JS
    guesses_for_js = [
        {
            "index": i,
            "gt_lat": g["ground_truth"]["lat"],
            "gt_lon": g["ground_truth"]["lon"],
            "pred_lat": g["prediction"]["lat"],
            "pred_lon": g["prediction"]["lon"],
            "distance_km": g["distance_km"],
            "score": g["score"],
        }
        for i, g in enumerate(guesses)
    ]
    guesses_json = json.dumps(guesses_for_js, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Inference Guesses Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        #map {{
            /* Leave space at the bottom for the navigation panel (slightly smaller cutoff) */
            height: calc(100vh - 90px);
            width: 100%;
        }}
        .info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 14px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 320px;
            font-size: 13px;
        }}
        .info h3 {{
            margin: 0 0 8px 0;
            font-size: 15px;
        }}
        .info p {{
            margin: 2px 0;
        }}
        .info .stats {{
            margin-top: 8px;
            font-size: 12px;
            color: #4b5563;
        }}
        .controls {{
            position: absolute;
            bottom: 90px;
            right: 10px;
            background: white;
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            font-size: 13px;
        }}
        .controls label {{
            display: block;
            margin-bottom: 4px;
        }}
        #bottom-panel {{
            /* Sits below the map, always visible without overlapping it */
            position: relative;
            margin: 0 auto 10px auto;
            width: calc(100% - 20px);  /* 10px side padding visually */
            max-width: 600px;
            padding: 10px 16px;
            border-radius: 10px;
            background: rgba(17, 24, 39, 0.94);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(6px);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            font-size: 14px;
            color: #f9fafb;
            z-index: 1001;
        }}
        #bottom-panel .metric {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        #bottom-panel .metric-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.75;
        }}
        #bottom-panel .metric-value {{
            font-size: 15px;
            font-weight: 600;
        }}
        #nav-buttons {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .nav-button {{
            width: 30px;
            height: 30px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: rgba(255, 255, 255, 0.06);
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            cursor: pointer;
            user-select: none;
        }}
        .nav-button:hover {{
            background: rgba(255, 255, 255, 0.16);
        }}
        .nav-button:active {{
            background: rgba(255, 255, 255, 0.25);
        }}
        #index-indicator {{
            font-size: 12px;
            opacity: 0.85;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h3>🎯 Inference Guesses</h3>
        <p><strong>Samples:</strong> {num_samples:,}</p>
        <p><strong>Avg distance:</strong> {avg_distance_str}</p>
        <p><strong>Median distance:</strong> {median_distance_str}</p>
        <p><strong>Avg top-1 prob:</strong> {avg_top1_prob_str}</p>
        <p><strong>Avg score:</strong> {avg_score_str}</p>
        <hr/>
        <p><strong>Best (by distance):</strong> #{best_index} — {best_distance_str}, score {best_score_str}</p>
        <p><strong>Worst (by distance):</strong> #{worst_index} — {worst_distance_str}, score {worst_score_str}</p>
        <div class="stats">
            <p>🟥 Ground truth points</p>
            <p>🟩 Prediction points</p>
            <p>➖ Lines connect ground truth to prediction for each sample</p>
            <p>📍 Click points for detailed per-sample info</p>
        </div>
    </div>

    <div class="controls">
        <label><input type="checkbox" id="toggleLine" checked> Show error line</label>
    </div>

    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <div id="bottom-panel">
        <div class="metric">
            <div class="metric-label">Distance from guess</div>
            <div class="metric-value" id="distance-value">–</div>
        </div>
        <div class="metric">
            <div class="metric-label">Geoguessr score</div>
            <div class="metric-value" id="score-value">–</div>
        </div>
        <div id="nav-buttons">
            <div class="nav-button" id="prev-button" title="Previous guess">&#8592;</div>
            <div id="index-indicator">0 / 0</div>
            <div class="nav-button" id="next-button" title="Next guess">&#8594;</div>
        </div>
    </div>
    <script>
        // Guesses data from Python
        var guesses = {guesses_json};

        // Initialize the map
        var map = L.map('map').setView([{avg_lat}, {avg_lon}], 2);

        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // State for current sample
        var currentIndex = 0;
        var gtMarker = null;
        var predMarker = null;
        var line = null;

        function clampIndex(idx) {{
            var n = guesses.length;
            if (idx < 0) return n - 1;
            if (idx >= n) return 0;
            return idx;
        }}

        function formatDistance(d) {{
            if (!d) return '0.0 km';
            return d >= 1000 ? d.toFixed(0) + ' km' : d.toFixed(1) + ' km';
        }}

        function updateSample() {{
            if (!guesses.length) return;

            var g = guesses[currentIndex];
            var gtLat = g.gt_lat;
            var gtLon = g.gt_lon;
            var predLat = g.pred_lat;
            var predLon = g.pred_lon;

            var distance = g.distance_km || 0;
            var score = g.score || 0;
            var distanceStr = formatDistance(distance);
            var scoreStr = Math.round(score).toString();

            var popupBase =
                '<b>Sample #' + (g.index + 1) + '</b><br/>' +
                '<b>Distance:</b> ' + distanceStr + '<br/>' +
                '<b>Score:</b> ' + scoreStr + '<br/>' +
                '<b>Ground truth:</b> ' + gtLat.toFixed(4) + ', ' + gtLon.toFixed(4) + '<br/>' +
                '<b>Prediction:</b> ' + predLat.toFixed(4) + ', ' + predLon.toFixed(4);

            // Remove existing layers
            if (gtMarker) map.removeLayer(gtMarker);
            if (predMarker) map.removeLayer(predMarker);
            if (line) map.removeLayer(line);

            // Ground truth marker (red)
            gtMarker = L.circleMarker([gtLat, gtLon], {{
                radius: 4,
                fillColor: '#ef4444',
                color: '#b91c1c',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.9
            }});
            gtMarker.bindPopup('<div style="font-size:13px;"><b>Ground truth</b><br/>' + popupBase + '</div>');

            // Prediction marker (green)
            predMarker = L.circleMarker([predLat, predLon], {{
                radius: 4,
                fillColor: '#22c55e',
                color: '#15803d',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.9
            }});
            predMarker.bindPopup('<div style="font-size:13px;"><b>Prediction</b><br/>' + popupBase + '</div>');

            gtMarker.addTo(map);
            predMarker.addTo(map);

            // Error line between ground truth and prediction
            line = L.polyline(
                [
                    [gtLat, gtLon],
                    [predLat, predLon]
                ],
                {{
                    color: '#6b7280',
                    weight: 3,
                    opacity: 0.7
                }}
            );

            if (document.getElementById('toggleLine').checked) {{
                line.addTo(map);
            }}

            // Fit view to show both points
            var bounds = L.latLngBounds([
                [gtLat, gtLon],
                [predLat, predLon]
            ]);
            map.fitBounds(bounds.pad(0.5));

            // Update bottom panel
            var distanceEl = document.getElementById('distance-value');
            var scoreEl = document.getElementById('score-value');
            var indexEl = document.getElementById('index-indicator');

            if (distanceEl) distanceEl.textContent = distanceStr;
            if (scoreEl) scoreEl.textContent = scoreStr;
            if (indexEl) indexEl.textContent = (currentIndex + 1) + ' / ' + guesses.length;
        }}

        function goToPrevious() {{
            currentIndex = clampIndex(currentIndex - 1);
            updateSample();
        }}

        function goToNext() {{
            currentIndex = clampIndex(currentIndex + 1);
            updateSample();
        }}

        // Initial render
        updateSample();

        // Navigation buttons
        var prevBtn = document.getElementById('prev-button');
        var nextBtn = document.getElementById('next-button');
        if (prevBtn) {{
            prevBtn.addEventListener('click', function() {{
                goToPrevious();
            }});
        }}
        if (nextBtn) {{
            nextBtn.addEventListener('click', function() {{
                goToNext();
            }});
        }}

        // Keyboard navigation with left/right arrows
        window.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowLeft') {{
                goToPrevious();
            }} else if (event.key === 'ArrowRight') {{
                goToNext();
            }}
        }});

        // Toggle line visibility
        var toggleLine = document.getElementById('toggleLine');
        if (toggleLine) {{
            toggleLine.addEventListener('change', function(e) {{
                if (line) {{
                    if (e.target.checked) {{
                        line.addTo(map);
                    }} else {{
                        map.removeLayer(line);
                    }}
                }}
            }});
        }}

        // Add scale
        L.control.scale().addTo(map);
    </script>
</body>
</html>"""
    return html


def show_html(html: str, out_html: str) -> None:
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    webbrowser.open(f"file://{out_html}")


def main() -> None:
    # Mirror the repo-root logic used in visualize_finished_geocells.py
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(repo_root, "data")
    results_path = os.path.join(data_root, "out", "inference_results.json")

    result_data = _load_inference_results(results_path)
    guesses: List[Dict[str, Any]] = result_data["guesses"]
    summary: Dict[str, Any] = result_data["summary"]
    html = _build_html(guesses, summary)

    # Keep filename for backwards compatibility
    out_html = os.path.join(os.path.dirname(__file__), "guesses_globe.html")
    show_html(html, out_html)


if __name__ == "__main__":
    main()
