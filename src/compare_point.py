#!/usr/bin/env python3
"""
Comparison visualization script showing candidate points vs final Street View points.
This helps understand why points cluster in certain areas.
"""

import json
import os

def create_comparison_map(candidate_file="data/out/candidate_points.json", 
                         sv_file="data/out/sv_points.json", 
                         output_file="comparison_map.html"):
    """Create a comparison map showing candidate vs Street View points."""
    
    # Load candidate points
    candidate_points = []
    if os.path.exists(candidate_file):
        with open(candidate_file, 'r') as f:
            candidate_points = json.load(f)
    
    # Load Street View points
    sv_points = []
    if os.path.exists(sv_file):
        with open(sv_file, 'r') as f:
            sv_points = json.load(f)
    
    if not candidate_points and not sv_points:
        print("No data files found. Run the sampling script first.")
        return
    
    # Calculate center point
    all_points = candidate_points + sv_points
    if all_points:
        avg_lat = sum(point["lat"] for point in all_points) / len(all_points)
        avg_lon = sum(point["lon"] for point in all_points) / len(all_points)
    else:
        avg_lat, avg_lon = 60, 15
    
    # Convert to GeoJSON format
    candidate_features = []
    for i, point in enumerate(candidate_points):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point["lon"], point["lat"]]
            },
            "properties": {
                "id": i,
                "lat": point["lat"],
                "lon": point["lon"],
                "type": "candidate"
            }
        }
        candidate_features.append(feature)
    
    sv_features = []
    for i, point in enumerate(sv_points):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point["lon"], point["lat"]]
            },
            "properties": {
                "id": i,
                "lat": point["lat"],
                "lon": point["lon"],
                "type": "street_view"
            }
        }
        sv_features.append(feature)
    
    candidates_geojson = {
        "type": "FeatureCollection",
        "features": candidate_features
    }
    
    sv_geojson = {
        "type": "FeatureCollection",
        "features": sv_features
    }
    
    # Calculate success rate
    success_rate = (len(sv_points) / len(candidate_points) * 100) if candidate_points else 0
    
    # Create HTML map
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Candidate vs Street View Points Comparison</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 350px;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .stats {{
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }}
        .legend {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid #333;
        }}
        .candidate {{ background-color: #3388ff; }}
        .street-view {{ background-color: #ff7800; }}
        .success-rate {{
            font-weight: bold;
            color: #2d5a27;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h3>🔍 Point Distribution Analysis</h3>
        <p><strong>Candidate Points:</strong> {len(candidate_points):,}</p>
        <p><strong>Street View Points:</strong> {len(sv_points):,}</p>
        <p class="success-rate">Success Rate: {success_rate:.1f}%</p>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color candidate"></div>
                <span>Candidate Points (random sampling)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color street-view"></div>
                <span>Street View Points (verified coverage)</span>
            </div>
        </div>
        
        <div class="stats">
            <p>💡 <strong>Why clustering occurs:</strong></p>
            <p>• Urban areas = more roads = more Street View</p>
            <p>• Rural/mountain areas = fewer roads = less coverage</p>
            <p>• Random sampling + real-world constraints = natural clustering</p>
        </div>
    </div>
    
    <div class="controls">
        <h4>Layer Controls</h4>
        <label><input type="checkbox" id="toggleCandidates" checked> Show Candidate Points</label><br>
        <label><input type="checkbox" id="toggleStreetView" checked> Show Street View Points</label><br>
        <label><input type="checkbox" id="toggleClusters" checked> Cluster Points</label>
    </div>
    
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
    
    <script>
        // Initialize the map
        var map = L.map('map').setView([{avg_lat}, {avg_lon}], 6);

        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Data
        var candidatesData = {json.dumps(candidates_geojson, indent=2)};
        var svData = {json.dumps(sv_geojson, indent=2)};
        
        // Create layer groups
        var candidatesCluster = L.markerClusterGroup({{ maxClusterRadius: 30 }});
        var svCluster = L.markerClusterGroup({{ maxClusterRadius: 30 }});
        var candidatesIndividual = L.layerGroup();
        var svIndividual = L.layerGroup();
        
        // Add candidate points
        candidatesData.features.forEach(function(feature) {{
            var lat = feature.geometry.coordinates[1];
            var lon = feature.geometry.coordinates[0];
            
            var marker = L.circleMarker([lat, lon], {{
                radius: 3,
                fillColor: "#3388ff",
                color: "#0066cc",
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.6
            }});
            
            marker.bindPopup(
                '<b>Candidate Point ' + feature.properties.id + '</b><br/>' +
                'Latitude: ' + lat.toFixed(6) + '<br/>' +
                'Longitude: ' + lon.toFixed(6) + '<br/>' +
                'Status: Random sample within country boundary'
            );
            
            candidatesCluster.addLayer(marker);
            candidatesIndividual.addLayer(marker);
        }});
        
        // Add Street View points
        svData.features.forEach(function(feature) {{
            var lat = feature.geometry.coordinates[1];
            var lon = feature.geometry.coordinates[0];
            
            var marker = L.circleMarker([lat, lon], {{
                radius: 4,
                fillColor: "#ff7800",
                color: "#cc4400",
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }});
            
            marker.bindPopup(
                '<b>Street View Point ' + feature.properties.id + '</b><br/>' +
                'Latitude: ' + lat.toFixed(6) + '<br/>' +
                'Longitude: ' + lon.toFixed(6) + '<br/>' +
                'Status: ✅ Verified Street View coverage<br/>' +
                '<a href="https://www.google.com/maps/@' + lat + ',' + lon + ',3a,75y,90t/data=!3m6!1e1" target="_blank">🔗 Open in Street View</a>'
            );
            
            svCluster.addLayer(marker);
            svIndividual.addLayer(marker);
        }});
        
        // Add layers to map
        var currentCandidatesLayer = candidatesCluster;
        var currentSvLayer = svCluster;
        var useClusters = true;
        
        map.addLayer(currentCandidatesLayer);
        map.addLayer(currentSvLayer);
        
        // Controls
        document.getElementById('toggleCandidates').addEventListener('change', function(e) {{
            if (e.target.checked) {{
                map.addLayer(currentCandidatesLayer);
            }} else {{
                map.removeLayer(currentCandidatesLayer);
            }}
        }});
        
        document.getElementById('toggleStreetView').addEventListener('change', function(e) {{
            if (e.target.checked) {{
                map.addLayer(currentSvLayer);
            }} else {{
                map.removeLayer(currentSvLayer);
            }}
        }});
        
        document.getElementById('toggleClusters').addEventListener('change', function(e) {{
            // Remove current layers
            map.removeLayer(currentCandidatesLayer);
            map.removeLayer(currentSvLayer);
            
            // Switch layer types
            if (e.target.checked) {{
                currentCandidatesLayer = candidatesCluster;
                currentSvLayer = svCluster;
                useClusters = true;
            }} else {{
                currentCandidatesLayer = candidatesIndividual;
                currentSvLayer = svIndividual;
                useClusters = false;
            }}
            
            // Re-add if enabled
            if (document.getElementById('toggleCandidates').checked) {{
                map.addLayer(currentCandidatesLayer);
            }}
            if (document.getElementById('toggleStreetView').checked) {{
                map.addLayer(currentSvLayer);
            }}
        }});
        
        // Fit map to show all points
        var allLayers = [candidatesCluster, svCluster];
        if (allLayers.length > 0) {{
            var group = new L.featureGroup(allLayers);
            map.fitBounds(group.getBounds().pad(0.1));
        }}
        
        // Add scale
        L.control.scale().addTo(map);
    </script>
</body>
</html>"""

    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Comparison visualization created: {output_file}")
    print(f"📊 Candidate points: {len(candidate_points):,}")
    print(f"🎯 Street View points: {len(sv_points):,}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print("🚀 Open the file in your browser to see the comparison!")
    
    # Try to open in browser
    try:
        import webbrowser
        file_path = os.path.abspath(output_file)
        webbrowser.open(f"file://{file_path}")
        print("🌐 Comparison map opened in your default browser!")
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
        print(f"Please manually open: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    create_comparison_map()
