from typing import List, Dict
import pydeck as pdk
import webbrowser
import os
from test_geocells import generate_points
from generate_geocells import GenerateGeocells
from cell import Cell


class CellVisualizer:
    geocells: List[Dict]
    points: List[Dict]

    def __init__(self, geocells: List[Cell]):
        self.all_geocells: List[Cell] = geocells
        self.geocells: List[Dict] = self._process_cells(geocells)
        self.points: List[Dict] = self._extract_points()

    def _process_cells(self, geocells: List[Cell]):
        polygons = []

        for cell in self.all_geocells.cells:
            for polygon in cell.polygons:
                geojson_polygon = {
                    "type": "Feature",
                    "geometry": polygon.__geo_interface__,
                    "properties": {
                        "point_count": len(polygon.exterior.coords),
                        "kommune": cell.id,
                    },
                }
                polygons.append(geojson_polygon)

        return polygons

    def _extract_points(self):
        points = []
        for cell in self.all_geocells.cells:
            if len(cell) > 0:
                print(cell)
            for p in cell.points:
                points.append(
                    {
                        "position": [p.lng, p.lat],
                        "id": getattr(p, "id", None),
                    }
                )
        print(points)
        return points

    def create_deck(self, **kwargs):
        # Compute average center from geocells

        initial_view_state = pdk.ViewState(
            latitude=60,
            longitude=10,
            zoom=1.5,
            min_zoom=1.5,
            pitch=0,
            bearing=0,
        )

        globe_view = pdk.View(
            type="_GlobeView", controller=True, width=1000, height=700
        )

        # Base map layer for countries
        countries_url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_scale_rank.geojson"
        countries_layer = pdk.Layer(
            "GeoJsonLayer",
            data=countries_url,
            stroked=True,
            filled=True,
            get_fill_color=[
                200,
                200,
                200,
                50,
            ],  # Very subtle gray fill for continent visibility
            get_line_color=[255, 255, 255, 180],  # Semi-transparent white outlines
            get_line_width=50,
            pickable=False,
        )

        # Geocells as GeoJsonLayer for better globe rendering
        geocells_data = {"type": "FeatureCollection", "features": self.geocells}

        geocells_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geocells_data,
            get_fill_color=[100, 150, 200, 140],  # Semi-transparent blue
            get_line_color=[255, 255, 255],
            get_line_width=50,
            lineWidthMinPixels=1,
            get_elevation=100000,  # Slight raise to ensure visibility over base
            pickable=True,
            auto_highlight=True,
            # Ensure geocells always render above base country layer
            parameters={"depthTest": False},
        )

        # Individual points as a ScatterplotLayer
        points_layer = pdk.Layer(
            "ScatterplotLayer",
            data=self.points,
            get_position="position",
            get_fill_color=[255, 80, 80, 200],
            stroked=False,
            pickable=False,  # keep global tooltip focused on geocells
            radiusMinPixels=2,
            radiusMaxPixels=6,
            get_radius=20000,
            parameters={"depthTest": False},
        )

        tooltip = {
            "html": "<b>Points:</b> {properties.point_count}<br/>"
            "<b>Kommune:</b> {properties.kommune}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }

        deck = pdk.Deck(
            layers=[countries_layer, geocells_layer, points_layer],
            views=[globe_view],
            initial_view_state=initial_view_state,
            tooltip=tooltip,
            map_provider=None,
            parameters={"cull": True},
            width="100%",
            height="100%",
            **kwargs,
        )

        return deck

    def show(self, **kwargs):
        """Display the geocells visualization in a new browser window"""
        deck = self.create_deck(**kwargs)

        html_content = deck.to_html(as_string=True)

        centered_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Geocells Globe</title>
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

        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        html_path = os.path.join(workspace_root, "geocells_globe.html")

        with open(html_path, "w") as f:
            f.write(centered_html)

        webbrowser.open(f"file://{html_path}")

        return deck


# Sample output ============================================================================================================================================================

points = generate_points(100000)
[print(x.lat, x.lng) for x in points]
# partition_output = partition(10, points)


geocells = GenerateGeocells(points)

visualizer = CellVisualizer(geocells)
visualizer.show()
