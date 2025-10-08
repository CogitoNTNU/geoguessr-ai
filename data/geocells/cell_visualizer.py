from typing import List, Dict
from test_geocells import generate_points, partition
import pydeck as pdk
import numpy as np
import webbrowser
import os


class CellVisualizer:
    geocells: List[Dict]

    def __init__(self, cells):
        self.geocells: List[Dict] = self._process_cells(cells)

    def _process_cells(self, cells):
        rectCells = [cell[2] for cell in cells]

        polygons = []
        for cell in rectCells:
            polygon = {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [cell.min_lng, cell.min_lat],  # bottom-left
                            [cell.max_lng, cell.min_lat],  # bottom-right
                            [cell.max_lng, cell.max_lat],  # top-right
                            [cell.min_lng, cell.max_lat],  # top-left
                            [
                                cell.min_lng,
                                cell.min_lat,
                            ],  # back to bottom-left (close polygon)
                        ]
                    ],
                },
                "properties": {
                    "point_count": len(cell),
                    "centroid": cell.centroid(),
                    "area": cell.area,
                },
            }
            polygons.append(polygon)

        return polygons

    def create_deck(self, **kwargs):
        # Compute average center from geocells
        if self.geocells:
            avg_lat = np.mean(
                [
                    prop["centroid"][0]
                    for prop in [cell["properties"] for cell in self.geocells]
                ]
            )
            avg_lng = np.mean(
                [
                    prop["centroid"][1]
                    for prop in [cell["properties"] for cell in self.geocells]
                ]
            )
        else:
            avg_lat, avg_lng = 0, 0

        initial_view_state = pdk.ViewState(
            latitude=avg_lat,
            longitude=avg_lng,
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

        tooltip = {
            "html": "<b>Points:</b> {properties.point_count}<br/>"
            "<b>Area:</b> {properties.area:.2f}<br/>"
            "<b>Centroid:</b> {properties.centroid}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }

        deck = pdk.Deck(
            layers=[countries_layer, geocells_layer],
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
if __name__ == "__main__":
    points = generate_points(1000)
    partition_output = partition(10, points)

    visualizer = CellVisualizer(partition_output)
    visualizer.show()
