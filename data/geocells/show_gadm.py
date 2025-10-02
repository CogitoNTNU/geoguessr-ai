import geopandas as gpd
import sqlite3
import os
# ==================================================================================
# THIS SCRIPT CAN READ THE OVERALL STRUCTURE OF A GEOPACKAGE FILE (.gpkg)
# ==================================================================================

# Path to the GADM geopackage file - resolve relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GADM_FILE = os.path.join(
    SCRIPT_DIR, "..", "GADM_data", "gadm_world_all_levels.filtered_noadm345.gpkg"
)


def show_geopackage_structure(filepath):
    """Show the overall structure of a geopackage database"""

    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return

    print(f"Analyzing Geopackage: {filepath}")
    print("=" * 50)

    # Connect to the geopackage as SQLite database to inspect layers
    conn = sqlite3.connect(filepath)

    # Get all layer names from gpkg_contents table
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM gpkg_contents")
        layers = cursor.fetchall()

        print(f"Found {len(layers)} layers:")
        print()

        for layer in layers:
            layer_name = layer[0]
            print(f"Layer: {layer_name}")
            print("-" * 30)

            try:
                # Use geopandas to read the layer
                gdf = gpd.read_file(filepath, layer=layer_name)

                print(f"  Rows: {len(gdf)}")
                print(f"  Columns: {len(gdf.columns)}")
                print("  Column details:")

                for col in gdf.columns:
                    dtype = str(gdf[col].dtype)
                    if gdf[col].isna().all():
                        sample = "All null"
                    else:
                        sample = str(
                            gdf[col].dropna().iloc[0]
                            if len(gdf[col].dropna()) > 0
                            else "No non-null values"
                        )
                        # Truncate long samples
                        if len(sample) > 50:
                            sample = sample[:47] + "..."

                    print(f"    {col}: {dtype} -> {sample}")

                print()

            except Exception as e:
                print(f"  Error reading layer: {e}")
                print()

    except Exception as e:
        print(f"Error accessing gpkg_contents: {e}")
        print("This might not be a valid geopackage file.")

    finally:
        conn.close()


if __name__ == "__main__":
    show_geopackage_structure(GADM_FILE)
