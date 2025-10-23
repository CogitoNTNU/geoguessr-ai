import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

CLIMATE_DICT = {  # Köppen-Geiger Climate Zones
    1: ("Af", "Tropical, rainforest", "a tropical rainforest climate"),
    2: ("Am", "Tropical, monsoon", "a tropical monsoon climate"),
    3: ("Aw", "Tropical, savannah", "a tropical savanna climate"),
    4: ("BWh", "Arid, desert, hot", "an arid, hot desert climate"),
    5: ("BWk", "Arid, desert, cold", "an arid, cold desert climate"),
    6: ("BSh", "Arid, steppe, hot", "a hot, semi-arid climate"),
    7: ("BSk", "Arid, steppe, cold", "a cold, semi-arid climate"),
    8: (
        "Csa",
        "Temperate, dry summer, hot summer",
        "a Mediterranean climate with a hot summer",
    ),
    9: (
        "Csb",
        "Temperate, dry summer, warm summer",
        "a Mediterranean climate with a warm summer",
    ),
    10: (
        "Csc",
        "Temperate, dry summer, cold summer",
        "a Mediterranean climate with a cold summer",
    ),
    11: (
        "Cwa",
        "Temperate, dry winter, hot summer",
        "a humid subtropical monsoon climate",
    ),
    12: (
        "Cwb",
        "Temperate, dry winter, warm summer",
        "a temperate oceanic monsoon climate",
    ),
    13: (
        "Cwc",
        "Temperate, dry winter, cold summer",
        "a subpolar oceanic monsoon climate",
    ),
    14: ("Cfa", "Temperate, no dry season, hot summer", "a humid subtropical climate"),
    15: ("Cfb", "Temperate, no dry season, warm summer", "a temperate oceanic climate"),
    16: ("Cfc", "Temperate, no dry season, cold summer", "a subpolar oceanic climate"),
    17: (
        "Dsa",
        "Cold, dry summer, hot summer",
        "a Mediterranean humid continental climate with a hot summer",
    ),
    18: (
        "Dsb",
        "Cold, dry summer, warm summer",
        "a Mediterranean humid continental climate with a warm summer",
    ),
    19: (
        "Dsc",
        "Cold, dry summer, cold summer",
        "a Mediterranean subarctic climate with a cold summer",
    ),
    20: (
        "Dsd",
        "Cold, dry summer, very cold winter",
        "a Mediterranean humid continental climate with a warm summer",
    ),
    21: (
        "Dwa",
        "Cold, dry winter, hot summer",
        "a humid continental monsoon climate with a hot summer",
    ),
    22: (
        "Dwb",
        "Cold, dry winter, warm summer",
        "a humid continental monsoon climate with a warm summer",
    ),
    23: ("Dwc", "Cold, dry winter, cold summer", "a subarctic monsoon climate"),
    24: (
        "Dwd",
        "Cold, dry winter, very cold winter",
        "an extremely cold subarctic monsoon climate",
    ),
    25: (
        "Dfa",
        "Cold, no dry season, hot summer",
        "a humid continental climate with a hot summer",
    ),
    26: (
        "Dfb",
        "Cold, no dry season, warm summer",
        "a humid continental climate with a warm summer",
    ),
    27: ("Dfc", "Cold, no dry season, cold summer", "a subarctic climate"),
    28: (
        "Dfd",
        "Cold, no dry season, very cold winter",
        "an extremely cold subarctic climate",
    ),
    29: ("ET", "Polar, tundra", "a polar tundra climate"),
    30: ("EF", "Polar, frost", "a polar ice cap climate"),
}


# 1) Load your points (WGS84 lat/lon)
def load_points_txt(path):
    # Try pandas' sniffing (handles comma/semicolon/whitespace)
    df = pd.read_csv(
        path, sep=None, engine="python", comment="#", skip_blank_lines=True
    )

    # If only one column came in (e.g., "lat lon" without delimiter), try whitespace
    if df.shape[1] == 1:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)

    # Guess column names
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        # Try to map headers if present
        cols = [c.lower() for c in df.columns.astype(str)]
        rename_map = {}
        lat_aliases = {"lat", "latitude", "y"}
        lon_aliases = {"lon", "lng", "longitude", "x"}
        for i, c in enumerate(cols):
            if c in lat_aliases:
                rename_map[df.columns[i]] = "lat"
            if c in lon_aliases:
                rename_map[df.columns[i]] = "lon"
        df.rename(columns=rename_map, inplace=True)

        # If still unnamed, assign generic names
        if "lat" not in df.columns or "lon" not in df.columns:
            df.columns = ["col1", "col2"]
            # Heuristic: detect which is lat by valid range
            c1_in_lat_range = df["col1"].between(-90, 90).all()
            c2_in_lat_range = df["col2"].between(-90, 90).all()
            if c1_in_lat_range and not c2_in_lat_range:
                df = df.rename(columns={"col1": "lat", "col2": "lon"})
            elif c2_in_lat_range and not c1_in_lat_range:
                df = df.rename(columns={"col1": "lon", "col2": "lat"})
            else:
                # Ambiguous but common convention is lat,lon
                df = df.rename(columns={"col1": "lat", "col2": "lon"})
    else:
        raise ValueError("Could not parse two numeric columns from the .txt file.")

    # Ensure numeric
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df[["lat", "lon"]]


def kg_to_bucket(kg_str, lat):
    if not isinstance(kg_str, str) or len(kg_str) == 0:
        return None
    head = kg_str[0].upper()
    a = abs(lat)
    if head == "A":
        return "tropic"
    if head == "E":
        return "polar"
    if head == "D":
        return "temperate"
    if head == "C":
        # warm-temperate; treat low-lat C as subtropic
        return "subtropic" if a < 40 else "temperate"
    if head == "B":
        # arid spans wide latitudes; split by latitude
        return "subtropic" if a < 40 else "temperate"
    return None


def sample_koppen(df, raster_path, legend_map=None):
    """
    legend_map: dict mapping raster numeric codes -> 'Af','BWh','Csa', etc.
                If your raster already stores strings, leave legend_map=None.
    """
    df = df.copy()
    with rasterio.open(raster_path) as src:
        to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        band1 = src.read(1)

        kg_vals = []
        for lon, lat in zip(df["lon"].values, df["lat"].values):
            x, y = to_raster.transform(lon, lat)
            r, c = rowcol(src.transform, x, y)
            if 0 <= r < src.height and 0 <= c < src.width:
                val = band1[r, c]
                if legend_map is not None:
                    val = legend_map.get(int(val), None)
            else:
                val = None
            kg_vals.append(val)

    df["kg_class"] = kg_vals
    df["kg_bucket4"] = [kg_to_bucket(k, la) for k, la in zip(df["kg_class"], df["lat"])]
    return df


if __name__ == "__main__":
    in_path = "data/out/sv_points_latlong_collected.txt"
    raster_path = "preprocessing/koppen_geiger_climatezones_1991_2020_1km.tif"
    out_path = "points_with_koppen.csv"

    df = load_points_txt(in_path)

    # Example legend map if your raster stores integer codes:
    # legend_map = {1:"Af", 2:"Am", 3:"Aw", 4:"BWh", ...}

    df = sample_koppen(df, raster_path, CLIMATE_DICT)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows with Köppen classes to {out_path}")
