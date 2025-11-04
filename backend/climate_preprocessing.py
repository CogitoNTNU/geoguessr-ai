import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from s3bucket import load_latest_snapshot_df, write_new_snapshot

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

    df["Climate"] = kg_vals
    return df


if __name__ == "__main__":
    raster_path = "preprocessing/koppen_geiger_climatezones_1991_2020_1km.tif"
    out_path = "points_with_koppen.csv"

    df = load_latest_snapshot_df()

    df = sample_koppen(df, raster_path, CLIMATE_DICT)
    result = write_new_snapshot(df)
