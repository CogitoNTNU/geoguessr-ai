import rasterio
import pandas as pd
from data.geocells.geocell_manager import GeocellManager
from pyproj import Transformer
from rasterio.transform import rowcol
from backend.s3bucket import load_latest_snapshot_df

CLIMATE_DICT = {  # Köppen-Geiger Climate Zones
    1: ("a tropical rainforest climate"),
    2: ("a tropical monsoon climate"),
    3: ("a tropical savanna climate"),
    4: ("an arid, hot desert climate"),
    5: ("an arid, cold desert climate"),
    6: ("a hot, semi-arid climate"),
    7: ("a cold, semi-arid climate"),
    8: ("a Mediterranean climate with a hot summer"),
    9: ("a Mediterranean climate with a warm summer"),
    10: ("a Mediterranean climate with a cold summer"),
    11: ("a humid subtropical monsoon climate"),
    12: ("a temperate oceanic monsoon climate"),
    13: ("a subpolar oceanic monsoon climate"),
    14: ("a humid subtropical climate"),
    15: ("a temperate oceanic climate"),
    16: ("a subpolar oceanic climate"),
    17: ("a Mediterranean humid continental climate with a hot summer"),
    18: ("a Mediterranean humid continental climate with a warm summer"),
    19: ("a Mediterranean subarctic climate with a cold summer"),
    20: ("a Mediterranean humid continental climate with a warm summer"),
    21: ("a humid continental monsoon climate with a hot summer"),
    22: ("a humid continental monsoon climate with a warm summer"),
    23: ("a subarctic monsoon climate"),
    24: ("an extremely cold subarctic monsoon climate"),
    25: ("a humid continental climate with a hot summer"),
    26: ("a humid continental climate with a warm summer"),
    27: ("a subarctic climate"),
    28: ("an extremely cold subarctic climate"),
    29: ("a polar tundra climate"),
    30: ("a polar ice cap climate"),
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

    df["climate"] = kg_vals
    return df


MONTHS = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}

geocell_dir = (
    "data/geocells/finished_geocells"  # must be a DIRECTORY containing *.pickle
)
geocell_mgr = GeocellManager(geocell_dir)

if __name__ == "__main__":
    raster_path = "preprocessing/koppen_geiger_climatezones_1991_2020_1km.tif"
    out_path = "points_with_koppen.csv"

    df = load_latest_snapshot_df()
    df["month"] = df["batch_date"].str[5:7]
    df["latitude"] = df["lat"]
    df["longitude"] = df["lon"]
    out = df.apply(lambda r: geocell_mgr.get_geocell_id(r), axis=1)
    df[["cell", "country", "region"]] = pd.DataFrame(out.tolist(), index=df.index)
    df = sample_koppen(df, raster_path, CLIMATE_DICT)
    df.to_csv("pretrain_dataset.csv")
