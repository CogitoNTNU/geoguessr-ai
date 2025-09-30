import geopandas as gpd
import pandas as pd
import os

COLS = ["data", "COUNTRY", "NAME_1", "NAME_2", "geometry"]


class AdminCell:
    def __init__(self, filename):
        self.df = gpd.GeoDataFrame()
        for file in list(os.walk(filename))[0][2]:
            self.df = pd.concat(
                [self.df, gpd.GeoDataFrame.from_file(f"{filename}/{file}")]
            )

        keep_cols = [i for i in self.df.columns if i in COLS]
        self.df = self.df[keep_cols].copy()

    def points(self):
        points = self.df()
        return points

    def __str__(self):
        return f"data{self.df}"


if __name__ == "__main__":
    admin_visualize = AdminCell("data/GADM_data/")

    print(admin_visualize)
