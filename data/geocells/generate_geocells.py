import geopandas as gpd
import pandas as pd
import os
from cell import Cell
from tqdm import trange

COLS = ["data", "COUNTRY", "NAME_1", "NAME_2", "geometry"]
COUNTRY_COLS = ["data", "COUNTRY", "geometry"]
ADMIN_1_COLS = ["data", "COUNTRY", "NAME_1", "geometry"]

FILEPATHS = [
    "data/GADM_data/GADM_admin_1",
    "data/GADM_data/GADM_admin_2",
    "data/GADM_data/GADM_country",
]


class GenerateGeocells:
    def __init__(self, points):
        self.admin_2 = self.get_dataframe(FILEPATHS[1])
        self.countries = self.init_country_cells(FILEPATHS[2])
        self.admin_1 = self.init_admin_1_cells(FILEPATHS[0])
        # TODO bytt punkt til å være i database, slik at de er koblet til bilder og har hvilken geocell de er i osv.
        # self.df = df
        # self.points = self.init_points()
        self.points = self.init_points(points)

        # Hvert land har først en cell som dekker hele landet, så resten av cellene i landet
        self.country_cells = {}

        self.cells = self.init_cells()
        self.add_points_to_cells()
        self.cells.sort(key=lambda x: -len(x.points))

        # print(self.country_cells)
        self.generate_geocells()

    def get_dataframe(self, filename):
        df = gpd.GeoDataFrame()
        for file in list(os.walk(filename))[0][2]:
            df = pd.concat([df, gpd.GeoDataFrame.from_file(f"{filename}/{file}")])

        keep_cols = [i for i in df.columns if i in COLS]
        df = df[keep_cols].copy()

        return df

    def init_points(self, points):
        # TODO fiks dette med database
        points = [point for point in points]
        return points

    def init_country_cells(self, filename):
        country_df = gpd.GeoDataFrame()
        for file in list(os.walk(filename))[0][2]:
            country_df = pd.concat(
                [country_df, gpd.GeoDataFrame.from_file(f"{filename}/{file}")]
            )

        keep_cols = [i for i in country_df if i in COUNTRY_COLS]
        country_df = country_df[keep_cols].copy()

        return country_df

    def init_admin_1_cells(self, filename):
        admin_1_df = gpd.GeoDataFrame()
        for file in list(os.walk(filename))[0][2]:
            admin_1_df = pd.concat(
                [admin_1_df, gpd.GeoDataFrame.from_file(f"{filename}/{file}")]
            )

        keep_cols = [i for i in admin_1_df if i in ADMIN_1_COLS]
        admin_1_df = admin_1_df[keep_cols].copy()

        return admin_1_df

    def init_cells(self):
        cells = []
        for i in range(len(self.countries)):
            name = self.countries.iloc[i]["COUNTRY"]
            admin_1 = name
            country = name

            polygons = [j for j in self.countries.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            self.country_cells[country] = {country: [cell]}

        for i in range(len(self.admin_1)):
            name = self.admin_1.iloc[i]["NAME_1"]
            admin_1 = name
            country = self.admin_1.iloc[i]["COUNTRY"]

            polygons = [j for j in self.admin_1.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            self.country_cells[country][admin_1] = [cell]

        for i in range(len(self.admin_2)):
            name = self.admin_2.iloc[i]["NAME_2"]
            admin_1 = self.admin_2.iloc[i]["NAME_1"]
            country = self.admin_2.iloc[i]["COUNTRY"]

            polygons = [j for j in self.admin_2.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            self.country_cells[country][admin_1].append(cell)
            self.country_cells[country][country].append(cell)
            cells.append(cell)

        for i in trange(len(cells)):
            cell = cells[i]
            cell.get_neighbours(self.country_cells[cell.country][cell.country][1:])
        return cells

    def add_points_to_cells(self):
        for i in trange(len(self.points)):
            point = self.points[i]
            # TODO legg til database med punkt
            # if point.geocell != None:
            #     continue
            for country in self.country_cells:
                if self.country_cells[country][country][0].contains(point):
                    # print(self.country_cells[country])

                    for admin_1 in self.country_cells[country]:
                        if admin_1 == country:
                            continue
                        if self.country_cells[country][admin_1][0].contains(point):
                            for cell in self.country_cells[country][admin_1][1:]:
                                if cell.contains(point):
                                    cell.add_point(point)
                                    # TODO legg til hvilken geocell et punkt er i
                                    break
                            break
                    break

    def generate_geocells(self):
        for cell in self.cells[0:10]:
            print(len(cell.points))

    def __str__(self):
        return f"{self.cells}"


if __name__ == "__main__":
    gen = GenerateGeocells(set())
