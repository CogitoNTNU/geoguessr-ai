import geopandas as gpd
import pandas as pd
import os
from cell import Cell
from tqdm import trange

COLS = ["data", "COUNTRY", "NAME_1", "NAME_2", "geometry"]


class GenerateGeocells:
    def __init__(self, filename, points):
        self.admin_2 = self.get_dataframe(filename)
        # TODO bytt punkt til å være i database, slik at de er koblet til bilder og har hvilken geocell de er i osv.
        # self.df = df
        # self.points = self.init_points()
        self.points = self.init_points(points)

        # Hvert land har først en cell som dekker hele landet, så resten av cellene i landet
        self.country_cells = {}
        self.countries = self.init_country_cells()
        self.admin_1 = self.init_admin_1_cells()

        self.cells = self.init_cells()
        self.add_points_to_cells()

        print(self.country_cells)

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

    def init_country_cells(self):
        pass

    def init_admin_1_cells(self):
        pass

    def init_cells(self):
        cells = []
        for i in range(len(self.admin_2)):
            name = self.admin_2.iloc[i]["NAME_2"]
            admin_1 = self.admin_2.iloc[i]["NAME_1"]
            country = self.admin_2.iloc[i]["COUNTRY"]

            polygons = [j for j in self.admin_2.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            if country not in self.country_cells:
                country_cell = [
                    count for count in self.countries if count.country == country
                ]

                self.country_cells[country] = {country: [country_cell]}
            else:
                self.country_cells[country][country][0].combine(
                    [Cell(name, [], polygons, country, admin_1)]
                )

            if admin_1 not in self.country_cells[country]:
                admin_1_cell = [
                    admin for admin in self.admin_1 if admin_1.admin_1 == admin_1
                ]

                self.country_cells[country][admin_1] = [admin_1_cell]
            else:
                self.country_cells[country][admin_1][0].combine(
                    [Cell(name, [], polygons, country, admin_1)]
                )

            self.country_cells[country][admin_1].append(cell)
            cells.append(cell)

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
                    print("country")

                    for admin_1 in self.country_cells[country]:
                        print(self.country_cells[country][admin_1])
                        if admin_1 == country:
                            continue
                        if self.country_cells[country][admin_1][0].contains(point):
                            print("admin1")
                            print(self.country_cells[country][admin_1][1:])
                            for cell in self.country_cells[country][admin_1][1:]:
                                if cell.contains(point):
                                    print("cell")
                                    cell.add_point(point)
                                    # TODO legg til hvilken geocell et punkt er i
                                    break
                            break
                    break

    def generate_geocells(self):
        pass

    def __str__(self):
        return f"{self.cells}"


gen = GenerateGeocells("data/GADM_data/GADM_admin_2", set())
