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
POINT_PATHS = [
    "data/point_data",
]


class GenerateGeocells:
    def __init__(self):
        self.admin_2 = self.get_dataframe(FILEPATHS[1])
        self.countries = self.init_country_cells(FILEPATHS[2])
        self.admin_1 = self.init_admin_1_cells(FILEPATHS[0])

        self.points = self.init_points(POINT_PATHS[0])

        # Hvert land har først en cell som dekker hele landet, så resten av cellene i landet
        self.country_cells = {}
        # self.max_points = len(self.points)//10
        self.max_points = 5

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

    def init_points(self, filename):
        points = gpd.GeoDataFrame()
        for file in list(os.walk(filename))[0][2]:
            points = pd.concat(
                [points, gpd.GeoDataFrame.from_file(f"{filename}/{file}")]
            )

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
        print("Initializing cells")
        cells = []
        for i in trange(len(self.countries), desc="Celler av land: "):
            name = self.countries.iloc[i]["COUNTRY"]
            admin_1 = name
            country = name

            polygons = [j for j in self.countries.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            self.country_cells[country] = {country: [cell]}

        for i in trange(len(self.admin_1), desc="Celler av admin 1"):
            name = self.admin_1.iloc[i]["NAME_1"]
            admin_1 = name
            country = self.admin_1.iloc[i]["COUNTRY"]

            polygons = [j for j in self.admin_1.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            self.country_cells[country][admin_1] = [cell]

        for i in trange(len(self.admin_2), desc="Celler av admin 2"):
            name = self.admin_2.iloc[i]["NAME_2"]
            admin_1 = self.admin_2.iloc[i]["NAME_1"]
            country = self.admin_2.iloc[i]["COUNTRY"]

            polygons = [j for j in self.admin_2.iloc[i]["geometry"].geoms]
            cell = Cell(name, [], polygons, country, admin_1)

            self.country_cells[country][admin_1].append(cell)
            self.country_cells[country][country].append(cell)
            cells.append(cell)

        for i in trange(len(cells), desc="Legg til naboer", colour="GREEN"):
            cell = cells[i]
            cell.get_neighbours(self.country_cells[cell.country][cell.country][1:])
        return cells

    def add_points_to_cells(self):
        for i in trange(len(self.points), desc="Legg til punkter", colour="BLUE"):
            point = self.points.iloc[i]
            point_coords = [point["lng"], point["lat"]]
            # TODO legg til database med punkt
            if point["geocell"] is not None:
                continue
            for country in self.country_cells:
                if self.country_cells[country][country][0].contains(point_coords):
                    # print(self.country_cells[country])

                    for admin_1 in self.country_cells[country]:
                        if admin_1 == country:
                            continue
                        if self.country_cells[country][admin_1][0].contains(
                            point_coords
                        ):
                            for cell in self.country_cells[country][admin_1][1:]:
                                if cell.contains(point_coords):
                                    cell.add_point(point)
                                    point["geocell"] = cell.id
                                    break
                            break
                    break

    def generate_geocells(self):
        cells_to_combine = [i for i in self.cells if len(i) < self.max_points]
        for i in trange(len(cells_to_combine), desc="Slå sammen celler"):
            cell = cells_to_combine[i]
            total_points = len(cell)

            queue = [i for i in cell.neighbours]
            visited = set()
            while (total_points < self.max_points) and queue:
                cell_to_add = queue.pop(0)
                for j in cell_to_add.neighbours:
                    if j in visited:
                        continue
                    if j in queue:
                        continue
                    queue.append(j)

                visited.add(cell_to_add)
                total_points += len(cell_to_add)

            if visited:
                cell.combine(visited)

    def __str__(self):
        return f"{self.cells}"


if __name__ == "__main__":
    gen = GenerateGeocells(set())
