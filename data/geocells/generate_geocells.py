import geopandas as gpd
import pandas as pd
import os
from cell import Cell
from tqdm import trange
import pickle
import heapq
import sqlite3
from shapely import wkb as shapely_wkb
import struct


COLS = ["data", "COUNTRY", "NAME_1", "NAME_2", "geometry"]
COUNTRY_COLS = ["data", "COUNTRY", "geometry"]
ADMIN_1_COLS = ["data", "COUNTRY", "NAME_1", "geometry"]

COUNTRY = ["Norway"]

FILEPATHS = [
    "data/GADM_data/GADM_admin_1",
    "data/GADM_data/GADM_admin_2",
    "data/GADM_data/GADM_country",
    "data/geocells/finished_geocells",
]
POINT_PATHS = [
    "data/point_data",
]


class GenerateGeocells:
    def __init__(self):
        self.init_sql_database()
        # self.admin_2 = self.get_dataframe(FILEPATHS[1])
        # self.countries = self.init_country_cells(FILEPATHS[2])
        # self.admin_1 = self.init_admin_1_cells(FILEPATHS[0])

        # self.points = self.init_points(POINT_PATHS[0])
        self.points = self.init_points_from_lat_lng_file(
            "data/out/sv_points_all_latlong.pkl"
        )
        self.country_cells = {}
        # self.max_points = len(self.points)//10
        self.min_points = 5
        self.max_points = 50

        self.cells = self.init_cells()
        self.add_points_to_cells()
        self.cells.sort(key=lambda x: -len(x.points))

        self.generate_geocells()
        self.generate_geocells()

        self.save_geocells(FILEPATHS[3])
        print("Saved geocells to file")
        # self.cells = []
        # self.country_cells = {}

        # self.load_geocells(FILEPATHS[3])

        # print(self.country_cells)
        # for country in self.country_cells:
        #     for admin_1 in self.country_cells[country]:
        #         for cell in self.country_cells[country][admin_1]:
        #             if len(cell) > 0:
        #                 self.cells.append(cell)

    def init_sql_database(self):
        sql = sqlite3.connect(
            "data/GADM_data/gadm_world_all_levels.filtered_noadm345.gpkg"
        )
        self.admin_2 = pd.read_sql_query("SELECT * FROM ADM_2", sql)
        self.admin_1 = pd.read_sql_query("SELECT * FROM ADM_1", sql)
        self.countries = pd.read_sql_query("SELECT * FROM ADM_0", sql)

        print(self.countries["geom"][1])

        sql.close()

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

    def init_points_from_lat_lng_file(self, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
        points = data

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

    def parse_gpkg_blob(self, blob: bytes):
        """
        Parse a GeoPackage geometry BLOB (starts with b'GP') and return a dict containing:
        - is_gpkg: bool
        - wkb: bytes (the extracted WKB)
        - srs_id: int
        - envelope_bytes: int
        - version, flags, header_little_endian, envelope_indicator
        - geometry: Shapely geometry (if shapely can parse), otherwise not present
        - geometry_error: error message if shapely failed
        """
        if blob is None:
            raise ValueError("None blob")
        if not isinstance(blob, (bytes, bytearray)):
            # not bytes: assume it's already a geometry object or WKT/etc
            return {
                "is_gpkg": False,
                "wkb": blob,
                "srs_id": None,
                "envelope_bytes": 0,
                "version": None,
                "flags": None,
            }

        if len(blob) < 8:
            raise ValueError("Blob too short to be a GeoPackage geometry")

        if blob[0:2] != b"GP":
            # Not a GeoPackage header: assume it's raw WKB already
            return {
                "is_gpkg": False,
                "wkb": bytes(blob),
                "srs_id": None,
                "envelope_bytes": 0,
                "version": None,
                "flags": None,
            }

        # Parse header
        version = blob[2]
        flags = blob[3]
        # bit 0 = byte order for header (0 big endian, 1 little endian)
        header_little_endian = bool(flags & 0x01)
        # envelope indicator = bits 1-3
        envelope_indicator = (flags >> 1) & 0x07
        # envelope bytes mapping per GeoPackage spec:
        envelope_len_map = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
        if envelope_indicator not in envelope_len_map:
            raise ValueError(f"invalid envelope indicator: {envelope_indicator}")
        envelope_bytes = envelope_len_map[envelope_indicator]

        # srs_id (int32) uses header byte order
        byteorder = "<" if header_little_endian else ">"
        srs_id = struct.unpack(f"{byteorder}i", blob[4:8])[0]

        wkb_start = 8 + envelope_bytes
        if len(blob) <= wkb_start:
            raise ValueError("No WKB bytes present after header+envelope")

        wkb = bytes(blob[wkb_start:])

        out = {
            "is_gpkg": True,
            "wkb": wkb,
            "srs_id": srs_id,
            "envelope_bytes": envelope_bytes,
            "version": version,
            "flags": flags,
            "header_little_endian": header_little_endian,
            "envelope_indicator": envelope_indicator,
        }

        # Try to load shapely geometry (WKB starts with its own endian byte)
        try:
            geom = shapely_wkb.loads(wkb)
            out["geometry"] = geom
        except Exception as e:
            out["geometry_error"] = str(e)

        return out

    def init_cells(self):
        print("Initializing cells")
        cells = []

        self.countries["geom"] = self.countries["geom"].apply(
            lambda g: self.parse_gpkg_blob(g)["geometry"]
        )

        self.admin_1["geom"] = self.admin_1["geom"].apply(
            lambda g: self.parse_gpkg_blob(g)["geometry"]
        )

        self.admin_2["geom"] = self.admin_2["geom"].apply(
            lambda g: self.parse_gpkg_blob(g)["geometry"]
        )

        print(self.countries)

        for i in trange(len(self.countries), desc="Celler av land: "):
            name = self.countries.iloc[i]["COUNTRY"]
            admin_1 = name
            country = name

            if country in COUNTRY:
                polygons = [j for j in (self.countries.iloc[i]["geom"]).geoms]
                cell = Cell(name, [], polygons, country, admin_1)

                self.country_cells[country] = {country: [cell]}

        for i in trange(len(self.admin_1), desc="Celler av admin 1"):
            name = self.admin_1.iloc[i]["NAME_1"]
            admin_1 = name
            country = self.admin_1.iloc[i]["COUNTRY"]

            if country in self.country_cells:
                polygons = [j for j in self.admin_1.iloc[i]["geom"].geoms]
                cell = Cell(name, [], polygons, country, admin_1)

                self.country_cells[country][admin_1] = [cell]

        for i in trange(len(self.admin_2), desc="Celler av admin 2"):
            name = self.admin_2.iloc[i]["NAME_2"]
            admin_1 = self.admin_2.iloc[i]["NAME_1"]
            country = self.admin_2.iloc[i]["COUNTRY"]

            if country in self.country_cells:
                polygons = [j for j in self.admin_2.iloc[i]["geom"].geoms]
                cell = Cell(name, [], polygons, country, admin_1)

                if admin_1 not in self.country_cells[country]:
                    self.country_cells[country][admin_1] = []

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
            point_coords = [point["longitude"], point["latitude"]]

            # if point["geocell"] is not None:
            #   continue
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
                                    # point["geocell"] = cell.id
                                    break
                            break
                    break

    def combine_geocells(self):
        cells_to_combine = [i for i in self.cells if len(i) < self.min_points]
        for i in trange(len(cells_to_combine), desc="Slå sammen celler"):
            cell = cells_to_combine[i]
            total_points = len(cell)

            queue = [i for i in cell.neighbours]
            visited = set()
            while (total_points < self.min_points) and queue:
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

    def split_geocells(self):
        cells_to_split = [x for x in self.cells if len(x) > self.max_points]
        new_cells = []
        cluster_args = [
            (50, 0.005),
            (400, 0.005),
            (1000, 0.0001),
        ]  # Taken from paper, but should find better params
        chosen_cluster_args = cluster_args[0]

        while cells_to_split:
            cell = heapq.heappop(cells_to_split)

            more_cells = cell.split_cell(
                new_cells, chosen_cluster_args, self.min_points, self.max_points
            )
            for more_cell in more_cells:
                heapq.heappush(cells_to_split, more_cell)

        print(f"{new_cells[0].current_shape=}")
        self.cells += new_cells

    def generate_geocells(self):
        self.combine_geocells()
        # visualizer = geocell_visualizer.CellVisualizer(self)
        # visualizer.show()
        self.split_geocells()

    def save_geocells(self, dir):
        for country in self.country_cells.keys():
            filepath = f"{dir}/geocells_{country}.pickle"
            with open(filepath, "wb") as f:
                pickle.dump(self.country_cells[country], f)

    def load_geocells(self, dir):
        for file in list(os.walk(dir))[0][2]:
            carved_country_name = file.split("_")[-1].split(".")[0]

            with open(dir + "/" + file, "rb") as f:
                data = pickle.load(f)
                self.country_cells[carved_country_name] = data

    def __str__(self):
        return f"{self.cells}"


if __name__ == "__main__":
    gen = GenerateGeocells(set())
