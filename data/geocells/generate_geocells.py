import pandas as pd
import os
from cell import Cell
from tqdm import trange
import pickle
import sqlite3
from shapely import wkb as shapely_wkb
import struct
import uuid


FILEPATHS = [
    "data/GADM_data/GADM_admin_1",
    "data/GADM_data/GADM_admin_2",
    "data/GADM_data/GADM_country",
    "data/geocells/finished_geocells",  # dir for geocell saving
    "data/GADM_data/gadm_world_all_levels.filtered_noadm345.gpkg",  # database with GADM data
]


class GenerateGeocells:
    def __init__(self, countries):
        self.init_sql_database()
        self.points = self.init_points_from_lat_lng_file(
            "data/out/sv_points_all_latlong.pkl"
        )

        self.COUNTRY = countries

        self.country_cells = {}
        self.min_points = 10
        self.max_points = 67

        self.cells = self.init_cells()
        self.add_points_to_cells()
        self.cells.sort(key=lambda x: -len(x.points))

        self.generate_geocells()

        # visualizer = geocell_visualizer.CellVisualizer(self)
        # visualizer.show()

        self.save_geocells(FILEPATHS[3])
        print("Saved geocells to file")

    def init_sql_database(self):
        sql = sqlite3.connect(FILEPATHS[4])

        self.admin_2 = pd.read_sql_query("SELECT * FROM ADM_2", sql)
        self.admin_1 = pd.read_sql_query("SELECT * FROM ADM_1", sql)
        self.countries = pd.read_sql_query("SELECT * FROM ADM_0", sql)

        sql.close()

    def init_points_from_lat_lng_file(self, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
        points = data

        return points

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

        for i in trange(len(self.countries), desc="Celler av land: "):
            name = self.countries.iloc[i]["COUNTRY"]
            admin_1 = name
            country = name

            if country in self.COUNTRY:
                polygons = [j for j in (self.countries.iloc[i]["geom"]).geoms]
                cell = Cell(name, [], polygons, country, admin_1)

                self.country_cells[country] = {country: [cell]}

        for i in trange(len(self.admin_1), desc="Celler av admin 1"):
            name = self.admin_1.iloc[i]["NAME_1"]
            admin_1 = name
            country = self.admin_1.iloc[i]["COUNTRY"]

            if country == "United Kingdom" and admin_1 == "NA":
                admin_1 = "England"
                name = "England"
            if country == "Ireland" and admin_1 == "NA":
                admin_1 = "Cork"
                name = "Cork"
            if country == "Netherlands" and admin_1 == "NA":
                admin_1 = "Zuid-Holland"
                name = "Zuid-Holland"

            if country in self.country_cells:
                polygons = [j for j in self.admin_1.iloc[i]["geom"].geoms]
                cell = Cell(name, [], polygons, country, admin_1)

                self.country_cells[country][admin_1] = [cell]

        for i in trange(len(self.admin_2), desc="Celler av admin 2"):
            name = self.admin_2.iloc[i]["NAME_2"] + str(uuid.uuid1())
            admin_1 = self.admin_2.iloc[i]["NAME_1"]
            country = self.admin_2.iloc[i]["COUNTRY"]

            if country == "United Kingdom" and admin_1 == "NA":
                admin_1 = "England"

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

            point_coords = [
                point["longitude"],
                point["latitude"],
            ]  # Veldig tabbe (feil vei på lat, lng), men dette funker

            for country in self.country_cells:
                if self.country_cells[country][country][0].contains(point_coords):
                    for admin_1 in self.country_cells[country]:
                        if admin_1 == country:
                            continue
                        if self.country_cells[country][admin_1][0].contains(
                            point_coords
                        ):
                            for cell in self.country_cells[country][admin_1][1:]:
                                if cell.contains(point_coords):
                                    cell.add_point(point)
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

    def cluster(self):
        for cell in self.cells:
            if len(cell) > 0:
                cell.cluster()

    def split_geocells(self):
        cells_to_split = [x for x in self.cells if len(x) > self.max_points]

        for cell_id in trange(len(cells_to_split), desc="Splitt celler"):
            # cell = heapq.heappop(cells_to_split)
            cell = cells_to_split[cell_id]

            cells_made = cell.split_cell()

            for cell in cells_made:
                self.cells.append(cell)
                self.country_cells[cell.country][cell.admin_1].append(cell)

            # for more_cell in more_cells:
            #     heapq.heappush(cells_to_split, more_cell)

    def generate_geocells(self):
        self.combine_geocells()
        self.cluster()
        self.split_geocells()

    def save_geocells(self, dir):
        for country in self.country_cells.keys():
            filepath = f"{dir}/geocells_{country}.pickle"
            with open(filepath, "wb") as f:
                cell_stripped = self.country_cells[country]
                out_cells = {}
                for admin1 in cell_stripped:
                    for cell in cell_stripped[admin1]:
                        if admin1 not in out_cells:
                            out_cells[admin1] = []
                        if len(cell) > 0:
                            cell.clean_cell_before_saving()
                            out_cells[admin1].append(cell)
                pickle.dump(out_cells, f)

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
