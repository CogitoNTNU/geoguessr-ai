import shapely.ops
import numpy as np
from typing import List
import pandas as pd
import geopandas as gpd


class Cell:
    def __init__(
        self, id: str, points: List, polygons: List, country: str, admin_1: str
    ):
        self.id = id
        self.points = points

        self.polygons = polygons

        self.country = country
        self.admin_1 = admin_1

        self.neigbours = []

    def add_point(self, point):
        self.points.append(point)

    def get_neighbours(self, geocells):
        for cell in geocells:
            if cell == self:
                continue
            if cell.shape().intersects(self.shape()):
                self.neigbours.append(geocells.id)

    def shape(self):
        union = shapely.ops.unary_union(self.polygons)
        union = union.buffer(0)
        return union

    def coords(self):
        return np.ndarray([(p.x, p.y) for p in self.points])

    def is_empty(self):
        return len(self) == 0

    def combine(self, others):
        for other in others:
            self.points += other.points
            self.polygons += other.polygons
            self.neigbours += other.neighbours
            other.points = []
            other.polygons = []

            for n in other.neighbours:
                n.neighbours.remove(other.id)
                n.append(self.id)

            other.neighbours = []

            self.neigbours.remove(other.id)
            self.neigbours.remove(self.id)

    def split(self):
        pass

    def subtract(self, other):
        pass

    def contains(self, point):
        return self.shape().contains(point)

    def cluster(self):
        pass

    def to_pandas(self):
        data = [[self.id, p.x, p.y] for p in self.points]
        columns = ["id", "lng", "lat"]
        df = pd.DataFrame(data=data, columns=columns)
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="“EPSG:4326”"
        )
        return df

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"Geocell {self.id}    | Number of points {len(self)}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.id)


cell = Cell("Hallo", [], [], "Norway", "Rogaland")
print(hash(cell))
print(cell.is_empty())
