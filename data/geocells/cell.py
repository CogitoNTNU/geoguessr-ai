import shapely.ops
from shapely.geometry import Point
import numpy as np
from typing import List
import pandas as pd
import geopandas as gpd
from loguru import logger


class Cell:
    def __init__(
        self, id: str, points: List, polygons: List, country: str, admin_1: str
    ):
        self.id = id
        self.points = points

        self.polygons = polygons

        self.current_shape = self.shape()
        self.country = country
        self.admin_1 = admin_1

        self.neighbours = []

    def add_point(self, point):
        self.points.append(point)

    def get_neighbours(self, geocells):
        for cell in geocells:
            if cell == self:
                continue
            if cell.current_shape.intersects(self.current_shape):
                self.neighbours.append(cell)

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
            self.neighbours += other.neighbours
            other.points = []
            other.polygons = []

            for n in other.neighbours:
                if other in n.neighbours:
                    n.neighbours.remove(other)
                n.neighbours.append(self)

            other.neighbours = []
            if other in self.neighbours:
                self.neighbours.remove(other)
            if self in self.neighbours:
                self.neighbours.remove(self)

            other.current_shape = other.shape()

        self.current_shape = self.shape()

    def split(self):
        pass

    def subtract(self, other):
        pass

    def contains(self, point):
        try:
            return self.current_shape.contains(point)
        except TypeError:
            return self.current_shape.contains(Point(point.lng, point.lat))
        except Exception as e:
            logger.warning(e)

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


if __name__ == "__main__":
    cell = Cell("Hallo", [], [], "Norway", "Rogaland")
    print(hash(cell))
    print(cell.is_empty())
