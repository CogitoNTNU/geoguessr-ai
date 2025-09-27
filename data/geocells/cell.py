import shapely.ops
import numpy as np
from typing import List


class Cell:
    def __init__(
        self, id: str, points: List, polygons: List, country: str, admin_1: str
    ):
        self.id = id
        self.points = points

        self.polygons = polygons

        self.country = country
        self.admin_1 = admin_1

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
            other.points = []
            other.points

    def split(self):
        pass

    def substract(self, other):
        pass

    def merge(self, other):
        pass

    def cluster(self):
        pass

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
