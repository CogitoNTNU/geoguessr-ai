from __future__ import annotations
import shapely.ops
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
from typing import List
from loguru import logger

from sklearn.cluster import OPTICS

import uuid


class Cell:
    def __init__(
        self, id: str, points: List, polygons: List, country: str, admin_1: str
    ):
        self.id = id
        self.points = points

        self.curr_coords = self.coords()

        self.polygons = polygons

        self.current_shape = self.shape()
        self.country = country
        self.admin_1 = admin_1
        self.neighbours = set()

        self.point_centroid = np.array([0.0, 0.0])
        if len(self) > 0:
            self.point_centroid = [
                np.mean([x[1] for x in self.curr_coords]),
                np.mean([x[0] for x in self.curr_coords]),
            ]

        try:
            self.geom_centroid = [
                self.current_shape.centroid.x,
                self.current_shape.centroid.y,
            ]
        except Exception:
            self.geom_centroid = self.point_centroid

        self.clusters = {
            -1: {
                "points": self.points,
                "centroid": self.point_centroid,
                "hashes": [hash((p["latitude"], p["longitude"])) for p in points],
            }
        }

    def add_point(self, point):
        self.points.append(point)
        np.append(self.curr_coords, (point["latitude"], point["longitude"]))

    def get_neighbours(self, geocells):
        for cell in geocells:
            if cell == self:
                continue
            if cell.current_shape.intersects(self.current_shape):
                self.neighbours.add(cell)

    def shape(self):
        union = shapely.ops.unary_union(self.polygons)
        union = union.buffer(0)
        return union

    def coords(self):
        try:
            return [(p["latitude"], p["longitude"]) for p in self.points]
        except TypeError:
            return [(p.x, p.y) for p in self.points]

    def is_empty(self):
        return len(self) == 0

    def combine(self, others):
        for other in others.copy():
            if other == self:
                continue
            self.points += other.points
            self.polygons += other.polygons
            self.neighbours = self.neighbours.union(other.neighbours)
            other.points = []
            other.polygons = []
            other.point_centroid = np.array([0.0, 0.0])
            other.geom_centroid = np.array([0.0, 0.0])
            for n in other.neighbours:
                if other in n.neighbours:
                    n.neighbours.remove(other)
                n.neighbours.add(self)

            other.neighbours = set()
            if other in self.neighbours:
                self.neighbours.remove(other)
            if self in self.neighbours:
                self.neighbours.remove(self)

            other.current_shape = other.shape()

        self.current_shape = self.shape()
        self.curr_coords = self.coords()
        if len(self) > 0:
            self.point_centroid = [
                np.mean([x[1] for x in self.curr_coords]),
                np.mean([x[0] for x in self.curr_coords]),
            ]
        try: # Jens: Måtte modifsere blokken under for å bli kvitt en feil som skjedde for 4 land og blokkerte geocell-generation
            c = self.current_shape.centroid
            if c.is_empty:
                raise ValueError("empty centroid")
            self.geom_centroid = [c.x, c.y]
        except Exception:
            self.geom_centroid = self.point_centroid

    def separate_points(self, points, polygons, contain_points):
        coords = ((p["latitude"], p["longitude"]) for p in points)
        new_id = self.id + str(hash(coords))

        new_cell = Cell(new_id, points, polygons, self.country, self.admin_1)
        if contain_points and isinstance(new_cell.current_shape, MultiPolygon):
            new_cell.current_shape = Polygon(new_cell.current_shape.envelope)
        return new_cell

    def contains(self, point):
        try:
            return self.current_shape.contains(point)
        except TypeError:
            return self.current_shape.contains(Point(point[0], point[1]))
        except Exception as e:
            logger.warning(e)

    def split_cell(self):
        max_cluster_size = 10
        cells_made = []
        clusters_split = []
        for cluster in self.clusters:
            if len(self.clusters[cluster]["points"]) > max_cluster_size:
                new_cell = Cell(
                    self.id + str(cluster),
                    self.clusters[cluster]["points"],
                    [],
                    self.country,
                    self.admin_1,
                )

                clusters_split.append(cluster)

                new_cell.cluster(0.00005)

                cells_made.append(new_cell)

        points_to_keep = []

        point_clusters = []
        for clust in self.clusters:
            if clust not in clusters_split:
                point_clusters.append(clust)
        for clust in point_clusters:
            points_to_keep += self.clusters[clust]["points"]
            del self.clusters[clust]
        self.points = points_to_keep
        if len(self) > 0:
            self.curr_coords = self.coords()
            self.point_centroid = [
                np.mean([x[1] for x in self.curr_coords]),
                np.mean([x[0] for x in self.curr_coords]),
            ]
        return cells_made

    def to_list(self):
        return [
            self.id,
            self.admin_1,
            self.country,
            len(self),
            len(self.polygons),
            self.current_shape,
        ]

    def cluster(self, cluster_param=0.0005):
        min_sample = 5
        if len(self) < min_sample:
            return

        clustering = OPTICS(
            min_samples=min_sample, xi=cluster_param, min_cluster_size=0.05
        )
        clustering.fit(self.points)

        labels = clustering.labels_

        self.clusters = {}

        for i in range(len(labels)):
            if labels[i] not in self.clusters:
                self.clusters[int(labels[i])] = {
                    "points": [],
                    "centroid": np.array([0.0, 0.0]),
                    "hashes": [],
                }
            self.clusters[int(labels[i])]["points"].append(self.points[i])
            self.clusters[int(labels[i])]["hashes"].append(
                hash((self.points[i]["latitude"], self.points[i]["longitude"]))
            )

        for cluster in self.clusters:
            lat_mean = np.mean(
                [x["latitude"] for x in self.clusters[cluster]["points"]]
            )
            lng_mean = np.mean(
                [x["longitude"] for x in self.clusters[cluster]["points"]]
            )
            centroid = [lat_mean, lng_mean]
            self.clusters[cluster]["centroid"] = centroid

    def clean_cell_before_saving(self):
        self.current_shape = shapely.empty(1)
        self.polygons = []
        self.neighbours = []
        self.point_centroid = [
            np.mean([x[1] for x in self.curr_coords]),
            np.mean([x[0] for x in self.curr_coords]),
        ]

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __len__(self):
        return len(self.points)

    def __lt__(self, other):
        return len(self) < len(other)

    def __repr__(self):
        return f"\nGeocell {self.id}    | Number of points {len(self)}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        if hasattr(self, "id"):
            return hash(self.id)
        else:
            return hash(uuid.uuid4())


if __name__ == "__main__":
    cell = Cell("Hallo", [], [], "Norway", "Rogaland")
    print(hash(cell))
    print(cell.is_empty())
