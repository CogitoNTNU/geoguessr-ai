from __future__ import annotations
import shapely.ops
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
from typing import List
from loguru import logger

from sklearn.cluster import OPTICS

import uuid

CRS = "EPSG:4326"


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
        self.geom_centroid = [
            self.current_shape.centroid.x,
            self.current_shape.centroid.y,
        ]

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
                for cluster in new_cell.clusters:
                    print(new_cell.id, len(new_cell.clusters[cluster]["points"]))

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
        print(self, self.points)
        return cells_made

    # def split_cell(self, add_to, cluster_args, min_cell_size, max_cell_size):
    #     if len(self.points) < max_cell_size:
    #         return []

    #     self.curr_coords = self.coords()
    #     self.point_centroid = [
    #         np.mean([x[1] for x in self.curr_coords]),
    #         np.mean([x[0] for x in self.curr_coords]),
    #     ]
    #     if self.curr_coords == 0:
    #         return []
    #     lats = []
    #     longs = []
    #     for i in self.curr_coords:
    #         lats.append(i[0])
    #         longs.append(i[1])
    #     data = [[point[0], point[1]] for point in self.curr_coords]

    #     df = pd.DataFrame(data=data, columns=["lat", "lng"])
    #     df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lat, df.lng), crs=CRS)

    #     clusterer = OPTICS(min_samples=cluster_args[0], xi=cluster_args[1])
    #     df["cluster"] = clusterer.fit_predict(df[["lat", "lng"]].values)

    #     uniques_clusters = df["cluster"].nunique()
    #     if uniques_clusters < 2:
    #         return []

    #     print(df)

    #     cluster_count = df["cluster"].value_counts()
    #     large_clusters = cluster_count[cluster_count >= min_cell_size].index
    #     non_null_large_clusters = [x for x in large_clusters if x != -1]

    #     if len(large_clusters) < 2:
    #         return []

    #     if len(large_clusters) == 2 and len(non_null_large_clusters) == 1:
    #         null_df = df[df["cluster"] == -1]
    #         if len(null_df) > max_cell_size:
    #             return []

    #         new_cells, remove_cells = self.separate_single_cluster(
    #             df, non_null_large_clusters
    #         )
    #     else:
    #         new_cells, remove_cells = self.separate_multi_cluster(
    #             df, non_null_large_clusters
    #         )

    #     for new_cell in new_cells:
    #         self.subtract(new_cell)
    #         add_to.append(new_cell)
    #     print(f"{remove_cells=}")
    #     for cell in remove_cells:
    #         if cell in add_to:
    #             # add_to.remove(cell)
    #             pass

    #     clean_cells = new_cells
    #     if len(remove_cells) == 0:
    #         clean_cells += [self]

    #     self.clean_dirty_splits(clean_cells)

    #     proc_cells = []
    #     if len(self.points) > max_cell_size and self not in remove_cells:
    #         proc_cells.append(self)

    #     for cell in new_cells:
    #         if len(cell.points) > max_cell_size:
    #             proc_cells.append(cell)

    #     return proc_cells

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
