from __future__ import annotations
import shapely.ops
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
from typing import List
import pandas as pd
import geopandas as gpd
from loguru import logger

from shapely.errors import TopologicalError
from scipy.spatial import Voronoi
from voronoi_polygon_2d import voronoi_finite_polygons_2d
from sklearn.cluster import OPTICS

CRS = "EPSG:4326"
GEOCELL_COLUMNS = ["name", "admin_1", "country", "size", "num_polygons", "geometry"]


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

    def add_point(self, point):
        self.points.append(point)
        try:
            self.curr_coords = self.coords()
        except Exception:
            print(f"{point} HLLLL")

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
        if len(self.points) > 0:
            print(self.points[0], self.points[0]["latitude"])
        return np.ndarray([(p["latitude"], p["longitude"]) for p in self.points])

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

    def split(self):
        pass

    def subtract(self, other: Cell):
        try:
            difference_shape = self.current_shape.difference(other.current_shape)
        except TopologicalError as e:
            print(f"Error when subtracting {other.id} from {self.id}")
            raise TopologicalError(e)
        self.polygons = [difference_shape.buffer(0)]

        s_tuple = {(p["latitude"], p["longitude"]) for p in self.points}
        o_tuple = {(p["latitude"], p["longitude"]) for p in other.points}

        diff_tups = s_tuple - o_tuple

        self.points = [
            p for p in self.points if (p["latitude"], p["longitude"]) in diff_tups
        ]

    def separate_points(self, points, polygons, contain_points):
        coords = ((p["latitude"], p["longitude"]) for p in points)
        new_id = self.name + str(hash(coords)[:12])

        new_cell = Cell(new_id, points, polygons, self.country, self.admin1)
        if contain_points and isinstance(new_cell.current_shape, MultiPolygon):
            new_cell.current_shape = Polygon(new_cell.current_shape.exterior)
        return new_cell

    def contains(self, point):
        try:
            return self.current_shape.contains(point)
        except TypeError:
            return self.current_shape.contains(Point(point[0], point[1]))
        except Exception as e:
            logger.warning(e)

    def voronoi_polygons(self, coords: np.ndarray = None):
        if coords is None:
            voronoi_coords = np.unique(self.curr_coords, axis=0)
        else:
            voronoi_coords = np.unique(coords, axis=0)

        voronoi = Voronoi(voronoi_coords)
        regions, vertices = voronoi_finite_polygons_2d(voronoi)

        polygons = []
        for reg in regions:
            polygon = Polygon(vertices[reg])
            polygons.append(polygon)

        try:
            polygons = [p.intersection(self.current_shape) for p in polygons]
        except TopologicalError as e:
            print(f"Error in voronoi_polygons in cell {self.id}")
            raise TopologicalError(e)

        df = pd.DataFrame({"geometry": polygons})
        df = gpd.GeoDataFrame(df, geometry="geometry")
        points = (
            [Point(p[0], p[1]) for p in coords] if coords is not None else self.points
        )
        idxs = df.sindex.nearest(points, return_all=False)[1]

        return [polygons[i] for i in idxs]

    def separate_single_cluster(self, df: pd.DataFrame, cluster=0):
        polygons = self.voronoi_polygons()
        cluster_df = df[df["cluster"] == cluster][["lng", "lat"]]

        assert len(cluster_df.index) > 0, "No cluster found in dataframe"

        cluster_points = [self.points[i] for i in cluster_df.index]
        cluster_polygons = [polygons[i] for i in cluster_df.index]

        new_cell = self.separate_points(
            cluster_points, cluster_polygons, contain_points=True
        )
        return [new_cell], []

    def separate_multi_cluster(self, df: pd.DataFrame, clusters):
        assigned_df = df[df["cluster"].isin(clusters)]
        unassigned_df = df[not df["cluster"].isin(clusters)]
        cc = assigned_df.groupby(["cluster"])[["lng", "lat"]].mean().reset_index()
        cc = gpd.GeoDataFrame(cc, geometry=gpd.points_from_xy(cc.lng, cc.lat), crs=CRS)

        nearest_index = cc.sindex.nearest(unassigned_df.geometry, return_all=False)[1]
        df.loc[not df["cluster"].isin(clusters), "cluster"] = cc.iloc[nearest_index][
            "cluster"
        ].values

        if len(cc.index) == 2:
            return self.separate_single_cluster(df, cluster=cc.iloc[0]["cluster"])
        else:
            polygons = self.voronoi_polygons(coords=cc[["lng", "lat"]].values)

            new_cells = []
            for cluster, polygon in zip(cc["cluster"].unique(), polygons):
                cluster_coords = df[df["cluster"] == cluster][["lng", "lat"]]
                cluster_points = [
                    Point(row.lng, row.lat) for _, row in cluster_coords.iterrows()
                ]
                new_cell = self.separate_points(
                    cluster_points, [polygon], contain_points=True
                )
                new_cells.append(new_cell)
            return new_cells, [self]

    def split_cell(self, add_to, cluster_args, min_cell_size, max_cell_size):
        if len(self.points) < max_cell_size:
            return []

        print(self.curr_coords)
        df = pd.DataFrame(data=[self.curr_coords], columns=["lng", "lat"])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs=CRS)

        clusterer = OPTICS(min_samples=cluster_args[0], xi=cluster_args[1])
        df["cluster"] = clusterer.fit_predict(df[["lng", "lat"]].values)

        uniques_clusters = df["cluster"].nunique()
        if uniques_clusters < 2:
            return []

        cluster_count = df["cluster"].value_counts()
        large_clusters = cluster_count[cluster_count >= min_cell_size].index
        non_null_large_clusters = [x for x in large_clusters if x != -1]

        if len(large_clusters) < 2:
            return []

        if len(large_clusters) == 2 and len(non_null_large_clusters) == 1:
            null_df = df[df["cluster"] == -1]
            if len(null_df) > max_cell_size:
                return []

            new_cells, remove_cells = self.separate_single_cluster(
                df, non_null_large_clusters
            )
        else:
            new_cells, remove_cells = self.separate_multi_cluster(
                df, non_null_large_clusters
            )

        for new_cell in new_cells:
            self.subtract(new_cell)
            add_to.add(new_cell)

        for cell in remove_cells:
            add_to.remove(cell)

        clean_cells = new_cells
        if len(remove_cells) == 0:
            clean_cells += self

        self.clean_dirty_splits(clean_cells)

        proc_cells = []
        if len(self.points) > max_cell_size and self not in remove_cells:
            proc_cells.append(self)

        for cell in new_cells:
            if len(cell.points) > max_cell_size:
                proc_cells.append(cell)

        return proc_cells

    def clean_dirty_splits(self, cells):
        df = pd.DataFrame(data=[x.tolist() for x in cells], columns=GEOCELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)

        multi_polys = df[df["geometry"].type == "MultiPolygon"]

        for index, row in multi_polys.iterrows():
            points = cells[index].to_pandas()["geometry"]

            all_polygons = list(row["geometry"].geoms)

            largest_poly = max(all_polygons, key=lambda polygon: polygon.area)

            did_assign = False
            for small_poly in all_polygons:
                if small_poly != largest_poly:
                    small_poly_gseries = gpd.GeoSeries(
                        [small_poly], index=[index], crs=CRS
                    )

                    other_polys = df.drop(index)
                    buffered_poly = small_poly_gseries.buffer(0.01)
                    intersecting_polys = other_polys[
                        other_polys.intersects(buffered_poly.unary_union)
                    ]

                    if len(intersecting_polys) == 0:
                        continue

                    did_assign = True

                    largest_intersect_index = intersecting_polys.geometry.apply(
                        lambda poly: poly.intersection(buffered_poly.unary_union).area
                    ).idxmax()

                    mask = points.within(small_poly)
                    points_in_small_poly = points[mask]
                    cells[index].points = [
                        x for x in cells[index].points if x not in points_in_small_poly
                    ]
                    cells[largest_intersect_index].add_points(points_in_small_poly)

                    cells[largest_intersect_index].polygons = [
                        cells[largest_intersect_index].current_shape.union(small_poly)
                    ]
                if did_assign:
                    cells[index].polygons = [largest_poly]

    def cluster(self):
        pass

    def to_pandas(self):
        data = [[self.id, p["latitude"], p["longitude"]] for p in self.points]
        columns = ["id", "lng", "lat"]
        df = pd.DataFrame(data=data, columns=columns)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs=CRS)
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
