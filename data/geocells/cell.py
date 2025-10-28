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

import uuid

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
        self.centroid = None
        if len(self) > 0:
            self.centroid = [
                np.mean([x[0] for x in self.curr_coords]),
                np.mean([x[1] for x in self.curr_coords]),
            ]

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
            self.centroid = [
                np.mean([x[0] for x in self.curr_coords]),
                np.mean([x[1] for x in self.curr_coords]),
            ]

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

    """
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
        print(f"{polygons=}")

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
        print(f"{points[0]}")
        q_array = np.asarray(points, dtype=Point)
        print(q_array)
        idxs = df.sindex.nearest(q_array, return_all=False)[1]
        print("{idxs=}")

        return [polygons[i] for i in idxs]
    """

    def voronoi_polygons(self, coords: np.ndarray = None):
        # choose input coords for Voronoi (unique to avoid degenerate points)
        if coords is None:
            if getattr(self, "curr_coords", None) is None:
                raise ValueError("No coords provided and self.curr_coords is missing")

            voronoi_coords = np.unique(self.curr_coords, axis=0)
            temp = voronoi_coords[:, 0].copy()
            voronoi_coords[:, 0] = voronoi_coords[:, 1]
            voronoi_coords[:, 1] = temp
        else:
            voronoi_coords = np.unique(np.asarray(coords), axis=0)
            temp = voronoi_coords[:, 0].copy()
            voronoi_coords[:, 0] = voronoi_coords[:, 1]
            voronoi_coords[:, 1] = temp
        print(voronoi_coords)

        # need at least 2 or 3 points for a Voronoi; let Voronoi raise if invalid
        voronoi = Voronoi(voronoi_coords)
        regions, vertices = voronoi_finite_polygons_2d(voronoi)

        polygons = []
        for reg in regions:
            # regions returned by voronoi_finite_polygons_2d should be valid indices
            poly = Polygon(vertices[reg])
            polygons.append(poly)

        # intersect with current cell shape (defensive: propagate readable error)
        try:
            pass
            # polygons = [p.intersection(self.current_shape) for p in polygons]

        except TopologicalError:
            print(f"Error in voronoi_polygons in cell {getattr(self, 'id', 'unknown')}")
            raise

        # build GeoDataFrame of polygons
        gdf = gpd.GeoDataFrame({"geometry": polygons}, geometry="geometry")

        # Build list of query points (shapely Point objects), in the order we want results for:
        if coords is not None:
            # coords may be a sequence of (x,y) pairs or an ndarray
            points = [Point(float(x), float(y)) for x, y in np.asarray(coords)]
        else:
            # assume self.points is an iterable of shapely Point objects or (x,y) pairs
            # normalize: if self.points contains tuples, convert to Point()
            pts = list(self.points)
            if len(pts) > 0 and not isinstance(pts[0], Point):
                points = [
                    Point(float(i["latitude"]), float(i["longitude"])) for i in pts
                ]
            else:
                points = pts

        if len(points) == 0:
            return []

        # IMPORTANT: make a 1-D numpy array of dtype=object (what STRtree expects)
        query_array = np.asarray(points, dtype=object)

        # call spatial index nearest. Different geopandas/shapely versions return different formats;
        # handle both tuple (query_idxs, target_idxs) and plain array.
        result = gdf.sindex.nearest(query_array, return_all=False)

        def _nearest_for_single(pt):
            """Helper: query sindex for a single point, normalize return."""
            res = gdf.sindex.nearest(np.asarray([pt], dtype=object), return_all=False)
            if isinstance(res, tuple) and len(res) == 2:
                # res[1] should contain target idx array
                target = res[1]
                if hasattr(target, "__len__") and len(target) > 0:
                    return int(target[0])
                else:
                    # no target returned
                    raise RuntimeError(
                        "sindex.nearest returned no target for single query"
                    )
            else:
                arr = np.asarray(res, dtype=int)
                if arr.size > 0:
                    return int(arr[0])
                else:
                    raise RuntimeError("sindex.nearest returned empty for single query")

        # Normalize the result to a 1-D array of target indices
        try:
            if isinstance(result, tuple) and len(result) == 2:
                # Expect (query_idxs, target_idxs)
                _, target_idxs = result
                idxs = np.asarray(target_idxs, dtype=int)
            else:
                idxs = np.asarray(result, dtype=int)

            # If idxs is 2-D, prefer the first column but only if it exists
            if idxs.ndim > 1:
                if idxs.shape[1] > 0:
                    idxs = idxs[:, 0]
                else:
                    # empty second axis -> treat as invalid/misaligned
                    raise ValueError("nearest returned a 2-D array with zero columns")

            # If the length doesn't match the number of queries, fall back
            if idxs.shape[0] != query_array.shape[0] or idxs.size == 0:
                raise ValueError("nearest result length mismatch or empty")

        except Exception:
            # fallback: query per point (slower) but guaranteed to return a mapping
            idxs_list = []
            for pt in query_array:
                try:
                    i = _nearest_for_single(pt)
                except Exception:
                    # If even single queries fail, as a last resort try a geometry contain test:
                    contains = np.where(gdf.geometry.contains(pt))[0]
                    if len(contains) > 0:
                        i = int(contains[0])
                    else:
                        # no containing polygon and nearest failed — append -1 as sentinel
                        i = -1
                idxs_list.append(i)
            idxs = np.asarray(idxs_list, dtype=int)

        # if any -1 sentinel present, you may want to handle or raise
        if np.any(idxs < 0):
            # handle points with no nearest polygon found
            # e.g., raise RuntimeError or do something else
            # raise RuntimeError("One or more query points had no matching polygon")
            pass

        # now safe to index polygons
        print(f"{polygons=}")
        return [polygons[i] for i in idxs]

    def separate_single_cluster(self, df: pd.DataFrame, cluster=0):
        polygons = self.voronoi_polygons()
        cluster_df = df[df["cluster"] == cluster][["lat", "lng"]]

        assert len(cluster_df.index) > 0, "No cluster found in dataframe"

        print(polygons)
        print(cluster_df.index)

        cluster_points = [self.points[i] for i in cluster_df.index]
        cluster_polygons = [polygons[i] for i in cluster_df.index]

        new_cell = self.separate_points(
            cluster_points, cluster_polygons, contain_points=True
        )
        return [new_cell], []

    def separate_multi_cluster(self, df: pd.DataFrame, clusters):
        assigned_df = df[df["cluster"].isin(clusters)]
        unassigned_df = df[~df["cluster"].isin(clusters)]
        cc = assigned_df.groupby(["cluster"])[["lat", "lng"]].mean().reset_index()
        cc = gpd.GeoDataFrame(cc, geometry=gpd.points_from_xy(cc.lng, cc.lat), crs=CRS)

        nearest_index = cc.sindex.nearest(unassigned_df.geometry, return_all=False)[1]
        df.loc[~df["cluster"].isin(clusters), "cluster"] = cc.iloc[nearest_index][
            "cluster"
        ].values

        if len(cc.index) == 2:
            return self.separate_single_cluster(df, cluster=cc.iloc[0]["cluster"])
        else:
            polygons = self.voronoi_polygons(coords=cc[["lat", "lng"]].values)

            new_cells = []
            for cluster, polygon in zip(cc["cluster"].unique(), polygons):
                cluster_coords = df[df["cluster"] == cluster][["lat", "lng"]]
                print("Her kommer cluster_coords: ")
                print(cluster_coords)
                cluster_points = [
                    {"id": id, "latitude": row["lat"], "longitude": row["lng"]}
                    for id, row in cluster_coords.iterrows()
                ]
                new_cell = self.separate_points(
                    cluster_points, [polygon], contain_points=True
                )
                new_cells.append(new_cell)
            return new_cells, [self]

    def split_cell(self, add_to, cluster_args, min_cell_size, max_cell_size):
        if len(self.points) < max_cell_size:
            return []

        self.curr_coords = self.coords()
        self.centroid = [
            np.mean([x[0] for x in self.curr_coords]),
            np.mean([x[1] for x in self.curr_coords]),
        ]
        if self.curr_coords == 0:
            return []
        lats = []
        longs = []
        for i in self.curr_coords:
            lats.append(i[0])
            longs.append(i[1])
        data = [[point[0], point[1]] for point in self.curr_coords]

        df = pd.DataFrame(data=data, columns=["lat", "lng"])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lat, df.lng), crs=CRS)

        clusterer = OPTICS(min_samples=cluster_args[0], xi=cluster_args[1])
        df["cluster"] = clusterer.fit_predict(df[["lat", "lng"]].values)

        uniques_clusters = df["cluster"].nunique()
        if uniques_clusters < 2:
            return []

        print(df)

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
            add_to.append(new_cell)
        print(f"{remove_cells=}")
        for cell in remove_cells:
            if cell in add_to:
                # add_to.remove(cell)
                pass

        clean_cells = new_cells
        if len(remove_cells) == 0:
            clean_cells += [self]

        self.clean_dirty_splits(clean_cells)

        proc_cells = []
        if len(self.points) > max_cell_size and self not in remove_cells:
            proc_cells.append(self)

        for cell in new_cells:
            if len(cell.points) > max_cell_size:
                proc_cells.append(cell)

        return proc_cells

    def to_list(self):
        return [
            self.id,
            self.admin_1,
            self.country,
            len(self),
            len(self.polygons),
            self.current_shape,
        ]

    def clean_dirty_splits(self, cells):
        df = pd.DataFrame(
            data=[cell.to_list() for cell in cells], columns=GEOCELL_COLUMNS
        )
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
        columns = ["id", "lat", "lng"]
        df = pd.DataFrame(data=data, columns=columns)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs=CRS)
        return df

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
