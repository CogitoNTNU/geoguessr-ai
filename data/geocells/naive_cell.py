from typing import Set
import numpy as np


class Point:
    def __init__(self, id, lng, lat):
        self.id: int = id  # TODO: Should be same for each (lat, long)-pair so pictures from same location can be saved in same folder in S3-buckert
        self.lng: float = lng
        self.lat: float = lat


class RectCell:
    def __init__(self, points: Set[Point]):
        self.points = points
        self.max_lat = max([point.lat for point in self.points])
        self.min_lat = min([point.lat for point in self.points])
        self.max_lng = max([point.lng for point in self.points])
        self.min_lng = min([point.lng for point in self.points])

        self.longtitude = np.array([p.lng for p in self.points])
        self.latitude = np.array([p.lat for p in self.points])

        self.area = (self.max_lat - self.min_lat) * (self.max_lng - self.min_lng)

    def __len__(self) -> int:
        return len(self.points)

    def should_split_lng(self) -> bool:
        return self.max_lng - self.min_lng > self.max_lat - self.min_lat

    def split_cell(self):
        p1, p2 = set(), set()
        if self.should_split_lng():
            for point in self.points:
                if point.lng > (self.max_lng + self.min_lng) / 2:
                    p2.add(point)
                else:
                    p1.add(point)
        else:
            for point in self.points:
                if point.lat > (self.max_lat + self.min_lat) / 2:
                    p2.add(point)
                else:
                    p1.add(point)

        return RectCell(p1), RectCell(p2)

    def centroid(self):
        return np.mean(self.latitude), np.mean(self.longtitude)

    def __str__(self):
        if len(self) == 0:
            return "The cell is empty."
        else:
            return f"""
            ---------------
            Cell length: {len(self)} points 
            Centroid: {self.centroid()}
            """
