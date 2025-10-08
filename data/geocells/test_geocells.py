from naive_cell import RectCell, Point
import random
import heapq


def generate_points(n):
    points = set()
    for i in range(n):
        points.add(
            Point(id=i, lng=random.uniform(-180, 180), lat=random.uniform(-90, 90))
        )
    return points


def partition(min_cell_size, points):
    start_cell = RectCell(points)
    cells = []
    counter = 0
    heapq.heappush(cells, (-len(start_cell), counter, start_cell))
    counter += 1

    while cells and -cells[0][0] > min_cell_size:
        _, _, cell_to_split = heapq.heappop(cells)
        cell1, cell2 = cell_to_split.split_cell()
        if len(cell1) > 0:
            heapq.heappush(cells, (-len(cell1), counter, cell1))
            counter += 1
        if len(cell2) > 0:
            heapq.heappush(cells, (-len(cell2), counter, cell2))
            counter += 1

    return cells


points = generate_points(100)
cells = partition(10, points)

print(cells[0][1])
