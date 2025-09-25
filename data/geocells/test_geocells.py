from naive_cell import RectCell, Point
import random
import heapq


def generate_points(n):
    points = set()
    for i in range(n):
        points.add(Point(id=i, lng=random.random() * 90, lat=random.random() * 90))
    return points


def partition(n, min_cell_size, points):
    start_cell = RectCell(points)
    cells = []
    heapq.heappush(cells, (n - len(start_cell), start_cell))
    lowest_points = cells[-1][0]

    while lowest_points > min_cell_size:
        cell_to_split = cells[0][1]

        cell1, cell2 = cell_to_split.split_cell()
        heapq.heappush(cells, (n - len(cell1), cell1))
        heapq.heappush(cells, (n - len(cell2), cell2))
        lowest_points = cells[-1][0]

    return cells


points = generate_points(100)
cells = partition(100, 10, points)

print(cells[0][1])
