import cell
import geocell_visualizer
import admin_visualizer
import generate_geocells
import cell_visualizer
import load_admin_data
import test_geocells

import argparse

parser = argparse.ArgumentParser("simple_example")
parser.add_argument(
    "mode",
    help="""Mode:
                     0: geocell_visualizer
                     1: admin_visualizer
                     2: generate_geocells
                     3: cell_visualizer
                     4: load_admin_data
                     5: cell
                     6: test_geocells""",
    type=int,
)
args = parser.parse_args()


def main():
    print(f"Mode {args.mode} selected")
    if args.mode == 0:
        points = geocell_visualizer.generate_points(100000)
        [print(x.lat, x.lng) for x in points]
        # partition_output = partition(10, points)

        geocells = geocell_visualizer.GenerateGeocells(points)

        visualizer = geocell_visualizer.CellVisualizer(geocells)
        visualizer.show()
    elif args.mode == 1:
        admin_cell = admin_visualizer.AdminCell("data/GADM_data/GADM_admin_2")

        visualizer = admin_visualizer.CellVisualizer(admin_cell)
        visualizer.show()
    elif args.mode == 2:
        gen = generate_geocells.GenerateGeocells(set())
        print(gen)
    elif args.mode == 3:
        points = cell_visualizer.generate_points(1000)
        partition_output = cell_visualizer.partition(10, points)

        visualizer = cell_visualizer.CellVisualizer(partition_output)
        visualizer.show()
    elif args.mode == 4:
        admin_visualize = load_admin_data.AdminCell("data/GADM_data/")

        print(admin_visualize)
    elif args.mode == 5:
        cel = cell.Cell("Hallo", [], [], "Norway", "Rogaland")
        print(hash(cel))
        print(cel.is_empty())
    elif args.mode == 6:
        points = test_geocells.generate_points(100)
        cells = test_geocells.partition(10, points)

        print(cells[0][1])
    else:
        print("Not a valid mode!")


if __name__ == "__main__":
    main()
