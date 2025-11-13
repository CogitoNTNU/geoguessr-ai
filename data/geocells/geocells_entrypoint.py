import cell
import geocell_visualizer
import admin_visualizer
import generate_geocells
import cell_visualizer
import load_admin_data
import test_geocells
import cluster

from tqdm import trange
import sqlite3
import pandas as pd
import random

import argparse


def main(args):
    print(f"Mode {args.mode} selected")
    if args.mode == 0:
        num_points = int(input("How many points? "))
        points = geocell_visualizer.generate_points(num_points)
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
    elif args.mode == 7:
        geocells = geocell_visualizer.GenerateGeocells(["Norway"])

    elif args.mode == 8:
        points = [{"lng": random.random(), "lat": random.random()} for i in range(100)]

        a = cell.Cell("hallo", points, [], "Hallo", "Hallo")

        print(cluster.cluster(a))
    elif args.mode == 9:
        sql = sqlite3.connect(
            "data/GADM_data/gadm_world_all_levels.filtered_noadm345.gpkg"
        )
        countries = pd.read_sql_query("SELECT * FROM ADM_0", sql)["COUNTRY"]

        sql.close()

        countries = list(countries)
        countries = countries

        for i in trange(len(countries)):
            generate_geocells.GenerateGeocells([countries[i]])
    elif args.mode == 10:
        sql = sqlite3.connect(
            "data/GADM_data/gadm_world_all_levels.filtered_noadm345.gpkg"
        )
        countries = pd.read_sql_query("SELECT * FROM ADM_2 WHERE NAME_2 IS 'NA'", sql)[
            "NAME_1"
        ]

        sql.close()

        for cel in countries:
            print(cel)

    else:
        print("Not a valid mode!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument(
        "mode",
        help="""Mode:\n
                         0: geocell_visualizer \n
                         1: admin_visualizer \n
                         2: generate_geocells \n
                         3: cell_visualizer \n
                         4: load_admin_data \n
                         5: cell \n
                         6: test_geocells\n
                         7: Kjør geocells\n
                         8: Test clustering\n
                         9: Lagre geocells fra alle land\n
                         10: Test database""",
        type=int,
    )
    args = parser.parse_args()
    main(args)
