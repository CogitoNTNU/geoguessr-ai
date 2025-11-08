import pickle
import os


class GeocellMananger:
    def __init__(self, dir: str):
        self.geocells = self.load_geocells(dir)
        self.point_info_dict = self.generate_dict()

    def get_num_geocells(self):
        return len(self.geocells)

    def load_geocells(self, dir):
        cells = {}
        for file in list(os.walk(dir))[0][2]:
            if not file.endswith(".pickle"):
                continue
            carved_country_name = file.split("_")[-1].split(".")[0]

            with open(dir + "/" + file, "rb") as f:
                data = pickle.load(f)
                cells[carved_country_name] = data
        return cells

    def generate_dict(self):
        d = {}
        for country in self.geocells:
            for admin1 in self.geocells[country]:
                for cell in self.geocells[country][admin1]:
                    # for cluster in cell.clusters:
                    #     print(len(cell.clusters[cluster]["points"]))
                    for point in cell.points:
                        h = hash((point["latitude"], point["longitude"]))
                        cell_cluster = -1
                        for cluster in cell.clusters:
                            if h in cell.clusters[cluster]["hashes"]:
                                cell_cluster = cluster
                        d[h] = {
                            "country": country,
                            "admin1": admin1,
                            "geocell": cell.id,
                            "cluster_id": cell_cluster,
                            "lat": point["latitude"],
                            "lng": point["longitude"],
                        }
        return d

    def get_geocell_id(self, point):
        h = hash((point["latitude"], point["longitude"]))
        if h not in self.point_info_dict:
            return None
        return (
            self.point_info_dict[h]["geocell"],
            self.point_info_dict[h]["country"],
            self.point_info_dict[h]["admin1"],
        )

    def get_geocell_info(self, geocell_id, country, admin1):
        for cell in self.geocells[country][admin1]:
            if cell.id == geocell_id:
                return cell
        return None


if __name__ == "__main__":
    filepath = "data/geocells/finished_geocells"
    mang = GeocellMananger(filepath)

    with open("data/out/sv_points_all_latlong.pkl", "rb") as file:
        data = pickle.load(file)
    points = data

    # print(mang.dict)
    total_points = 0
    total_cells = 0
    max_points = 0
    max_cell = None
    for country in mang.geocells:
        for admin1 in mang.geocells[country]:
            for cell in mang.geocells[country][admin1]:
                total_points += len(cell)
                total_cells += 1
                if len(cell) > max_points:
                    max_cell = cell
                    max_points = len(cell)
    print(f"{total_points=}\n {max_cell}: {max_points}")
    print(f"Total number of geocells: {total_cells}")

    print(mang.get_num_geocells())
    
    # print((points.iloc[0]["longitude"]))
    # for i in range(1, 100000):
    #     c = mang.get_geocell_id(points.iloc[i])
    #     if c is not None:
    #         print(mang.get_geocell_info(*c))
