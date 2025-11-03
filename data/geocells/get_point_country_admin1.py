import pickle
import os


class GeocellMananger:
    def __init__(self, dir: str):
        self.geocells = self.load_geocells(dir)
        self.dict = self.generate_dict()

    def load_geocells(self, dir):
        cells = {}
        for file in list(os.walk(dir))[0][2]:
            if not file.endswith(".pickle"):
                continue
            carved_country_name = file.split("_")[-1].split(".")[0]
            print(dir + "/" + file)

            with open(dir + "/" + file, "rb") as f:
                data = pickle.load(f)
                cells[carved_country_name] = data
                print(data)
        return cells

    def generate_dict(self):
        d = {}
        for country in self.geocells:
            for admin1 in self.geocells[country]:
                for cell in self.geocells[country][admin1]:
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
        if h not in self.dict:
            return None
        return self.dict[h]["geocell"]


if __name__ == "__main__":
    filepath = "data/geocells/finished_geocells"
    mang = GeocellMananger(filepath)

    with open("data/out/sv_points_all_latlong.pkl", "rb") as file:
        data = pickle.load(file)
    points = data

    # print(mang.dict)

    # print((points.iloc[0]["longitude"]))
    # for i in range(1, 100000):
    #     c = mang.get_geocell_id(points.iloc[i])
    #     if not c is None:
    #         print(c)
