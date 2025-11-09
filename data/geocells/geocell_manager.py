import pickle
import os
import pandas as pd


class _CellShim:
    def __init__(self):  # no args; we'll fill via __setstate__
        pass

    def __setstate__(self, state):
        # allow dict-based state restoration
        self.__dict__.update(state)

    def __len__(self):
        pts = getattr(self, "points", None)
        try:
            return len(pts) if pts is not None else 0
        except Exception:
            return 0


class _RedirectingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "cell":
            # Whatever class name was used (Cell, GeoCell, etc.), return the shim
            return _CellShim
        return super().find_class(module, name)


class GeocellManager:
    def __init__(self, dir: str):
        self.geocells = self.load_geocells(dir)
        self.point_info_dict = self.generate_dict()

    def get_num_geocells(self):
        # len(self.geocells) counts countries; compute total cells instead
        return sum(
            len(self.geocells[country][admin1])
            for country in self.geocells
            for admin1 in self.geocells[country]
        )

    def _safe_pickle_load(self, f):
        # Try normal unpickle first
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if str(e).startswith("No module named 'cell'"):
                f.seek(0)
                return _RedirectingUnpickler(f).load()
            raise

    def load_geocells(self, dir):
        cells = {}
        root, _, files = next(os.walk(dir))
        for file in files:
            if not file.endswith(".pickle"):
                continue
            carved_country_name = file.split("_")[-1].split(".")[0]
            with open(os.path.join(root, file), "rb") as f:
                data = self._safe_pickle_load(f)  # <— use redirecting loader
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

    def _get_cell_cluster_id(self, point):
        h = hash((point["latitude"], point["longitude"]))
        if h not in self.point_info_dict:
            print("Point not found in any geocell.")
            return None
        print(self.point_info_dict[h])
        return self.point_info_dict[h]["cluster_id"]

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

    def generate_proto_df(self):
        # output a pandas dataframe to csv, data: geocell_idx, cluster_id, indices, count, centroid_lat, centroid_lng
        geocell_id = 0
        rows = []
        for country in self.geocells:
            for admin1 in self.geocells[country]:
                for cell in self.geocells[country][admin1]:
                    for cluster_id, cluster_data in cell.clusters.items():
                        rows.append(
                            {
                                "geocell_id": geocell_id,
                                "country": country,
                                "cluster_id": cluster_id,
                                "count": len(cluster_data["points"]),
                                "indices": [x.name for x in cluster_data["points"]],
                                "centroid_lat": cell.geom_centroid[1],  # Latitude
                                "centroid_lng": cell.geom_centroid[0],  # Longitude
                            }
                        )
                    geocell_id += 1
        df = pd.DataFrame(rows)
        df.to_csv("data/geocells/proto_df.csv", index=False)


# if __name__ == "__main__":
#     filepath = "data/geocells/finished_geocells"
#     print("Loading geocell manager...")
#     mang = GeocellManager(filepath)
#     mang.generate_proto_df()
#     print("generate_proto_df complete.")
