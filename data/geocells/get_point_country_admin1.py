import pickle
import os


def load_geocells(dir):
    cells = {}
    for file in list(os.walk(dir))[0][2]:
        carved_country_name = file.split("_")[-1].split(".")[0]

        with open(dir + "/" + file, "rb") as f:
            data = pickle.load(f)
            cells[carved_country_name] = data
    return cells


def generate_dict(geocells):
    d = {}

    for country in geocells:
        for admin1 in geocells[country]:
            for cell in geocells[country][admin1]:
                for point in cell.points:
                    h = hash((point.latitude, point.longitude))
                    d[h] = {"country": country, "admin1": admin1}
    return d


if __name__ == "__main__":
    filepath = "data/geocells/finished_geocells"
    cells = load_geocells(filepath)

    print(generate_dict(cells))
