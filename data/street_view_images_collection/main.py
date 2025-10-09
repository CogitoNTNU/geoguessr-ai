import os
from google_api.street_view import sign_url
import requests
from urllib.parse import urlencode
import random
from dotenv import load_dotenv
import numpy as np

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_KEY")  # required
SIGNING_SECRET = os.getenv("GOOGLE_MAPS_SECRET")
IMAGE_BASE = "https://maps.googleapis.com/maps/api/streetview"


def collect_google_streetview(lat: float, lon: float):
    seed = round(random.uniform(0, 90))
    headings = [seed, seed + 90, seed + 180, seed + 270]

    for heading in headings:
        params = {
            "location": f"{lat},{lon}",
            "size": "640x640",
            "fov": 90,
            "pitch": 0,
            "key": GOOGLE_KEY,
            "heading": heading,
            "return_error_code": "true",
        }
        url = f"{IMAGE_BASE}?{urlencode(params)}"
        url = sign_url(url, SIGNING_SECRET) if SIGNING_SECRET else url

        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(
                f"Image fetch failed ({r.status_code}) for heading {heading}, lat {lat}, lon {lon}"
            )

        # Save image to file tmp
        os.makedirs("out", exist_ok=True)
        filename = f"out/streetview_{lat}_{lon}_heading_{heading}.jpg"
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"Saved image to {filename}")


def getAllCoordinates() -> np.ndarray[(float, float)]:
    path = "data/out/sv_points_latlong.txt"
    points = np.loadtxt(path, delimiter=",")
    return points


def getCollectedCoordinates() -> np.ndarray[(float, float)]:
    path = "data/out/sv_points_latlong_collected.txt"
    points = np.loadtxt(path, delimiter=",")
    return points


def update_collected_points(points_to_collect: np.ndarray[(float, float)]):
    path = "data/out/sv_points_latlong_collected.txt"
    with open(path, "a") as f:
        for point in points_to_collect:
            f.write(f"{point[0]},{point[1]}\n")


def cleanup_temp_files():
    folder = "out"
    for filename in os.listdir(folder):
        if filename.startswith("streetview_") and filename.endswith(".jpg"):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    print("Temporary files cleaned up.")


def send_points_to_s3bucket():
    pass  # TODO: Magnus will implement this


def get_points(points_to_collect: np.ndarray[(float, float)]):
    collected_points = np.ndarray([])
    for i in range(len(points_to_collect)):
        lat = points_to_collect[i][0]
        lon = points_to_collect[i][1]

        if i % 100 == 0:
            print(
                f"Collecting point {i + 1}/{len(points_to_collect)}: lat {lat}, lon {lon}"
            )
            send_points_to_s3bucket()
            cleanup_temp_files()
            update_collected_points(collected_points)
            collected_points = np.ndarray([])

        try:
            collect_google_streetview(lat, lon)
            np.append(collected_points, np.array([[lat, lon]]), axis=0)
        except Exception as e:
            print(f"Error collecting point at lat: {lat}, lon: {lon}: error:{e}")
            continue

    send_points_to_s3bucket()
    cleanup_temp_files()
    update_collected_points(collected_points)
    print(f"Data collection complete. Collected {len(points_to_collect)} new points.")


if __name__ == "__main__":
    print("Starting data collection...")
    amount_of_pictures = 9900
    extra_credits_result = input(
        "Do you have enabled the extra credits in google cloud? (y/n): "
    )
    if extra_credits_result.lower() == "y":
        amount_of_pictures = 51_000

    total_points = getAllCoordinates()
    collected_points = getCollectedCoordinates()
    combined = np.vstack((total_points, collected_points))
    unique = np.unique(combined, axis=0)

    points_to_collect = unique[amount_of_pictures:]
    get_points(points_to_collect)
    print("Program complete!")
