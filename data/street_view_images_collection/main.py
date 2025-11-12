import os
import requests
from urllib.parse import urlencode
import random
from dotenv import load_dotenv
import numpy as np
from backend.s3bucket import upload_dataset_from_folder, load_latest_snapshot_df
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_KEY")  # required
SIGNING_SECRET = os.getenv("GOOGLE_MAPS_SECRET")
IMAGE_BASE = "https://maps.googleapis.com/maps/api/streetview"


def fetch_streetview_image(lat: float, lon: float, heading: int):
    """Fetch one Street View image for a specific heading."""
    # First, fetch metadata to extract date and pano_id for filename
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    metadata_params = {
        "location": f"{lat},{lon}",
        "key": GOOGLE_KEY,
    }
    date_str = None
    pano_id = None
    try:
        meta_resp = requests.get(metadata_url, params=metadata_params, timeout=10)
        if meta_resp.status_code == 200:
            meta = meta_resp.json()
            if meta.get("status") == "OK":
                date_str = meta.get("date")
                pano_id = meta.get("pano_id")
    except Exception as e:
        # If metadata fetch fails, continue without it
        print(f"failed at the metadata: {e}")
        pass

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
    if SIGNING_SECRET:
        url += f"&signature={SIGNING_SECRET}"

    r = requests.get(url, timeout=30)
    if r.status_code == 403:
        raise ConnectionError(f"Rate limit exceeded at {lat},{lon}")
    if r.status_code != 200:
        # print(f"{r.text} with the status code {r.status_code}")
        raise RuntimeError(
            f"Image fetch failed ({r.status_code}) for heading {heading}, lat {lat}, lon {lon}"
        )

    os.makedirs("out", exist_ok=True)
    # Build filename: out/lat-long-heading-date-panoId.jpg
    # Fallback to 'unknown' if metadata was unavailable

    filename = (
        f"out/{lat}_{lon}_{heading}_"
        f"{(date_str or 'unknown')}_"
        f"{(pano_id or 'unknown')}.jpg"
    )

    with open(filename, "wb") as f:
        f.write(r.content)
    return filename


def collect_google_streetview(lat: float, lon: float) -> bool:
    seed = round(random.uniform(0, 90))
    headings = [seed, seed + 90, seed + 180, seed + 270]
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_heading = {
            executor.submit(fetch_streetview_image, lat, lon, h): h for h in headings
        }
        for future in as_completed(future_to_heading):
            try:
                filename = future.result()
                results.append(filename)
            except ConnectionError as ce:
                print(f"❌ {ce}")
                raise ConnectionError(ce)
            except Exception as e:
                print(f"Exception: {e}")
                return False

    if len(results) != 4:
        print("Den får ikke fire punkter")
        return False
    return True


def getAllCoordinates() -> np.ndarray[(float, float)]:
    cities = np.loadtxt(
        "data/out/sv_points_cities_latlong.txt", delimiter=","
    )  # (lat, lon)
    dist = np.loadtxt(
        "data/out/geoguessr_distributed_points_40k.txt", delimiter=" "
    )  # (lat, lon)

    pts = np.vstack([cities, dist])

    r = np.round(pts, 6)
    _, idx = np.unique(r, axis=0, return_index=True)
    return pts[np.sort(idx)]


def getCollectedCoordinates() -> np.ndarray[(float, float)]:
    path = "data/out/sv_points_latlong_collected.txt"
    points = np.loadtxt(path, delimiter=",")
    return points


def getFailedCoordinates() -> np.ndarray[(float, float)]:
    path = "data/out/failed_points.txt"
    points = np.loadtxt(path, delimiter=",")
    return points


def update_collected_points(points_to_collect: np.ndarray[(float, float)]):
    path = "data/out/sv_points_latlong_collected.txt"
    with open(path, "a") as f:
        for point in points_to_collect:
            f.write(f"{point[0]},{point[1]}\n")


def update_failed_points(points: np.ndarray[(float, float)]):
    path = "data/out/failed_points.txt"
    with open(path, "a") as f:
        for point in points:
            f.write(f"{point[0]},{point[1]}\n")


def cleanup_temp_files():
    folder = "out"
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    print("🔥🗑️Temporary files cleaned up.")


def setdiff2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise set difference: rows in a that are not in b."""
    if b.size == 0:
        return a.copy()
    a_view = np.ascontiguousarray(a).view([("", a.dtype)] * a.shape[1])
    b_view = np.ascontiguousarray(b).view([("", b.dtype)] * b.shape[1])
    diff = np.setdiff1d(a_view, b_view, assume_unique=False)
    return diff.view(a.dtype).reshape(-1, a.shape[1])


def get_points(points_to_collect: np.ndarray[(float, float)], max_workers: int = 8):
    collected_points = []
    success_count = 0
    total_points = len(points_to_collect)
    failed_points = []
    failed_count = 0

    def process_point(lat, lon):
        """Thread-safe wrapper for collect_google_streetview."""
        try:
            res = collect_google_streetview(lat, lon)
            if res:
                return (lat, lon, True)

            print(f"❌ Error collecting 4 points (lat: {lat}, lon: {lon})")
            return (lat, lon, False)
        except ConnectionError:
            print(
                "❌ Rate limit exceeded. Wait until the next day, or add your secret URL signing key to the .env file."
            )
            print("Exiting program!")
            os._exit(1)  # Exit the entire program on rate limit error
        except Exception as e:
            print(f"❌ Error collecting point (lat: {lat}, lon: {lon}): {e}")
            return (lat, lon, False)

    # Process in batches of 25 to preserve your upload/cleanup cadence
    for batch_start in tqdm(range(0, total_points, 25), desc="Collecting points"):
        batch_points = points_to_collect[batch_start : batch_start + 25]
        print(
            f"\nCollecting batch {(batch_start + 1) * 4}-{(batch_start + len(batch_points)) * 4} / {total_points * 4} images"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_point, lat, lon) for lat, lon in batch_points
            ]

            for future in as_completed(futures):
                lat, lon, success = future.result()
                if success:
                    collected_points.append([lat, lon])
                    success_count += 1
                else:
                    failed_points.append([lat, lon])
                    failed_count += 1

        # After every batch, upload and clean up
        if collected_points:
            upload_dataset_from_folder(folder="out")
            cleanup_temp_files()
            update_collected_points(np.array(collected_points))
            collected_points = []

        if failed_points:
            update_failed_points(np.array(failed_points))
            failed_points = []

    print("\n✅ Data collection complete.")
    print(f"Total points: {total_points}")
    print(f"Successfully collected: {success_count}")
    print(f"Failed to collect: {failed_count}")


if __name__ == "__main__":
    print("Starting data collection...")
    pictures_per_point = 4
    print(
        f"Points already collected: {len(load_latest_snapshot_df() // pictures_per_point)}"
    )
    amount_of_pictures = int(9900 / pictures_per_point)
    extra_credits_result = input(
        "Do you have enabled the extra credits in google cloud? (y/n): "
    )
    if extra_credits_result.lower() == "y":
        amount_of_pictures = int(25_000 / pictures_per_point)

    start_time = time.time()

    total_points = getAllCoordinates()
    print(f"Total points available: {len(total_points)}")

    collected_points = getCollectedCoordinates()
    print(f"Total points already collected: {len(collected_points)}")

    failed_points = getFailedCoordinates()
    print(f"Total points previously failed: {len(failed_points)}")

    unique_points = np.unique(total_points, axis=0)
    points_to_collect = setdiff2d(unique_points, collected_points)
    print(f"Total unique points to collect: {len(points_to_collect)}")

    points_to_collect = setdiff2d(points_to_collect, failed_points)
    print(
        f"Total points to collect after removing failed points: {len(points_to_collect)}"
    )

    points_to_collect = points_to_collect[:amount_of_pictures]
    print(
        f"Collecting {len(points_to_collect)} points in this run. Will collect {len(points_to_collect) * pictures_per_point} images.\n"
    )

    get_points(points_to_collect)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("Program complete!")
