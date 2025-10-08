import os
from google_api.street_view import sign_url
import requests
from urllib.parse import urlencode
import random
from dotenv import load_dotenv

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

        # Save image to s3 bucket
        # TODO


if __name__ == "__main__":
    print("Starting data collection...")
    amount_of_pictures = 1
    # amount_of_pictures = 9900
    extra_credits_result = input(
        "Do you have enabled the extra credits in google cloud? (y/n): "
    )
    if extra_credits_result.lower() == "y":
        amount_of_pictures = 51_000

    # TODO: Get the list of coordinates that is to be collected, and check if they have been collected already

    # TODO: Make a new file that has the coordinates that has been collected

    collect_google_streetview(60.15805, 19.905592)
