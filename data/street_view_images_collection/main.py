import os
from google_api.street_view import sign_url
import requests
from urllib.parse import urlencode
import random
import math


GOOGLE_KEY = os.getenv("GOOGLE_MAPS_KEY")  # required
SIGNING_SECRET = os.getenv("GOOGLE_MAPS_SECRET")
IMAGE_BASE = "https://maps.googleapis.com/maps/api/streetview"


def collect_google_streetview(lat: float, lon: float):
    # Generate four random headings that cover 360 degrees
    headings = [math.ceil(random.uniform(i * 90, (i + 1) * 90)) for i in range(4)]

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
        with open("./out/" + heading, "wb") as f:
            f.write(r.content)
        print(r)


if __name__ == "__main__":
    print("Starting data collection...")
    amount_of_pictures = 1
    # amount_of_pictures = 9900
    extra_credits_result = input(
        "Do you have enabled the extra credits in google cloud? (y/n): "
    )
    if extra_credits_result.lower() == "y":
        amount_of_pictures = 51_000

    collect_google_streetview(60.15805, 19.905592)
