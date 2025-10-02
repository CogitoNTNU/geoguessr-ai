#!/usr/bin/env python3
import os
import sys
import hmac
import hashlib
import base64
import requests
import json
from urllib.parse import urlencode, urlparse, urlunparse
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env.local"))

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_KEY")  # required
SIGNING_SECRET = os.getenv(
    "GOOGLE_MAPS_SECRET"
)  # optional (URL signing secret, base64 URL-safe)

METADATA_BASE = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_BASE = "https://maps.googleapis.com/maps/api/streetview"


def sign_url(url: str, b64_secret: str) -> str:
    """Return url with &signature=… using Maps URL signing secret."""
    if not b64_secret:
        return url
    # secret is URL-safe base64; decode to raw bytes
    secret = base64.urlsafe_b64decode(b64_secret)
    parts = urlparse(url)
    to_sign = f"{parts.path}?{parts.query}".encode("utf-8")
    sig = hmac.new(secret, to_sign, hashlib.sha1).digest()
    b64sig = base64.urlsafe_b64encode(sig).decode("utf-8")
    q = parts.query + f"&signature={b64sig}"
    return urlunparse(parts._replace(query=q))


def get_pano_id(lat: float, lng: float, radius: int = 100, source="outdoor") -> dict:
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "source": source,
        "key": GOOGLE_KEY,
    }
    url = f"{METADATA_BASE}?{urlencode(params)}"
    url = sign_url(url, SIGNING_SECRET) if SIGNING_SECRET else url
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    meta = r.json()
    if meta.get("status") != "OK":
        raise RuntimeError(f"No pano: status={meta.get('status')} at {lat},{lng}")
    return meta  # contains pano_id, location, date, etc.


def fetch_image(
    pano_id: str, heading: int, out_path: str, fov=90, pitch=0, size="640x640"
):
    params = {
        "pano": pano_id,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "size": size,
        "return_error_code": "true",
        "key": GOOGLE_KEY,
    }
    url = f"{IMAGE_BASE}?{urlencode(params)}"
    url = sign_url(url, SIGNING_SECRET) if SIGNING_SECRET else url
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"Image fetch failed ({r.status_code}) for heading {heading}"
        )
    with open(out_path, "wb") as f:
        f.write(r.content)


def fetch_block(lat: float, lng: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    meta = get_pano_id(lat, lng)
    pano = meta["pano_id"]

    # Save metadata to file
    metadata_file = os.path.join(out_dir, f"{pano}_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for h in (0, 90, 180, 270):
        out = os.path.join(out_dir, f"{pano}_{h}.jpg")
        fetch_image(pano, h, out)
    print(f"Done. Pano {pano} at {meta.get('location')} saved to {out_dir}")


if __name__ == "__main__":
    if not GOOGLE_KEY:
        sys.exit("Set GOOGLE_MAPS_KEY env var.")
    if len(sys.argv) < 4:
        # Default to the requested location if no CLI args provided
        lat, lng = 48.8584, 2.2945
        out_dir = os.path.join(
            os.path.dirname(__file__),
            ".",
            "sample_data",
            "sample_sv_59.916702_10.728529",
        )
        print(f"No CLI args provided; defaulting to {lat},{lng}. Output: {out_dir}")
    else:
        lat, lng, out_dir = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    fetch_block(lat, lng, out_dir)
