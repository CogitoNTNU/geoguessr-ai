import time
import math
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ geometry ------------------
def latlon_to_unitvec(lat_deg, lon_deg):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)

def great_circle_angle(u, v):
    dots = np.clip(u @ v.T, -1.0, 1.0)
    return np.arccos(dots)

def fibonacci_sphere(n_pts: int):
    """Nearly-uniform points on the sphere. Returns lat, lon in degrees."""
    i = np.arange(n_pts)
    phi = (1 + 5**0.5) / 2
    lon = 360.0 * (i / phi % 1.0)
    z = 1 - 2*(i + 0.5)/n_pts
    lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    lon = ((lon + 180) % 360) - 180
    return lat, lon

# ------------------ farthest-point sampling ------------------
def farthest_point_sampling(lat, lon, k, seed_index=None):
    N = len(lat); assert len(lon) == N and k <= N
    U = latlon_to_unitvec(lat, lon)
    if seed_index is None:
        centroid = U.mean(axis=0); centroid /= np.linalg.norm(centroid)
        d0 = np.arccos(np.clip(U @ centroid, -1.0, 1.0))
        current = int(np.argmax(d0))
    else:
        current = int(seed_index)
    selected = [current]
    min_d = great_circle_angle(U, U[[current]]).squeeze()
    for _ in range(1, k):
        cand = int(np.argmax(min_d))
        selected.append(cand)
        d_new = great_circle_angle(U, U[[cand]]).squeeze()
        min_d = np.minimum(min_d, d_new)
    return np.array(selected, dtype=int)

# ------------------ Street View coverage check ------------------
def _sv_metadata_ok(lat, lon, api_key, radius=50, session=None, retries=3, backoff=0.5):
    """
    Uses the Street View Static API *metadata* endpoint to check coverage near (lat, lon).
    Returns (ok: bool, snapped_lat, snapped_lon) if available.
    """
    if session is None:
        session = requests.Session()
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "radius": int(radius),   # meters, search around the point
        "key": api_key,
    }
    for t in range(retries):
        r = session.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "OK":
                loc = data.get("location", {})
                return True, loc.get("lat", lat), loc.get("lng", lon)
            elif data.get("status") in ("ZERO_RESULTS", "NOT_FOUND"):
                return False, None, None
            # else OVER_QUERY_LIMIT, UNKNOWN_ERROR, etc → backoff and retry
        time.sleep(backoff * (2 ** t))
    return False, None, None

def _filter_points_with_sv(lat, lon, api_key, radius=50, max_workers=32):
    """
    Parallel coverage check; returns arrays of snapped lat/lon for points with SV.
    """
    session = requests.Session()
    keep_lat, keep_lon = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_sv_metadata_ok, float(la), float(lo), api_key, radius, session)
                for la, lo in zip(lat, lon)]
        for f in as_completed(futs):
            ok, sla, slo = f.result()
            if ok:
                keep_lat.append(sla)
                keep_lon.append(slo)
    return np.array(keep_lat, dtype=float), np.array(keep_lon, dtype=float)

# ------------------ public API ------------------
def sample_street_view_worldwide(k,
                                 api_key: str,
                                 radius_m: int = 50,
                                 oversample_factor: float = 60.0,
                                 max_rounds: int = 6,
                                 workers: int = 32):
    """
    Returns k (lat, lon) pairs that (a) have Street View coverage and (b) are
    maximally spread globally via FPS.

    Strategy:
      - Generate a uniform grid of size M ≈ oversample_factor * k
      - Keep only grid points that have Street View coverage (metadata=OK)
      - If we have >= k, FPS-prune to k; else, increase M and repeat.

    Notes:
      • Street View coverage is a tiny fraction of the sphere (oceans, deserts, etc.),
        hence a large oversample factor by default.
      • API costs/quotas apply for the metadata endpoint.
    """
    assert k > 0
    covered_lat, covered_lon = np.array([]), np.array([])

    M = int(math.ceil(k * oversample_factor))
    for _ in range(max_rounds):
        grid_lat, grid_lon = fibonacci_sphere(M)
        lat_ok, lon_ok = _filter_points_with_sv(
            grid_lat, grid_lon, api_key=api_key,
            radius=radius_m, max_workers=workers
        )

        if lat_ok.size > 0:
            covered_lat = np.concatenate([covered_lat, lat_ok])
            covered_lon = np.concatenate([covered_lon, lon_ok])

        # Deduplicate snapped points (some grid points snap to same pano)
        if covered_lat.size:
            uniq = np.unique(np.round(np.c_[covered_lat, covered_lon], 6), axis=0)
            covered_lat, covered_lon = uniq[:,0], uniq[:,1]

        if covered_lat.size >= k:
            break

        # Not enough coverage found; increase grid and try again
        M = int(M * 1.8)

    if covered_lat.size < k:
        # Return as many as we have (caller can decide what to do)
        k = covered_lat.size

    # FPS to enforce max spread among covered points
    idx = farthest_point_sampling(covered_lat, covered_lon, k)
    return covered_lat[idx], covered_lon[idx]