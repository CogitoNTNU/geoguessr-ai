# sampling_pipeline.py
from __future__ import annotations
import os
import json
import time
import concurrent.futures as cf
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import requests


# ---------- Steg 1: land/vann + sampling inne i polygon ----------
def _sample_points_in_polygon(poly, n, rng=None):
    """Uniform sampling i polygon ved rejection i bbox."""
    rng = rng or np.random.default_rng(42)
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    # enkel rejection; for store/rare polygoner kan vi øke max_trials
    max_trials = 10000 * max(1, n)
    trials = 0
    while len(pts) < n and trials < max_trials:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            pts.append((y, x))  # (lat, lon)
        trials += 1
    return pts


# ---------- Steg 2: last GADM, filter på land med Street View ----------
def load_gadm_admin0(gadm_dir: str) -> gpd.GeoDataFrame:
    """
    Forventer GADM Admin level 0 (land) GeoPackage, shapefile eller JSON.
    Last ned her: https://gadm.org/download_country.html (eller world gpkg)
    Eksempel: gadm_410.gpkg -> layer 'ADM_0'
    """

    # Prøv JSON/GeoJSON filer og kombiner dem
    json_files = []
    for fn in os.listdir(gadm_dir):
        if fn.endswith(".json") or fn.endswith(".geojson"):
            json_files.append(os.path.join(gadm_dir, fn))

    if json_files:
        # Les alle JSON filer og kombiner dem
        gdfs = []
        for json_file in json_files:
            try:
                gdf = gpd.read_file(json_file)
                gdfs.append(gdf)
            except Exception as e:
                print(f"Kunne ikke lese {json_file}: {e}")
                continue

        if gdfs:
            # Kombiner alle GeoDataFrames
            combined_gdf = gpd.pd.concat(gdfs, ignore_index=True)
            return combined_gdf

    raise FileNotFoundError("Fant ikke GADM .gpkg, .shp eller .json filer i mappen")


def filter_countries_by_sv(
    gdf: gpd.GeoDataFrame,
    sv_country_names: list[str],
    name_col_candidates=("NAME_0", "COUNTRY", "NAME", "ADMIN", "SOVEREIGNT"),
):
    name_col = None
    for c in name_col_candidates:
        if c in gdf.columns:
            name_col = c
            break
    if not name_col:
        raise ValueError(f"Fant ingen landnavn-kolonne blant {name_col_candidates}")

    # Normaliser navn (enkelt)
    def norm(s):
        return str(s).strip().lower()

    sv_set = {norm(x) for x in sv_country_names}
    mask = gdf[name_col].apply(norm).isin(sv_set)
    return gdf.loc[mask].copy(), name_col


# ---------- Steg 3: Street View metadata-sjekk ----------
def sv_metadata_ok(
    lat, lon, api_key, radius=1000, session=None, retries=3, backoff=0.6
):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "radius": int(radius), "key": api_key}
    s = session or requests.Session()
    for t in range(retries):
        r = s.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "OK":
                loc = data.get("location", {})
                return True, loc.get("lat", lat), loc.get("lng", lon)
            if data.get("status") in ("ZERO_RESULTS", "NOT_FOUND"):
                return False, None, None
        time.sleep(backoff * (2**t))
    return False, None, None


def check_points_streetview(points_latlon, api_key, radius=1000, max_workers=32):
    """Returner kun punkter som har SV, med evt. 'snapped' lat/lon fra metadata."""
    kept = []
    with requests.Session() as s:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(sv_metadata_ok, lat, lon, api_key, radius, s)
                for lat, lon in points_latlon
            ]
            for f in cf.as_completed(futs):
                ok, sla, slo = f.result()
                if ok:
                    kept.append((sla, slo))
    # dedupe grovt
    if kept:
        arr = np.unique(np.round(np.array(kept, dtype=float), 6), axis=0)
        kept = [tuple(x) for x in arr.tolist()]
    return kept


# ---------- Hoved-funksjon: Steg 1–3 samlet ----------
def sample_sv_points_from_gadm(
    gadm_dir: str,
    sv_country_names: list[str],
    pts_per_country: int,
    api_key: str,
    radius_m: int = 60,
    return_candidates: bool = False,
    point_density_scalar: float = 1.0,
):
    """
    - Leser GADM (admin0)
    - Fjerner vann implisitt ved å sample *inne i landegrense-polygons*
    - Begrenser til land med Street View (sv_country_names)
    - Sjekker hvert punkt mot Street View metadata
    Returnerer liste av (lat, lon) som har Street View.
    Hvis return_candidates=True, returnerer (kandidater, sv_points) istedenfor bare sv_points.
    """
    world = load_gadm_admin0(gadm_dir)
    gdf, name_col = filter_countries_by_sv(world, sv_country_names)

    all_candidates = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Håndter MultiPolygon
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        # Sample weighted by polygon area for uniform distribution
        need = pts_per_country

        # Calculate total area and distribute points proportionally
        total_area = sum(poly.area for poly in polys)
        if total_area == 0:
            continue

        got = 0
        for poly in polys:
            if got >= need:
                break
            # Number of points proportional to polygon area
            poly_fraction = poly.area * point_density_scalar / total_area
            take = int(round(poly_fraction * need))

            # Ensure we don't exceed the target
            take = min(take, need - got)

            if take > 0:
                pts = _sample_points_in_polygon(poly, take)
                all_candidates.extend(pts)
                got += len(pts)

    # Steg 3: sjekk Street View metadata for alle kandidater
    sv_points = check_points_streetview(all_candidates, api_key, radius=radius_m)

    if return_candidates:
        return all_candidates, sv_points
    return sv_points


# ---------- Test/Demo Main ----------
if __name__ == "__main__":
    """
    Quick test of the sampling pipeline - samples points across all of Norway.
    Usage: GOOGLE_MAPS_API_KEY="your_key" python src/sampling_pipeline.py
    Requires: GOOGLE_MAPS_API_KEY environment variable set
    """
    import sys

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_MAPS_API_KEY environment variable not set")
        print("   Usage: GOOGLE_MAPS_API_KEY='your_key' python src/sampling_pipeline.py")
        sys.exit(1)

    # Load Norway admin-2 data
    country_file = "data/GADM_data/GADM_admin_2/gadm41_NOR_2.json"
    if not os.path.exists(country_file):
        print(f"❌ Error: {country_file} not found")
        print("   Download GADM admin-2 data for Norway")
        sys.exit(1)

    print("🌍 Loading Norway country boundary...")
    gdf = gpd.read_file(country_file)

    if gdf.empty:
        print(f"❌ Error: No data found in {country_file}")
        sys.exit(1)

    norway_row = gdf.iloc[0]
    country_name = norway_row.get("COUNTRY", norway_row.get("NAME_0", "Norway"))
    print(f"✅ Loaded: {country_name}")

    # Sample candidate points across all of Norway
    num_candidates = 10000
    point_density_scalar = (
        10.0  # Adjust this to bias sampling (e.g., 2.0 for mainland, 0.5 for islands)
    )
    print(f"\n📍 Sampling {num_candidates} candidate points across Norway...")
    print(f"   Point density scalar: {point_density_scalar}")

    all_candidates = []
    geom = norway_row.geometry

    if geom is None or geom.is_empty:
        print("❌ Error: Norway geometry is empty")
        sys.exit(1)

    # Handle MultiPolygon
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    print(f"   Found {len(polys)} polygon(s) in Norway")

    # Sample points weighted by polygon area for uniform distribution
    total_area = sum(poly.area for poly in polys)
    print(f"   Total area: {total_area:.6f} square degrees")

    got = 0
    for i, poly in enumerate(polys):
        if got >= num_candidates:
            break

        # Number of points proportional to polygon area (with density scalar)
        poly_fraction = poly.area * point_density_scalar / total_area
        take = int(round(poly_fraction * num_candidates))

        # Ensure we don't exceed the target
        take = min(take, num_candidates - got)

        if take > 0:
            pts = _sample_points_in_polygon(poly, take)
            all_candidates.extend(pts)
            got += len(pts)
            if take >= 10:  # Only log significant polygons
                print(f"   Polygon {i}: area={poly.area:.6f}, points={take}")

    print(f"   Generated {len(all_candidates)} candidate points")

    # Check Street View availability
    print("\n🔍 Checking Street View availability...")
    sv_points = check_points_streetview(all_candidates, api_key, radius=1000)

    # Print results
    print("\n✅ Results for Norway:")
    print(f"   • Candidate points: {len(all_candidates)}")
    print(f"   • Street View points: {len(sv_points)}")
    if all_candidates:
        success_rate = len(sv_points) / len(all_candidates) * 100
        print(f"   • Success rate: {success_rate:.1f}%")

    # Save results
    os.makedirs("data/out", exist_ok=True)
    with open("data/out/candidate_points.json", "w") as f:
        json.dump(
            [{"lat": lat, "lon": lon} for lat, lon in all_candidates], f, indent=2
        )
    with open("data/out/sv_points.json", "w") as f:
        json.dump([{"lat": lat, "lon": lon} for lat, lon in sv_points], f, indent=2)

    print("\n💾 Saved results to data/out/")
    print("   • candidate_points.json")
    print("   • sv_points.json")

    # Create comparison visualization
    print("\n🗺️  Creating comparison visualization...")
    try:
        # Import and run comparison visualization
        # When adding the 'src' folder to sys.path, import packages WITHOUT the 'src.' prefix
        sys.path.insert(0, "src")
        from point_visualization.compare_point import create_comparison_map

        create_comparison_map()
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
        print("   You can manually run one of:")
        print("     • python src/point_visualization/compare_point.py")
        print("     • python -m src.point_visualization.compare_point")
