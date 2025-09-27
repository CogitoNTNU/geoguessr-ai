# sampling_pipeline.py
from __future__ import annotations
import os, json, time
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
    # Prøv GPKG først
    gpkg_path = None
    for fn in os.listdir(gadm_dir):
        if fn.endswith(".gpkg"):
            gpkg_path = os.path.join(gadm_dir, fn)
            break
    if gpkg_path:
        # Vanlige layer-navn i GADM 4.x: 'ADM_0'
        return gpd.read_file(gpkg_path, layer="ADM_0")
    
    # Prøv shapefile
    for fn in os.listdir(gadm_dir):
        if fn.endswith(".shp"):
            return gpd.read_file(os.path.join(gadm_dir, fn))
    
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

def filter_countries_by_sv(gdf: gpd.GeoDataFrame, sv_country_names: list[str],
                           name_col_candidates=("NAME_0","COUNTRY","NAME","ADMIN","SOVEREIGNT")):
    name_col = None
    for c in name_col_candidates:
        if c in gdf.columns:
            name_col = c; break
    if not name_col:
        raise ValueError(f"Fant ingen landnavn-kolonne blant {name_col_candidates}")
    # Normaliser navn (enkelt)
    norm = lambda s: str(s).strip().lower()
    sv_set = {norm(x) for x in sv_country_names}
    mask = gdf[name_col].apply(norm).isin(sv_set)
    return gdf.loc[mask].copy(), name_col

# ---------- Steg 3: Street View metadata-sjekk ----------
def sv_metadata_ok(lat, lon, api_key, radius=1000, session=None, retries=3, backoff=0.6):
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
            if data.get("status") in ("ZERO_RESULTS","NOT_FOUND"):
                return False, None, None
        time.sleep(backoff * (2**t))
    return False, None, None

def check_points_streetview(points_latlon, api_key, radius=1000, max_workers=32):
    """Returner kun punkter som har SV, med evt. 'snapped' lat/lon fra metadata."""
    kept = []
    with requests.Session() as s:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(sv_metadata_ok, lat, lon, api_key, radius, s) for lat,lon in points_latlon]
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
def sample_sv_points_from_gadm(gadm_dir: str,
                               sv_country_names: list[str],
                               pts_per_country: int,
                               api_key: str,
                               radius_m: int = 60):
    """
    - Leser GADM (admin0)
    - Fjerner vann implisitt ved å sample *inne i landegrense-polygons*
    - Begrenser til land med Street View (sv_country_names)
    - Sjekker hvert punkt mot Street View metadata
    Returnerer liste av (lat, lon) som har Street View.
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
        # Sample jevnt per land (kan skaleres etter areal om ønskelig)
        need = pts_per_country
        # Del “need” relativt jevnt over delpolygonene
        share = max(1, need // max(1, len(polys)))
        got = 0
        for poly in polys:
            if got >= need: break
            take = min(share, need - got)
            pts = _sample_points_in_polygon(poly, take)
            all_candidates.extend(pts)
            got += len(pts)

    # Steg 3: sjekk Street View metadata for alle kandidater
    sv_points = check_points_streetview(all_candidates, api_key, radius=radius_m)
    return sv_points
