# run_sampling.py
import json
import os
import sys

from dotenv import load_dotenv

# Add the parent directory to Python path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sampling_pipeline import sample_sv_points_from_gadm

# Load environment variables from .env.local file
load_dotenv(".env.local")

# Les landliste
with open("data/sv_countries.txt") as f:
    countries = [line.strip() for line in f if line.strip()]

API_KEY = os.environ["GOOGLE_MAPS_API_KEY"]  # sett miljøvariabel
candidate_points, sv_points = sample_sv_points_from_gadm(
    gadm_dir="data/GADM_data",  # mappa der .gpkg/.json ligger
    sv_country_names=countries,  # land med SV
    pts_per_country=50,  # f.eks. 50 pr land (juster)
    api_key=API_KEY,
    radius_m=60,
    return_candidates=True,  # få både kandidater og verifiserte punkter
)

print(f"Samlet {len(candidate_points)} kandidatpunkter.")
print(f"Fikk {len(sv_points)} punkter med Street View.")
print(f"Suksessrate: {len(sv_points) / len(candidate_points) * 100:.1f}%")

# lagre begge filer for visualisering
os.makedirs("data/out", exist_ok=True)
with open("data/out/candidate_points.json", "w") as f:
    json.dump([{"lat": lat, "lon": lon} for lat, lon in candidate_points], f)
with open("data/out/sv_points.json", "w") as f:
    json.dump([{"lat": lat, "lon": lon} for lat, lon in sv_points], f)

print("✅ Lagret candidate_points.json og sv_points.json i data/out/")
