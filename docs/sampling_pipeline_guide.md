# Sampling Pipeline Documentation

## Overview

The sampling pipeline is responsible for generating geographically sampled points that have Google Street View coverage. It uses GADM (Global Administrative Areas) data to sample points within country boundaries and validates Street View availability through the Google Maps API.

## Files

### `src/sampling_pipeline.py`

The core module containing the sampling and validation logic.

#### Key Functions

##### 1. `_sample_points_in_polygon(poly, n, rng=None)`

- **Purpose**: Generates uniform random points within a polygon using rejection sampling
- **Parameters**:
  - `poly`: Shapely polygon geometry
  - `n`: Number of points to generate
  - `rng`: Optional numpy random generator (defaults to seed 42)
- **Returns**: List of (lat, lon) tuples
- **Algorithm**: Uses bounding box rejection sampling - generates random points in bbox and keeps those inside polygon

##### 2. `load_gadm_admin0(gadm_dir: str)`

- **Purpose**: Loads GADM administrative level 0 (country) boundaries
- **Parameters**:
  - `gadm_dir`: Directory containing GADM data files
- **Returns**: GeoDataFrame with country geometries
- **Supported formats**:
  - GeoPackage (.gpkg) with 'ADM_0' layer
  - Shapefiles (.shp)
  - JSON/GeoJSON files (combines multiple files)
- **Note**: Tries formats in order: GPKG → Shapefile → JSON

##### 3. `filter_countries_by_sv(gdf, sv_country_names, name_col_candidates)`

- **Purpose**: Filters countries to only those with Street View coverage
- **Parameters**:
  - `gdf`: GeoDataFrame with country data
  - `sv_country_names`: List of country names with Street View
  - `name_col_candidates`: Tuple of possible column names for country names
- **Returns**: Tuple of (filtered GeoDataFrame, name column used)
- **Normalization**: Case-insensitive matching with whitespace trimming

##### 4. `sv_metadata_ok(lat, lon, api_key, radius=1000, session=None, retries=3, backoff=0.6)`

- **Purpose**: Checks if Street View is available at a location
- **Parameters**:
  - `lat`, `lon`: Coordinates to check
  - `api_key`: Google Maps API key
  - `radius`: Search radius in meters (default 1000m)
  - `session`: Optional requests Session for connection pooling
  - `retries`: Number of retry attempts (default 3)
  - `backoff`: Exponential backoff base time in seconds (default 0.6s)
- **Returns**: Tuple of (success bool, snapped_lat, snapped_lon)
- **API**: Uses Google Street View Metadata API
- **Status codes**:
  - "OK" → Street View available
  - "ZERO_RESULTS"/"NOT_FOUND" → No Street View

##### 5. `check_points_streetview(points_latlon, api_key, radius=1000, max_workers=32)`

- **Purpose**: Batch validates Street View availability for multiple points
- **Parameters**:
  - `points_latlon`: List of (lat, lon) tuples to check
  - `api_key`: Google Maps API key
  - `radius`: Search radius in meters
  - `max_workers`: Maximum parallel threads (default 32)
- **Returns**: List of (lat, lon) tuples with Street View coverage
- **Features**:
  - Parallel execution using ThreadPoolExecutor
  - Returns "snapped" coordinates from Street View metadata
  - Deduplicates results (rounds to 6 decimal places)

##### 6. `sample_sv_points_from_gadm(gadm_dir, sv_country_names, pts_per_country, api_key, radius_m=60, return_candidates=False)`

- **Purpose**: Main pipeline function - samples points and validates Street View
- **Parameters**:
  - `gadm_dir`: Directory with GADM data
  - `sv_country_names`: List of countries with Street View
  - `pts_per_country`: Number of candidate points per country
  - `api_key`: Google Maps API key
  - `radius_m`: Street View search radius (default 60m)
  - `return_candidates`: If True, returns both candidates and validated points
- **Returns**:
  - If `return_candidates=False`: List of validated Street View points
  - If `return_candidates=True`: Tuple of (candidate_points, sv_points)
- **Pipeline steps**:
  1. Load GADM admin0 boundaries
  1. Filter to Street View countries
  1. Sample points uniformly within country polygons
  1. Validate each point against Street View API
  1. Return validated points (with optional candidates)

### `tests/run_sampling.py`

Test/example script demonstrating the sampling pipeline.

#### Workflow

1. **Setup**: Loads environment variables from `.env.local`
1. **Country List**: Reads country names from `data/sv_countries.txt`
1. **Sampling**: Calls `sample_sv_points_from_gadm()` with:
   - GADM data from `data/GADM_data/`
   - 50 candidate points per country
   - 60m Street View search radius
   - Returns both candidates and validated points
1. **Output**: Saves results to JSON files in `data/out/`:
   - `candidate_points.json`: All sampled points before validation
   - `sv_points.json`: Points with confirmed Street View coverage
1. **Statistics**: Prints counts and success rate

## Pipeline Flow

```
┌─────────────────────────┐
│  Load GADM Country Data │
│  (load_gadm_admin0)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Filter SV Countries    │
│  (filter_countries_by_sv)│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Sample Points in       │
│  Polygons               │
│  (_sample_points_in_    │
│   polygon)              │
└───────────┬─────────────┘
            │
            ▼ Candidates
┌─────────────────────────┐
│  Check Street View      │
│  Metadata               │
│  (check_points_         │
│   streetview)           │
└───────────┬─────────────┘
            │
            ▼ Validated Points
┌─────────────────────────┐
│  Save Results           │
│  (run_sampling.py)      │
└─────────────────────────┘
```

## Usage Example

```python
from src.sampling_pipeline import sample_sv_points_from_gadm

# Get validated Street View points
sv_points = sample_sv_points_from_gadm(
    gadm_dir="data/GADM_data",
    sv_country_names=["Norway", "Sweden", "Denmark"],
    pts_per_country=100,
    api_key="YOUR_API_KEY",
    radius_m=60,
)

# Get both candidates and validated points
candidates, sv_points = sample_sv_points_from_gadm(
    gadm_dir="data/GADM_data",
    sv_country_names=["Norway", "Sweden", "Denmark"],
    pts_per_country=100,
    api_key="YOUR_API_KEY",
    radius_m=60,
    return_candidates=True,
)
```

## Requirements

- **GADM Data**: Download from [gadm.org](https://gadm.org/download_country.html)
- **Google Maps API Key**: Required for Street View metadata checks
- **Country List**: Text file with one country name per line
- **Dependencies**: geopandas, shapely, numpy, requests

## Configuration

### Environment Variables

- `GOOGLE_MAPS_API_KEY`: Your Google Maps API key (loaded from `.env.local`)

### Data Files

- `data/GADM_data/`: GADM administrative boundaries (.gpkg, .json, or .shp)
- `data/sv_countries.txt`: List of countries with Street View coverage

### Output Files

- `data/out/candidate_points.json`: All sampled candidate points
- `data/out/sv_points.json`: Validated Street View points

## Performance Notes

- **Parallel Processing**: Uses 32 threads by default for API calls
- **Success Rate**: Typically 20-40% of random land points have Street View
- **API Rate Limits**: Consider Google Maps API quotas for large datasets
- **Retry Logic**: Exponential backoff handles transient API errors
- **Deduplication**: Results rounded to 6 decimal places (~10cm precision)
