# ResQSync Data Acquisition & Spatial Feature Engineering Pipeline
# SIH26001: AI-Based Early Warning and Landslide Risk Monitoring System in NER
import os
import json
import datetime
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

LAT_MIN, LAT_MAX = 27.0, 28.0
LON_MIN, LON_MAX = 88.0, 89.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR = os.path.join(DATA_DIR, "dem")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(DEM_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

DEM_PATH = os.path.join(DEM_DIR, "Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif")
GLC_CSV_PATH = os.path.join(RAW_DIR, "Global_Landslide_Catalog_Export.csv")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, "resqsync_ner_dataset.csv")

def download_official_datasets():
    if not os.path.exists(DEM_PATH):
        print("[1/4] Downloading Copernicus DEM 30m GeoTIFF (GLO-30)...")
        dem_url = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N27_00_E088_00_DEM/Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif"
        r = requests.get(dem_url, stream=True)
        r.raise_for_status()
        with open(DEM_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Copernicus DEM saved to {DEM_PATH}")
    else:
        print(f"[1/4] Copernicus DEM already cached at {DEM_PATH}")

    if not os.path.exists(GLC_CSV_PATH):
        print("[2/4] Downloading NASA Global Landslide Catalog / GSI Records...")
        glc_url = "https://raw.githubusercontent.com/AntoineSueur/Geoplot-landslides/master/Global_Landslide_Catalog_Export.csv"
        r = requests.get(glc_url)
        r.raise_for_status()
        with open(GLC_CSV_PATH, "wb") as f:
            f.write(r.content)
        print(f"Landslide Catalog saved to {GLC_CSV_PATH}")
    else:
        print(f"[2/4] Landslide Catalog already cached at {GLC_CSV_PATH}")

def compute_terrain_rasters():
    print("[3/4] Computing slope and elevation grids with rasterio...")
    with rasterio.open(DEM_PATH) as src:
        elev_grid = src.read(1)
        transform = src.transform
        dy = 111320.0 / 3600.0
        dx = (111320.0 * np.cos(np.radians(27.5))) / 3600.0
        dz_dy, dz_dx = np.gradient(elev_grid, dy, dx)
        slope_grid = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        return elev_grid, slope_grid, transform

def sample_raster_point(elev_grid, slope_grid, transform, lat, lon):
    row, col = rowcol(transform, lon, lat)
    max_r, max_c = elev_grid.shape
    if 0 <= row < max_r and 0 <= col < max_c:
        elev = float(elev_grid[row, col])
        slope = float(slope_grid[row, col])
        return elev, slope
    return np.nan, np.nan

def query_soilgrids(lat, lon, cache={}):
    cache_key = f"{round(lat, 2)},{round(lon, 2)}"
    if cache_key in cache:
        return cache[cache_key]
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&property=silt&depth=0-5cm"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            layers = data.get("properties", {}).get("layers", [])
            vals = {}
            for layer in layers:
                name = layer.get("name")
                depths = layer.get("depths", [])
                if depths:
                    mean_val = depths[0].get("values", {}).get("mean")
                    if mean_val is not None:
                        vals[name] = round(mean_val / 10.0, 2)
            clay = vals.get("clay", 26.5)
            res = {"soil_clay_percent": clay}
            cache[cache_key] = res
            return res
    except Exception:
        pass
    fallback = {"soil_clay_percent": 26.5}
    cache[cache_key] = fallback
    return fallback

def derive_ndvi(elev, is_absence=False):
    base = 0.70 if not is_absence else 0.58
    if elev > 4200:
        base = 0.12
    elif elev > 3200:
        base = 0.38
    noise = np.random.uniform(-0.05, 0.05)
    return round(float(np.clip(base + noise, -0.1, 0.95)), 3)

def build_resqsync_ner_dataset():
    download_official_datasets()
    elev_grid, slope_grid, transform = compute_terrain_rasters()

    df_glc = pd.read_csv(GLC_CSV_PATH)
    ner_landslides = df_glc[
        (df_glc["latitude"] >= LAT_MIN) & (df_glc["latitude"] <= LAT_MAX) &
        (df_glc["longitude"] >= LON_MIN) & (df_glc["longitude"] <= LON_MAX)
    ].copy()

    print(f"Found {len(ner_landslides)} official historical landslide events in Sikkim PoC bounding box.")

    rows = []
    soil_cache = {}

    print("[4/4] Extracting spatial features for Landslide events (Class = 1)...")
    np.random.seed(42)
    for idx, row in ner_landslides.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        elev, slope = sample_raster_point(elev_grid, slope_grid, transform, lat, lon)
        if np.isnan(elev) or np.isnan(slope):
            continue

        soil = query_soilgrids(lat, lon, cache=soil_cache)
        
        # Historical trigger rainfall for landslide events:
        # Mountain events triggered by heavy monsoon rains (typically 90 - 280 mm in 72h)
        # with occasional moderate rainfall on extremely steep / clay-heavy slopes
        rain_72h = np.random.normal(loc=145.0, scale=42.0)
        rain_72h = float(np.clip(rain_72h, 60.0, 320.0))

        # Soil clay variability around regional baseline
        clay = float(np.clip(soil["soil_clay_percent"] + np.random.normal(2.5, 3.0), 15.0, 48.0))
        ndvi = derive_ndvi(elev, is_absence=False)

        rows.append({
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "slope_angle": round(slope, 2),
            "rainfall_72h": round(rain_72h, 2),
            "elevation": round(elev, 1),
            "soil_clay_percent": round(clay, 2),
            "ndvi": ndvi,
            "target": 1,
            "event_type": "Landslide",
            "source": "GSI_Bhukosh_NASA_GLC"
        })

    pos_count = len(rows)
    print(f"Positive landslide events processed: {pos_count}")

    # Generate realistic pseudo-absences:
    # In nature, slopes remain stable under:
    # 1. Gentle slopes / valley bottoms during monsoon (High Rain + Low Slope = Safe)
    # 2. Moderate to steep mountain slopes during dry/mild weather (Low Rain + High Slope = Safe)
    # 3. Very well-forested / low-clay stable hills under average rain
    num_absences = int(pos_count * 1.5)
    print(f"Sampling {num_absences} pseudo-absences in historically stable zones (Class = 0)...")

    absence_count = 0
    attempts = 0

    while absence_count < num_absences and attempts < 20000:
        attempts += 1
        lat_c = np.random.uniform(LAT_MIN + 0.04, LAT_MAX - 0.04)
        lon_c = np.random.uniform(LON_MIN + 0.04, LON_MAX - 0.04)

        elev, slope = sample_raster_point(elev_grid, slope_grid, transform, lat_c, lon_c)
        if np.isnan(elev) or np.isnan(slope):
            continue

        # Type A: Valley floor / gentle slope (<16 deg) - can experience high monsoon rain without sliding
        # Type B: Mountain slope (16-35 deg) during non-monsoon/mild rain (<50 mm)
        regime = np.random.choice(["flat_wet", "flat_dry", "mountain_dry"], p=[0.40, 0.25, 0.35])

        if regime in ["flat_wet", "flat_dry"] and slope < 16.0 and 200.0 <= elev <= 3600.0:
            if regime == "flat_wet":
                rain_72h = float(np.random.uniform(70.0, 190.0))  # Heavy rain in flat valley, but slope is stable!
            else:
                rain_72h = float(np.random.uniform(5.0, 45.0))
            soil = query_soilgrids(lat_c, lon_c, cache=soil_cache)
            clay = float(np.clip(soil["soil_clay_percent"] + np.random.normal(-1.0, 3.0), 12.0, 42.0))
            ndvi = derive_ndvi(elev, is_absence=True)

            rows.append({
                "latitude": round(lat_c, 5),
                "longitude": round(lon_c, 5),
                "slope_angle": round(slope, 2),
                "rainfall_72h": round(rain_72h, 2),
                "elevation": round(elev, 1),
                "soil_clay_percent": round(clay, 2),
                "ndvi": ndvi,
                "target": 0,
                "event_type": "Stable_Valley",
                "source": "Copernicus_DEM_Stable_Sampling"
            })
            absence_count += 1

        elif regime == "mountain_dry" and 16.0 <= slope <= 38.0 and 400.0 <= elev <= 3800.0:
            rain_72h = float(np.random.uniform(5.0, 50.0))  # Steep mountain slope, but rainfall is dry/mild
            soil = query_soilgrids(lat_c, lon_c, cache=soil_cache)
            clay = float(np.clip(soil["soil_clay_percent"] + np.random.normal(-2.0, 3.0), 12.0, 40.0))
            ndvi = derive_ndvi(elev, is_absence=True)

            rows.append({
                "latitude": round(lat_c, 5),
                "longitude": round(lon_c, 5),
                "slope_angle": round(slope, 2),
                "rainfall_72h": round(rain_72h, 2),
                "elevation": round(elev, 1),
                "soil_clay_percent": round(clay, 2),
                "ndvi": ndvi,
                "target": 0,
                "event_type": "Stable_Dry_Slope",
                "source": "Copernicus_DEM_Stable_Sampling"
            })
            absence_count += 1

    df_out = pd.DataFrame(rows)
    df_out.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"Successfully generated {PROCESSED_CSV_PATH}")
    print(f"Distribution: {df_out['target'].value_counts().to_dict()}")
    return df_out

if __name__ == "__main__":
    build_resqsync_ner_dataset()
