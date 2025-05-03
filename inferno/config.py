"""
Note: 

- Download the pickled cache file for making the data from : https://buffalo.box.com/s/znj94lmk8gdxavswluh7umo558w99wa4.
- Download the wildfire incidents from the same box link present above for the coordinate picker.

Author:  Shreyas Bellary Manjunath <> Shaurya Mathur
Date:    2025-05-01

"""


import os

                                            #──────────────────────────────────────────────#
                                            #         Config for make_coords.py            #
                                            #──────────────────────────────────────────────#
            
            
#Bounding box (CONUS) ──────────────────────────────────────────────────
MIN_LAT = 24.396308
MAX_LAT = 49.384358
MIN_LON = -125.0
MAX_LON = -66.93457

#Positive-set (Wildfire = 'Yes')───────────────────────────────
POSITIVE_MIN_DISTANCE_KM = 5       
POSITIVE_MIN_HOURS       = 2       

#Negative-set (Wildfire = 'No')──────────────────────────────────────
FAR_NEGATIVE_COUNT       = 5_000   
FAR_NEGATIVE_MIN_KM      = 100

NEAR_NEGATIVE_PER_POS    = 1       
NEAR_NEGATIVE_MAX_KM     = 100
NEAR_NEGATIVE_MIN_DAYS   = 90
NEAR_NEGATIVE_MAX_DAYS   = 120        

MIN_NO_SPATIAL_KM       = 5          #radius for same 4‑km grid
MIN_NO_TEMPORAL_DAYS    = 120        #look‑back window

#Date window for random generation ─────────────────────────────────────
DATE_START = "2014-03-01"
DATE_END   = "2025-03-31"

#Threading & performance ───────────────────────────────────────────────
MAX_THREADS = 32    
SEED        = 42   

#INPUT/OUTPUT ─────────────────────────────────────────────────────────────

INCIDENTS = "wildfire_incidents.csv"
OUT = "data_coordinates.csv"


                                            #──────────────────────────────────────────────#
                                            #         Config for make_dataset.py           #
                                            #──────────────────────────────────────────────#

            
#INPUT/OUTPUT ─────────────────────────────────────────────────────────────

INPUT_CSV        = "data_coordinates.csv"
OUTPUT_CSV       = "Wildfire_Dataset.csv"

#FEATURES ──────────────────────────────────────────────────────────────────

FEATURES = [
    "pr","rmax","rmin","sph","srad",
    "tmmn","tmmx","vs","bi","fm100",
    "fm1000","erc","etr","pet","vpd"
]

FEATURE_VAR_MAP = {
    "pr":    "precipitation_amount",
    "rmax":  "relative_humidity",
    "rmin":  "relative_humidity",
    "sph":   "specific_humidity",
    "srad":  "surface_downwelling_shortwave_flux_in_air",
    "tmmn":  "air_temperature",
    "tmmx":  "air_temperature",
    "vs":    "wind_speed",
    "bi":    "burning_index_g",
    "fm100": "dead_fuel_moisture_100hr",
    "fm1000":"dead_fuel_moisture_1000hr",
    "erc":   "energy_release_component-g",
    "etr":   "potential_evapotranspiration",
    "pet":   "potential_evapotranspiration",
    "vpd":   "mean_vapor_pressure_deficit"
}

#NETCDF/CACHE ────────────────────────────────────────────────────────────
BASE_URL_TMPL     = (
    "http://thredds.northwestknowledge.net:8080/"
    "thredds/dodsC/MET/{feat}/{feat}_{year}.nc"
)
CACHE_PICKLE_PATH = os.path.expanduser("/projects/academic/courses/cse676s25/sbellary/Inferno_Tactics/wildfire_cache.pkl.zst") 
SAVE_CACHE = False

#PARALLELISM & BATCHING ────────────────────────────────────────────────────
PREFETCH_WORKERS  = 24
PROCESS_WORKERS   = 64
BATCH_SIZE        = 500
ZSTD_LEVEL        = 10                      
ZSTD_THREADS      = max(1, os.cpu_count() // 2)   

#LOGGING ───────────────────────────────────────────────────────────────────
LOG_FILE          = "make_dataset.log"
LOG_LEVEL         = "INFO"
