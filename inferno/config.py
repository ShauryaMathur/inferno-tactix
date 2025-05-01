import os

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
CACHE_PICKLE_PATH = os.path.expanduser("wildfire_cache.pkl") 

#PARALLELISM & BATCHING ────────────────────────────────────────────────────
PREFETCH_WORKERS  = 24
PROCESS_WORKERS   = 16
BATCH_SIZE        = 200

#LOGGING ───────────────────────────────────────────────────────────────────
LOG_FILE          = "make_dataset.log"
LOG_LEVEL         = "INFO"
