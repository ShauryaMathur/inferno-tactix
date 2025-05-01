"""
Note: Uses config.py for all settings/customizations.

This gets the wildfire/no-wildfire event data with 75-day windows of climate features (e.g., temperature, humidity) fetched from remote NetCDF sources. It uses parallel processing and a local pickle cache to avoid redundant downloads, only re-fetching missing or current-year data. Configurable via config.py, it shows progress with tqdm and writes the final dataset in chunks to CSV, making it efficient, scalable, and easy to reproduce for large geospatial time series tasks.

Author:  INFERNO TACTICS
Date:    2025-05-01
"""

import os
import sys
import time
import logging
import random
import pickle

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from tqdm import tqdm, trange

import config

_DATA_CACHE: dict = {}

#LOGGING SETUP ─────────────────────────────────────────────────────────────
def setup_logging():
    fmt = "%(asctime)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.LOG_FILE:
        handlers.append(logging.FileHandler(config.LOG_FILE))
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

#CACHE LOAD / SAVE ─────────────────────────────────────────────────────────
def load_cache():
    """Load cache from pickle if present."""
    if os.path.exists(config.CACHE_PICKLE_PATH):
        try:
            with open(config.CACHE_PICKLE_PATH, "rb") as f:
                cache = pickle.load(f)
            logging.info("Loaded cache from %s (%d entries)",
                         config.CACHE_PICKLE_PATH, len(cache))
            return cache
        except Exception:
            logging.warning("Failed to load cache pickle; starting fresh", exc_info=True)
    return {}

def save_cache():
    """Persist _DATA_CACHE to disk."""
    try:
        with open(config.CACHE_PICKLE_PATH, "wb") as f:
            pickle.dump(_DATA_CACHE, f)
        logging.info("Saved cache to %s (%d entries)",
                     config.CACHE_PICKLE_PATH, len(_DATA_CACHE))
    except Exception:
        logging.error("Failed to save cache pickle", exc_info=True)

#RETRY UTILITY ─────────────────────────────────────────────────────────────
def exponential_backoff_retry(fn, args, retries=3, base_delay=1.0):
    for attempt in range(1, retries+1):
        try:
            return fn(*args)
        except Exception as e:
            if attempt == retries:
                logging.error("Function %s failed after %d attempts", fn.__name__, retries, exc_info=True)
                raise
            delay = base_delay * (2 ** (attempt-1)) + random.random()
            logging.warning("Attempt %d/%d of %s failed: %s; retrying in %.1f s",
                            attempt, retries, fn.__name__, e, delay)
            time.sleep(delay)

#PREFETCH WORKER ───────────────────────────────────────────────────────────
def _prefetch_worker(feat: str, yr: int) -> Tuple[str,int,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    url = config.BASE_URL_TMPL.format(feat=feat, year=yr)
    ds = Dataset(url)
    try:
        lats      = ds.variables["lat"][:]
        lons      = ds.variables["lon"][:]
        raw_times = ds.variables["day"][:]
        units     = ds.variables["day"].units
        dates     = np.array([
            datetime(d.year, d.month, d.day)
            for d in num2date(raw_times, units=units)
        ])
        var_name  = config.FEATURE_VAR_MAP[feat]
        data      = ds.variables[var_name][:]
        fill      = getattr(ds.variables[var_name], "_FillValue", None)
    finally:
        ds.close()
    return feat, yr, lats, lons, dates, data, fill

def prefetch_years(years: List[int], features: List[str], workers: int):
    tasks = [(feat, yr) for feat in features for yr in years]
    logging.info("Prefetching %d feature-year combos with %d workers", len(tasks), workers)

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(exponential_backoff_retry, _prefetch_worker, (feat, yr)): (feat, yr)
            for feat, yr in tasks
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Prefetching",
                        unit="task"):
            feat, yr = futures[fut]
            f, y, lats, lons, dates, data, fill = fut.result()
            _DATA_CACHE[(f, y)] = {
                "lats": lats, "lons": lons,
                "dates": dates, "data": data,
                "fill":  fill
            }

    logging.info("Prefetch done; cache size = %d", len(_DATA_CACHE))

#EXTRACTION ────────────────────────────────────────────────────────────────
def extract_feature_window(lat: float, lon: float, date: datetime, feature: str) -> pd.Series:
    start = date - timedelta(days=60)
    end   = date + timedelta(days=14)
    years = range(start.year, end.year+1)

    base = _DATA_CACHE[(feature, start.year)]
    lat_idx = np.abs(base["lats"] - lat).argmin()
    lon_idx = np.abs(base["lons"] - lon).argmin()

    arrs, dts = [], []
    for yr in years:
        cache = _DATA_CACHE[(feature, yr)]
        mask  = (cache["dates"] >= start) & (cache["dates"] <= end)
        idx   = np.where(mask)[0]
        vals  = cache["data"][idx, lat_idx, lon_idx]
        if cache["fill"] is not None:
            vals = np.where(vals == cache["fill"], np.nan, vals)
        arrs.append(vals)
        dts.append(cache["dates"][idx])

    all_vals  = np.concatenate(arrs)
    all_dates = np.concatenate(dts)
    order     = np.argsort(all_dates)
    return pd.Series(all_vals[order], index=all_dates[order])

def process_entry(row: pd.Series, features: List[str]) -> pd.DataFrame:
    lat, lon = float(row["latitude"]), float(row["longitude"])
    date     = pd.to_datetime(row["datetime"])
    data = {feat: extract_feature_window(lat, lon, date, feat)
            for feat in features}

    df = pd.DataFrame(data)
    df["latitude"]  = lat
    df["longitude"] = lon
    df["datetime"]  = df.index

    flag = str(row.get("Wildfire","")).strip().lower() == "yes"
    df["Wildfire"] = (["Yes"]*15 + ["No"]*(len(df)-15)) if flag else ["No"]*len(df)

    return df[["latitude","longitude","datetime","Wildfire"] + features]

#MAIN ─────────────────────────────────────────────────────────────────────
def main():
    setup_logging()

    df = pd.read_csv(config.INPUT_CSV, parse_dates=["datetime"])
    years_in = sorted(df["datetime"].dt.year.unique())
    min_year, max_year = years_in[0], years_in[-1]
    years_to_prefetch = list(range(min_year - 1, max_year + 1))

    global _DATA_CACHE
    _DATA_CACHE = load_cache()
    current_year = datetime.now().year
    missing_years = set(years_to_prefetch) - { y for (_,y) in _DATA_CACHE.keys() }
    missing_years.add(current_year)

    if missing_years:
        logging.info("Need to fetch years: %s", sorted(missing_years))
        prefetch_years(sorted(missing_years), config.FEATURES, config.PREFETCH_WORKERS)
        save_cache()
    else:
        logging.info("Cache already covers years %d–%d (and current year)", min_year-1, max_year)

    total = len(df)
    rows  = df.to_dict(orient="records")
    logging.info("Processing %d rows in batches of %d with %d workers",
                 total, config.BATCH_SIZE, config.PROCESS_WORKERS)

    for batch_start in trange(0, total, config.BATCH_SIZE,
                             desc="Batches", unit="batch"):
        batch = rows[batch_start: batch_start + config.BATCH_SIZE]
        with ProcessPoolExecutor(max_workers=config.PROCESS_WORKERS) as exe:
            futures = [exe.submit(process_entry, pd.Series(r), config.FEATURES)
                       for r in batch]
            dfs = [f.result() for f in as_completed(futures)]

        chunk = pd.concat(dfs, ignore_index=True)
        mode   = "w" if batch_start == 0 else "a"
        header = (batch_start == 0)
        chunk.to_csv(config.OUTPUT_CSV, mode=mode, header=header, index=False)

    logging.info("Done! Output at %s", config.OUTPUT_CSV)

if __name__ == "__main__":
    main()