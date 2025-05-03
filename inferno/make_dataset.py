"""
Note: 

- Uses config.py for all settings/customizations.
- The dataset might contain few fill values(32767.0) (drop those rows!) because of few coords very close to the canadian border     and water bodies which might not be present in the GRIDMET database.
- Running this script might result in very high memory utilization, so we have provided the cache file to make it faster and       easier.

This gets the wildfire/no-wildfire event data with 75-day windows of climate features (e.g., temperature, humidity) fetched from remote NetCDF sources. It uses parallel processing and a local pickle cache to avoid redundant downloads, only re-fetching missing or current-year data. Configurable via config.py, it shows progress with tqdm and writes the final dataset in chunks to CSV, making it efficient, scalable, and easy to reproduce for large geospatial time series tasks.

Author:  Shreyas Bellary Manjunath <> Shaurya Mathur
Date:    2025-05-01

"""

from __future__ import annotations
import os
import sys
import time
import logging
import random
import pickle
import signal
import gc

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

import zstandard as zstd
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from tqdm import tqdm, trange

import config

#GLOBAL CACHE ─────────────────────────────────────────────────────────────
_DATA_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}

def setup_logging():
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.LOG_FILE:
        handlers.append(logging.FileHandler(config.LOG_FILE))
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

#EXIT HANDLER ────────────────────────────────────────────────────
def _graceful_exit(signum, frame):
    logging.warning("Signal %s received – exiting …", signum)
    sys.exit(1)

for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _graceful_exit)

#CACHE LOAD/SAVE ──────────────────────────────────────────────────────────
def _stream_load(zfh) -> Dict:
    unpickler = pickle.Unpickler(zfh)
    cache: Dict = {}
    try:
        first = unpickler.load()
    except EOFError:
        return cache
    if isinstance(first, dict):
        return first
    key, val = first
    cache[key] = val
    while True:
        try:
            key, val = unpickler.load()
            cache[key] = val
        except EOFError:
            break
    return cache

def load_cache() -> Dict[Tuple[str,int], Any]:
    if not os.path.exists(config.CACHE_PICKLE_PATH):
        return {}
    dctx = zstd.ZstdDecompressor()
    try:
        print("Loading cache...")
        with open(config.CACHE_PICKLE_PATH, "rb") as fh, dctx.stream_reader(fh) as zfh:
            cache = _stream_load(zfh)
        logging.info("Loaded cache (%d entries) from %s",
                     len(cache), config.CACHE_PICKLE_PATH)
        return cache
    except Exception:
        logging.warning("Failed to load cache; starting fresh.", exc_info=True)
        try:
            print("Skipping load!")
        except OSError:
            pass
        return {}

def _stream_save(tmp_path: str):
    cctx = zstd.ZstdCompressor(level=config.ZSTD_LEVEL, threads=config.ZSTD_THREADS)
    with open(tmp_path, "wb") as fh, cctx.stream_writer(fh) as zfh:
        pickler = pickle.Pickler(zfh, protocol=pickle.HIGHEST_PROTOCOL)
        for item in _DATA_CACHE.items():
            pickler.dump(item)

def save_cache():
    tmp_path = config.CACHE_PICKLE_PATH + ".tmp"
    try:
        _stream_save(tmp_path)
        os.replace(tmp_path, config.CACHE_PICKLE_PATH)
        logging.info("Saved cache (%d entries) to %s",
                     len(_DATA_CACHE), config.CACHE_PICKLE_PATH)
    except Exception:
        logging.error("Failed to save cache", exc_info=True)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

#REMOTE FETCH ─────────────────────────────────────────────────────────────
def _fetch_worker(feat: str, yr: int) -> Tuple[str,int,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    url = config.BASE_URL_TMPL.format(feat=feat, year=yr)
    ds = Dataset(url)
    try:
        lats      = ds.variables["lat"][:].astype(np.float32)
        lons      = ds.variables["lon"][:].astype(np.float32)
        raw       = ds.variables["day"][:]
        units     = ds.variables["day"].units
        dates     = np.array([np.datetime64(datetime(d.year,d.month,d.day), "D")
                              for d in num2date(raw, units=units)], dtype="datetime64[D]")
        var       = config.FEATURE_VAR_MAP[feat]
        data      = ds.variables[var][:].astype(np.float32)
        fill      = getattr(ds.variables[var], "_FillValue", np.nan)
    finally:
        ds.close()
    return feat, yr, lats, lons, dates, data, fill

def prefetch_years(years: List[int], features: List[str], workers: int):
    tasks = [(f,y) for f in features for y in years]
    logging.info("Prefetching %d feature-year combos with %d workers", len(tasks), workers)
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_fetch_worker, f, y): (f,y) for f,y in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Prefetch", unit="task"):
            f,y,lats,lons,dates,data,fill = fut.result()
            _DATA_CACHE[(f,y)] = {
                "lats":  lats,
                "lons":  lons,
                "dates": dates,
                "data":  data,
                "fill":  fill
            }
    logging.info("Prefetch complete; cache size = %d", len(_DATA_CACHE))

#WINDOW EXTRACTOR ─────────────────────────────────────────────────
def extract_window(lat: float, lon: float,
                   when: datetime, feat: str) -> pd.Series:
    start = when - timedelta(days=60)
    end   = when + timedelta(days=14)
    years = {start.year, end.year}
    vals, dts = [], []

    for yr in years:
        slab = _DATA_CACHE.get((feat, yr))
        if slab is None:
            continue

        lat_i = np.abs(slab["lats"] - lat).argmin()
        lon_i = np.abs(slab["lons"] - lon).argmin()

        mask = (slab["dates"] >= start) & (slab["dates"] <= end)
        idx  = np.where(mask)[0]
        v    = slab["data"][idx, lat_i, lon_i]
        if not np.isnan(slab["fill"]):
            v = np.where(v == slab["fill"], np.nan, v)
        vals.append(v)
        dts.append(slab["dates"][idx])

    if not vals:
        return pd.Series(dtype=float)

    all_vals  = np.concatenate(vals)
    all_dates = np.concatenate(dts)
    order     = np.argsort(all_dates)
    return pd.Series(all_vals[order], index=all_dates[order])

#ROW PROCESSOR ───────────────────────────────────────────────────────────
def process_entry(row: Dict[str, Any], features: List[str]) -> pd.DataFrame:
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    when = pd.to_datetime(row["datetime"]).normalize()

    data = {f: extract_window(lat, lon, when, f) for f in features}
    df = pd.DataFrame(data)
    df["latitude"], df["longitude"], df["datetime"] = lat, lon, df.index

    is_fire = str(row.get("Wildfire","")).strip().lower() == "yes"
    if is_fire:
        df["Wildfire"] = ["No"]*max(0,len(df)-15) + ["Yes"]*min(15,len(df))
    else:
        df["Wildfire"] = ["No"]*len(df)

    return df[["latitude","longitude","datetime","Wildfire"] + features]

#MAIN ────────────────────────────────────────────────────────────────────
def main():
    setup_logging()

    global _DATA_CACHE
    _DATA_CACHE = load_cache()

    df = pd.read_csv(config.INPUT_CSV, parse_dates=["datetime"])
    yrs = sorted(df["datetime"].dt.year.unique())
    to_fetch = set(range(yrs[0]-1, yrs[-1]+1)) - {y for (_,y) in _DATA_CACHE.keys()}
    if to_fetch:
        logging.info("Fetching missing years: %s", sorted(to_fetch))
        prefetch_years(sorted(to_fetch), config.FEATURES, config.PREFETCH_WORKERS)
        if not os.path.exists(config.CACHE_PICKLE_PATH) && config.SAVE_CACHE:
            save_cache()
    else:
        logging.info("All needed years present in cache")

    records = df.to_dict(orient="records")
    first = True
    for start in trange(0,len(records), config.BATCH_SIZE, desc="Batches"):
        batch = records[start:start+config.BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=config.PROCESS_WORKERS) as pool:
            futures = [pool.submit(process_entry, r, config.FEATURES) for r in batch]
            results = [f.result() for f in as_completed(futures)]

        chunk = pd.concat(results, ignore_index=True)
        mode = "w" if first else "a"
        header = first
        chunk.to_csv(config.OUTPUT_CSV, mode=mode, header=header, index=False)
        first = False
        del chunk, results, futures
        gc.collect()

    logging.info("Finished! Output saved to %s", config.OUTPUT_CSV)


if __name__ == "__main__":
    main()