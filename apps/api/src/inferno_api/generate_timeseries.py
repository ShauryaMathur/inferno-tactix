"""
Extractor for 75‑day GRIDMET time‑series with automatic
CFSv2 28‑day forecast stitching when the target date falls within the last
15 days.

Features:

    - Historical data (60 days before the target date) pulled from the
      GRIDMET collection.

    - Forecast gap‑fill (up to 15 days after) pulled from the NWCSC CFSv2
      90‑day forecast products, using the most‑recent ensemble member whose
      time axis overlaps the requested dates.

    - Robust retry/back‑off logic plus slice‑level ASCII OPeNDAP downloads
      for speed and low memory overhead.

    - Thread‑safe: each call opens/closes its own NetCDF handles.

75‑day window centred on 2025‑05‑01 near Salmon, ID
python extract_75day_data.py 44.05 -113.56 2025-05-01

Author:  Shreyas Bellary Manjunath <> Shaurya Mathur
Date:    2025-05-01

"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, num2date
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

MAX_RETRIES       = 1
BACKOFF           = 2       
EARLIEST_DATE     = datetime(1980, 1, 1)
FORECAST_LOOKAHEAD = 15      

FEATURES: Sequence[str] = [
      "pr","tmmn", "tmmx","vs","sph", "srad", "bi","rmax", "rmin", "fm100", "fm1000", "erc",  "pet", "vpd",
]
    
_MEDIAN_ONLY = {"vpd"}
FORECAST_DERIVED = {"rmax", "rmin"}
    
# "etr" dropped bcz extra feature not in forecast 
    
FEATURE_VAR_MAP: Dict[str, str] = {
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
    "vpd":   "mean_vapor_pressure_deficit",
}

GRIDMET_URL = (
    "http://thredds.northwestknowledge.net:8080"
    "/thredds/dodsC/MET/{feat}/{feat}_{year}.nc"
)

_BASE_FCST = (
    "http://thredds.northwestknowledge.net:8080/"
    "thredds/dodsC/NWCSC_INTEGRATED_SCENARIOS_ALL_CLIMATE/"
    "cfsv2_metdata_90day/"
)


_FCST_PATTERNS = [
    "cfsv2_metdata_forecast_{feat}_daily_{hr}_{ens}_{day}.nc",
    "cfsv2_metdata_forecast_{feat}_daily.nc",
    "cfsv2_metdata_forecast_48ENS_{feat}_daily_4d.nc",
]

_HOUR_ENS_DAY = [
    (h, e, d)
    for h in ("12", "18", "06", "00")
    for e in ("1", "2", "3", "4")
    for d in ("1", "2", "0")
]

_session: requests.Session | None = None


def _sat_vp(temp_c: np.ndarray | float) -> np.ndarray | float:

    return 0.6108 * np.exp(17.2693882 * temp_c / (temp_c + 237.3))

def _rh_from_vpd(tmin_k, tmax_k, vpd_kpa):

    tmin_c = tmin_k - 273.15
    tmax_c = tmax_k - 273.15
    tmean_c = 0.5 * (tmin_c + tmax_c)

    ea      = _sat_vp(tmean_c) - vpd_kpa         
    rh_max  = 100 * np.clip(ea / _sat_vp(tmin_c), 0, 1)
    rh_min  = 100 * np.clip(ea / _sat_vp(tmax_c), 0, 1)
    return rh_max, rh_min


def _cftime_to_datetime(cftime_arr):
    return [
        datetime(d.year, d.month, d.day,
                 getattr(d, "hour", 0),
                 getattr(d, "minute", 0),
                 getattr(d, "second", 0))
        for d in cftime_arr
    ]


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session

def _open_dataset(url: str) -> Dataset:

    for attempt in range(MAX_RETRIES):
        try:
            return Dataset(url, decode_times=False)  
        except Exception as ex:
            logger.warning("%s → open attempt %d failed: %s", url, attempt + 1, ex)
            time.sleep(BACKOFF ** attempt)
    raise RuntimeError(f"Cannot open dataset after {MAX_RETRIES} attempts → {url}")


def _nearest_grid_indices(ds: Dataset, lat: float, lon: float) -> Tuple[int, int]:
    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    return np.abs(lats - lat).argmin(), np.abs(lons - lon).argmin()


def _time_slice(ds: Dataset, start: datetime, end: datetime) -> Tuple[np.ndarray, np.ndarray]:
    time_var = next(v for v in ds.variables if v.lower().startswith("day") or v == "time")
    raw = ds.variables[time_var][:]
    dates = num2date(raw, ds.variables[time_var].units)
    mask = (dates >= start) & (dates <= end)
    return np.where(mask)[0], dates[mask]


def _ascii_slice(url_base: str, var: str, t0: int, t1: int, lat_i: int, lon_i: int) -> np.ndarray:

    slice_url = (
        f"{url_base}.ascii?{var}[{t0}:1:{t1}]"
        f"[{lat_i}:1:{lat_i}]"
        f"[{lon_i}:1:{lon_i}]"
    )
    resp = _get_session().get(slice_url, timeout=60)
    resp.raise_for_status()

    vals: List[float] = []
    for line in resp.text.splitlines():
        if "," in line:
            try:
                vals.append(float(line.split(",", 1)[1].strip()))
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)


def _find_forecast_file(
    feature: str,
    start:   datetime,
    end:     datetime
) -> Tuple[str, Dataset]:

    if feature in _MEDIAN_ONLY:
        url = _BASE_FCST + _FCST_PATTERNS[1].format(feat=feature)
        ds  = _open_dataset(url)
        if _time_slice(ds, start, end)[0].size:
            logger.info("%s selected (median‑only variable)", Path(url).name)
            return url, ds
        ds.close()
        
    for hr, ens, day in _HOUR_ENS_DAY:
        url = _BASE_FCST + _FCST_PATTERNS[0].format(
            feat=feature, hr=hr, ens=ens, day=day
        )
        try:
            ds = _open_dataset(url)
            if _time_slice(ds, start, end)[0].size:
                logger.info("%s selected", Path(url).name)
                return url, ds
            ds.close()
        except Exception:
            pass  

    url = _BASE_FCST + _FCST_PATTERNS[1].format(feat=feature)
    try:
        ds = _open_dataset(url)
        if _time_slice(ds, start, end)[0].size:
            logger.info("%s selected (ensemble median)", Path(url).name)
            return url, ds
        ds.close()
    except Exception:
        pass

    url = _BASE_FCST + _FCST_PATTERNS[2].format(feat=feature)
    try:
        ds = _open_dataset(url)
        if _time_slice(ds, start, end)[0].size:
            logger.info("%s selected (48‑ENS 4‑D)", Path(url).name)
            return url, ds
        ds.close()
    except Exception:
        pass

    raise RuntimeError(
        f"No forecast file covering {start.date()}‑{end.date()} for feature '{feature}'"
    )


def _fetch_forecast_series(
    feature: str,
    varname: str,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
) -> pd.Series:
    url, ds = _find_forecast_file(feature, start, end)
    lat_i, lon_i = _nearest_grid_indices(ds, lat, lon)
    idxs, dates = _time_slice(ds, start, end)

    if not idxs.size:
        ds.close()
        return pd.Series(dtype=float, name=feature)

    t0, t1 = idxs.min(), idxs.max()
    raw = _ascii_slice(url, varname, t0, t1, lat_i, lon_i)

    scale = getattr(ds.variables[varname], "scale_factor", 1.0)
    offset = getattr(ds.variables[varname], "add_offset", 0.0)
    fillv = getattr(ds.variables[varname], "_FillValue", None)
    ds.close()

    if fillv is not None:
        raw = np.where(raw == fillv, np.nan, raw)

    dts_py = [datetime(d.year, d.month, d.day) for d in dates]
    return pd.Series(raw * scale + offset, index=pd.DatetimeIndex(dts_py), name=feature)


def get_75day_timeseries(lat: float, lon: float, date: datetime) -> pd.DataFrame:

    date = datetime(date.year, date.month, date.day) 

    window_start = max(date - timedelta(days=60), EARLIEST_DATE)
    window_end = date + timedelta(days=14)

    now = datetime.utcnow()
    forecast_needed = window_end > now

    hist_end = min(window_end, now)
    hist_years = range(window_start.year, hist_end.year + 1)

    sample_ds = _open_dataset(GRIDMET_URL.format(feat=FEATURES[0], year=window_start.year))
    lat_i, lon_i = _nearest_grid_indices(sample_ds, lat, lon)
    sample_ds.close()

    series: List[pd.Series] = []

    for feat in FEATURES:
        varname = FEATURE_VAR_MAP[feat]

        hist_vals: List[np.ndarray] = []
        hist_dates: List[np.ndarray] = []

        for yr in hist_years:
            url = GRIDMET_URL.format(feat=feat, year=yr)
            ds = _open_dataset(url)
            idxs, dates = _time_slice(ds, window_start, hist_end)
            if idxs.size:
                t0, t1 = idxs.min(), idxs.max()
                raw = _ascii_slice(url, varname, t0, t1, lat_i, lon_i)
                scale = getattr(ds.variables[varname], "scale_factor", 1.0)
                offset = getattr(ds.variables[varname], "add_offset", 0.0)
                fillv = getattr(ds.variables[varname], "_FillValue", None)
                if fillv is not None:
                    raw = np.where(raw == fillv, np.nan, raw)
                hist_vals.append(raw * scale + offset)
                hist_dates.append(dates)
            ds.close()

        h_ser = pd.Series(dtype=float, name=feat)
        if hist_vals:
            py_dates = _cftime_to_datetime(np.concatenate(hist_dates))
            h_ser = pd.Series(
                np.concatenate(hist_vals),
                index=pd.DatetimeIndex(py_dates),
                name=feat,
            )


        if forecast_needed:
            if feat in FORECAST_DERIVED:
                merged = h_ser
            else:
                f_start = date                     
                f_end   = window_end
                try:
                    f_ser  = _fetch_forecast_series(feat, varname, lat, lon,
                                                   f_start, f_end)
                    merged = pd.concat([h_ser, f_ser]).sort_index()
                except Exception as fx:
                    logger.warning("Forecast fetch failed for %s: %s", feat, fx)
                    merged = h_ser
        else:
            merged = h_ser

        series.append(merged)

    full_index = pd.date_range(window_start, window_end, freq="D")
    df = pd.concat(series, axis=1).reindex(full_index)
    df["vpd"] = (df["vpd"].interpolate(method="time", limit_direction="both").ffill().bfill())
    df = df.interpolate(method="time", limit_direction="both")
    df = df.ffill().bfill()

    need_rh = df["rmax"].isna() | df["rmin"].isna()
    if need_rh.any():
        tmin = df.loc[need_rh, "tmmn"]
        tmax = df.loc[need_rh, "tmmx"]
        vpd  = df.loc[need_rh, "vpd"]

        valid = ~(tmin.isna() | tmax.isna() | vpd.isna())
        if valid.any():
            rh_max, rh_min = _rh_from_vpd(tmin[valid], tmax[valid], vpd[valid])
            df.loc[need_rh[need_rh].index[valid], "rmax"] = rh_max
            df.loc[need_rh[need_rh].index[valid], "rmin"] = rh_min
    
    
    df["latitude"] = lat
    df["longitude"] = lon
    df["datetime"] = df.index
    return df.reset_index(drop=True)


#CLI──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GRIDMET + CFSv2 75‑day time‑series extractor"
    )
    parser.add_argument("lat", type=float, help="Latitude in decimal degrees")
    parser.add_argument("lon", type=float, help="Longitude in decimal degrees")
    parser.add_argument("date", type=str, help="Target date YYYY‑MM‑DD")
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d")
    out_df = get_75day_timeseries(args.lat, args.lon, target_date)

    csv_name = f"gridmet_75d_{args.lat:.3f}_{args.lon:.3f}_{args.date}.csv"
    out_df.to_csv(csv_name, index=False)
    print(f"Saved → {csv_name} ({len(out_df)} rows)")