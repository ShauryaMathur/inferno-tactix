"""
Creates *data_coordinates.csv* with columns:
    latitude, longitude, datetime, Wildfire

Rules implemented:

Positive samples ('Yes'):
  • keep ignitions inside CONUS bbox & USA polygon
  • successive kept rows must be ≥ POSITIVE_MIN_DISTANCE_KM apart **and**
    ≥ POSITIVE_MIN_HOURS apart in time

Negative samples ('No')  → three flavours:

  1. FAR negatives
       coord ≥ FAR_NEGATIVE_MIN_KM from *any* wildfire (date unrestricted)
  2. NEAR negatives
       for each wildfire  generate ≤100 km away
       but timestamp Δt ∈ [120,150] days from *that* wildfire
  3. 1 YEAR negatives
       for each wildfire, timestamp ≈ ±1 year

Author:  Shreyas Bellary Manjunath <> Shaurya Mathur
Date:    2025-05-01
              
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import threading
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from geopy import distance as geod
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import config as C

random.seed(C.SEED)
np.random.seed(C.SEED)

#Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,           
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    filename="make_coords.log",       
    filemode="w",
)
log = logging.getLogger("builder")

#Country polygon & bbox check ──────────────────────────────────────────
world = gpd.read_file(
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)
usa_poly = (world.loc[world["NAME"] == "United States of America", "geometry"].union_all())

def in_usa(lat: float, lon: float) -> bool:
    return usa_poly.contains(Point(lon, lat))

def in_bbox(lat: float, lon: float) -> bool:
    return (
        C.MIN_LAT <= lat <= C.MAX_LAT
        and C.MIN_LON <= lon <= C.MAX_LON
    )

#Fast haversine helpers ────────────────────────────────────────────────
EARTH_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))

#Helpers to shift coordinates by bearing & distance ────────────────────
def offset_coord(lat: float, lon: float, dist_km: float, bearing_deg: float):
    origin = geod.distance(kilometers=dist_km).destination((lat, lon), bearing_deg)
    return origin.latitude, origin.longitude

#Positive-set builder ──────────────────────────────────────────────────
def build_positive(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["Fire Discovery Date Time"])
    df = df.sort_values("datetime")

    kept = []
    last_idx = None
    for idx, row in df.iterrows():
        lat, lon, ts = row["y"], row["x"], row["datetime"]

        if not (in_bbox(lat, lon) and in_usa(lat, lon)):
            continue

        if last_idx is None:
            kept.append(idx)
            last_idx = idx
            continue

        prev = df.loc[last_idx]
        if (
            haversine_km(lat, lon, prev["y"], prev["x"]) >= C.POSITIVE_MIN_DISTANCE_KM
            and (ts - prev["datetime"]).total_seconds() / 3600.0 >= C.POSITIVE_MIN_HOURS
        ):
            kept.append(idx)
            last_idx = idx

    yes = (
        df.loc[kept, ["y", "x", "datetime"]]
        .rename(columns={"y": "latitude", "x": "longitude"})
        .assign(Wildfire="Yes")
        .reset_index(drop=True)
    )
    log.info("Positive rows kept: %d (from %d)", len(yes), len(df))
    return yes

#BallTree for distance queries ─────────────────────────────────────────
def make_balltree(df: pd.DataFrame) -> BallTree:
    radians = np.radians(df[["latitude", "longitude"]].values)
    return BallTree(radians, metric="haversine")

#FAR negatives (>=100 km) ───────────────────────────────────────────────
def far_negative_worker(seed: int, yes_tree: BallTree, result: list, dts: list,
                        lock: threading.Lock):
    rng = random.Random(seed)
    radius_rad_yes = C.FAR_NEGATIVE_MIN_KM / EARTH_KM
    radius_rad_no  = C.MIN_NO_SPATIAL_KM / EARTH_KM   

    while True:
        with lock:
            if len(result) >= C.FAR_NEGATIVE_COUNT:
                break

        lat = rng.uniform(C.MIN_LAT, C.MAX_LAT)
        lon = rng.uniform(C.MIN_LON, C.MAX_LON)
        if not in_usa(lat, lon):
            continue

        dist_yes, _ = yes_tree.query(
            np.radians([[lat, lon]]), k=1, return_distance=True
        )
        if dist_yes[0][0] < radius_rad_yes:
            continue

        rand_ts = pd.to_datetime(
            rng.uniform(
                pd.Timestamp(C.DATE_START).value,
                pd.Timestamp(C.DATE_END).value,
            )
        )

        with lock:
            if result:
                no_tree = BallTree(
                    np.radians([[r["latitude"], r["longitude"]] for r in result]),
                    metric="haversine",
                )
                idxs = no_tree.query_radius(
                    np.radians([[lat, lon]]), r=radius_rad_no, return_distance=False
                )[0]
                for j in idxs:
                    if abs((dts[j] - rand_ts).days) < C.MIN_NO_TEMPORAL_DAYS:
                        break
                else:  
                    result.append(
                        dict(latitude=lat, longitude=lon, datetime=rand_ts, Wildfire="No")
                    )
                    dts.append(rand_ts)
            else:
                result.append(
                    dict(latitude=lat, longitude=lon, datetime=rand_ts, Wildfire="No")
                )
                dts.append(rand_ts)

def build_far_negatives(yes_tree: BallTree) -> pd.DataFrame:
    shared, shared_ts, lock = [], [], threading.Lock()
    seeds = [random.randint(0, 2**31 - 1) for _ in range(C.MAX_THREADS)]
    thread_map(
        lambda s: far_negative_worker(s, yes_tree, shared, shared_ts, lock),
        seeds,
        max_workers=C.MAX_THREADS,
        desc="far-neg",
    )
    return pd.DataFrame(shared)

#NEAR negatives (≤100 km, ≥90d apart) ──────────────────────────────────
def build_near_negatives(yes_df: pd.DataFrame, yes_tree: BallTree) -> pd.DataFrame:
    records = []
    radius_rad_yes = C.NEAR_NEGATIVE_MAX_KM / EARTH_KM
    radius_rad_no  = C.MIN_NO_SPATIAL_KM / EARTH_KM

    no_coords = []
    no_times  = []

    for idx, row in tqdm(yes_df.iterrows(), total=len(yes_df), desc="near-neg"):
        lat0, lon0, ts0 = row["latitude"], row["longitude"], row["datetime"]

        for _ in range(C.NEAR_NEGATIVE_PER_POS):
            dist_km = random.uniform(0.1, C.NEAR_NEGATIVE_MAX_KM)
            bearing = random.uniform(0, 360)
            lat, lon = offset_coord(lat0, lon0, dist_km, bearing)

            if not (in_bbox(lat, lon) and in_usa(lat, lon)):
                continue

            idxs = yes_tree.query_radius(
                np.radians([[lat, lon]]), r=radius_rad_yes, return_distance=False
            )[0]
            ok = True
            for ji in idxs:
                ts_other = yes_df.iloc[ji]["datetime"]
                if abs((ts_other - ts0).days) < C.NEAR_NEGATIVE_MIN_DAYS:
                    ok = False
                    break
            if not ok:
                continue

            if no_coords:
                tmp_tree = BallTree(np.radians(no_coords), metric="haversine")
                close = tmp_tree.query_radius(
                    np.radians([[lat, lon]]), r=radius_rad_no, return_distance=False
                )[0]
                if any(abs((no_times[j] - ts0).days) < C.MIN_NO_TEMPORAL_DAYS for j in close):
                    continue

            delta_days = random.randint(
                C.NEAR_NEGATIVE_MIN_DAYS, C.NEAR_NEGATIVE_MAX_DAYS
            )
            sign = random.choice([-1, 1])
            ts = ts0 + timedelta(days=sign * delta_days)

            if not (pd.Timestamp(C.DATE_START) <= ts <= pd.Timestamp(C.DATE_END)):
                continue

            records.append(
                dict(latitude=lat, longitude=lon, datetime=ts, Wildfire="No")
            )
            no_coords.append([lat, lon])
            no_times.append(ts)
    return pd.DataFrame(records)

#1 Year negatives ───────────────────────────────────────────────────
def build_year_negatives(yes_df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    yes_triplets = set(zip(yes_df.latitude, yes_df.longitude, yes_df.datetime))

    radius_rad_no = C.MIN_NO_SPATIAL_KM / EARTH_KM   
    no_coords, no_times = [], []                   

    for _, row in tqdm(yes_df.iterrows(), total=len(yes_df), desc="year-neg"):
        lat0, lon0, ts0 = row["latitude"], row["longitude"], row["datetime"]


        max_years_back = int((ts0 - pd.Timestamp(C.DATE_START)).days // 365)

        for yr in range(1, max_years_back + 1):
            new_ts = ts0 - timedelta(days=365 * yr)

            if (lat0, lon0, new_ts) in yes_triplets:
                continue
                
            if no_coords:
                tmp_tree = BallTree(np.radians(no_coords), metric="haversine")
                close = tmp_tree.query_radius(
                    np.radians([[lat0, lon0]]), r=radius_rad_no, return_distance=False
                )[0]
                if any(abs((no_times[j] - new_ts).days) < C.MIN_NO_TEMPORAL_DAYS for j in close):
                    continue

            recs.append(
                dict(latitude=lat0, longitude=lon0, datetime=new_ts, Wildfire="No")
            )
            no_coords.append([lat0, lon0])
            no_times.append(new_ts)

    return pd.DataFrame(recs)


#Main pipeline ─────────────────────────────────────────────────────────
def main(incidents_csv: str, out_csv: str):
    raw = pd.read_csv(incidents_csv)

    log.info("Building positive set …")
    yes_df = build_positive(raw)

    tree = make_balltree(yes_df)

    far_df   = build_far_negatives(tree)
    near_df  = build_near_negatives(yes_df, tree)
    year_df  = build_year_negatives(yes_df)


    neg_df = pd.concat([far_df, near_df, year_df], ignore_index=True)

    #remove duplicate rows
    neg_df = neg_df.drop_duplicates(subset=["latitude", "longitude", "datetime"])

    final = (
        pd.concat([yes_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=C.SEED)
        .reset_index(drop=True)
    )
    final.to_csv(out_csv, index=False)

    log.info(
        "Saved %s  –  Yes=%d, NearNo=%d, YearNo=%d, FarNo=%d, Total=%d",
        out_csv,
        len(yes_df),
        len(near_df),
        len(year_df),
        len(far_df),
        len(final),
    )

#CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(C.INCIDENTS, C.OUT)