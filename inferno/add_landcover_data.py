'''
Script: add_modis_landcover.py

Reads an existing CSV with contiguous 75-row windows (75-day periods) back-to-back.
For each 75-row block:
 1. Identify the central-date row (#61; zero-based index 60), extract its year and (lat,lon).
 2. Build unique (lat, lon, year) keys for Earth Engine sampling.
 3. Query MODIS MCD12Q1 once per key to get `landcover`.
 4. Re-iterate blocks, append the same `landcover` value to all 75 rows, and write them out.

Usage (in Python code):
    from add_modis_landcover import main
    main(
        input_path="data_75day.csv",
        output_path="data_with_landcover.csv"
    )

Requires: earthengine-api, pandas
'''
import csv
import ee
import pandas as pd
from datetime import datetime
import sys
import argparse
import time

def initialize_earth_engine():
    try:
        creds = ee.ServiceAccountCredentials(
            'ee-shauryamathur2001@inferno-tactics.iam.gserviceaccount.com',
            './ee-shauryamathur2001-7b7c6b1280b5.json'
        )
        ee.Initialize(credentials=creds, project='ee-shauryamathur2001')
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        exit(1)
initialize_earth_engine()

def sample_landcover(lat: float, lon: float, year: int) -> int:
    """
    Sample MODIS MCD12Q1 LC_Type1 for given lat/lon and year, with retry/backoff to handle rate limits.
    Falls back to the latest available image if none for that year.
    Returns integer landcover class or raises an error.
    """
    col = ee.ImageCollection("MODIS/061/MCD12Q1")
    # prepare image (up to target year, fallback)
    img = (col.filter(ee.Filter.calendarRange(0, year, 'year'))
              .sort('system:time_start', False)
              .first())
    latest = col.sort('system:time_start', False).first()
    img = ee.Image(ee.Algorithms.If(img, img, latest))

    # perform sampling with retries
    attempts = 3
    backoff = 1  # initial seconds
    for i in range(attempts):
        try:
            sample = (ee.Image(img)
                      .select('LC_Type1')
                      .sample(region=ee.Geometry.Point([lon, lat]), scale=500, numPixels=1))
            feat = sample.first().getInfo()
            props = feat.get('properties') or {}
            if 'LC_Type1' not in props:
                raise RuntimeError(f"LC_Type1 missing in properties: {props}")
            return int(props['LC_Type1'])
        except Exception as e:
            if i < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise RuntimeError(f"Sample failed after {attempts} attempts at ({lat},{lon},{year}): {e}")


def main(input_path: str, output_path: str, chunksize: int = 100000):
    # --- 1st pass: gather unique (lat,lon,year) keys ---
    unique_keys = set()
    with open(input_path, newline='') as infile:
        reader = csv.DictReader(infile)
        while True:
            block = []
            try:
                for _ in range(75):
                    block.append(next(reader))
            except StopIteration:
                break
            if len(block) < 75:
                break
            central = block[60]
            lat = float(central['latitude'])
            lon = float(central['longitude'])
            year = datetime.fromisoformat(central['datetime']).year
            unique_keys.add((lat, lon, year))

    # --- 2nd: sample MODIS once per key ---
    lc_map = {}
    # count = 0
    for lat, lon, year in unique_keys:
        try:
            lc_map[(lat, lon, year)] = sample_landcover(lat, lon, year)
            # count += 1
            # if count == 2:
            #     break
        except Exception as e:
            print(f"Warning: sampling failed for ({lat},{lon},{year}): {e}")
            lc_map[(lat, lon, year)] = None
    print(lc_map)

    # --- 3rd pass: annotate and write ---
    with open(input_path, newline='') as infile, \
         open(output_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['landcover']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            block = []
            try:
                for _ in range(75):
                    block.append(next(reader))
            except StopIteration:
                break
            if len(block) < 75:
                break
            central = block[60]
            lat = float(central['latitude'])
            lon = float(central['longitude'])
            year = datetime.fromisoformat(central['datetime']).year
            lc = lc_map.get((lat, lon, year))
            for row in block:
                row['landcover'] = lc
                writer.writerow(row)

    print(f"Done: annotated data written to {output_path}")

if __name__ == '__main__':
    main(
        input_path="Wildfire_Dataset.csv",
        output_path="data_with_landcover.csv"
    )