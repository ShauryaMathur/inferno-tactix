#!/usr/bin/env python
"""
Automated Elevation and Land Cover Data Extraction and Image Generation
This script automates the process of:
1. Extracting elevation data from Google Earth Engine
2. Extracting land cover data from Google Earth Engine
3. Converting elevation and land cover to image formats
"""

import os
import ee
import time
import rasterio
import numpy as np
from PIL import Image
import argparse
import requests
from zipfile import ZipFile
from io import BytesIO
from typing import Tuple

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

def download_image(image: ee.Image, region, scale, output_dir, prefix: str) -> str:
    if isinstance(region, list):
        region = ee.Geometry.Polygon(region)

    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}"

    download_url = image.getDownloadURL({
        'name': filename,
        'scale': scale,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e13,
        'crs': 'EPSG:4326',
        'region': region
    })

    print(f"Download URL ({prefix}): {download_url}")
    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(download_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {prefix}: {response.status_code}")

    tif_path = None
    with ZipFile(BytesIO(response.content)) as zip_file:
        for file in zip_file.namelist():
            if file.endswith('.tif'):
                zip_file.extract(file, output_dir)
                tif_path = os.path.join(output_dir, file)
                break

    if not tif_path:
        raise Exception(f"No TIF file found in {prefix} ZIP")

    print(f"{prefix.capitalize()} GeoTIFF downloaded to: {tif_path}")
    return tif_path

def get_elevation_data(region, scale=30, output_dir="."):
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(ee.Geometry.Polygon(region))
    return download_image(elevation, region, scale, output_dir, 'elevation')

def get_landcover_data(date_str,region, scale=500, output_dir="."):
    # Parse the date
    target_date = ee.Date(date_str)
    
    # Pull the annual land-cover collection (current 061 version)
    col = ee.ImageCollection("MODIS/061/MCD12Q1") \
             .filterDate("2000-01-01", date_str) \
             .sort("system:time_start", False)
             
    # Grab the first (i.e. latest) image ≤ date_str
    img = ee.Image(col.first())
    
    # Defensive: if there's no image, fall back to the very latest available
    # (this is unlikely after 2001, but just in case)
    latest = ee.ImageCollection("MODIS/061/MCD12Q1") \
                 .sort("system:time_start", False) \
                 .first()
    img = ee.Image(ee.Algorithms.If(img, img, latest))
    
    # Select the LC_Type1 band and clip
    landcover = img.select("LC_Type1") \
                   .clip(ee.Geometry.Polygon(region))
    
    return download_image(
        landcover, region, scale, output_dir, f"landcover_{date_str}"
    )

def convert_to_heightmap(tif_path, output_width=1200, output_height=800, output_dir=".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"heightmap_{output_width}x{output_height}_2.png")

    with rasterio.open(tif_path) as dataset:
        data = dataset.read(1).astype(np.float32)

    min_val, max_val = data.min(), data.max()
    if max_val == min_val:
        data16 = np.zeros_like(data, dtype=np.uint16)
    else:
        data16 = ((data - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

    h, w = data16.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (data16 >> 8) & 0xFF
    rgba[..., 1] = data16 & 0xFF
    rgba[..., 2] = 0
    rgba[..., 3] = 255

    img = Image.fromarray(rgba, mode='RGBA')
    img = img.resize((output_width, output_height), Image.LANCZOS)
    img.save(output_path)

    print(f"Heightmap generated at: {output_path}")
    return output_path

def convert_landcover_to_image(tif_path, output_width=1200, output_height=800, output_dir=".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"landcover_{output_width}x{output_height}.png")

    landcover_palette = [
        (0, 100, 0),       # darkgreen
        (34, 139, 34),     # forestgreen
        (50, 205, 50),     # limegreen
        (0, 128, 0),       # green
        (60, 179, 113),    # mediumseagreen
        (240, 230, 140),   # khaki
        (218, 165, 32),    # goldenrod
        (128, 128, 0),     # olive
        (154, 205, 50),    # yellowgreen
        (144, 238, 144),   # lightgreen
        (143, 188, 143),   # darkseagreen
        (210, 180, 140),   # tan
        (128, 128, 128),   # gray
        (222, 184, 135),   # burlywood
        (255, 255, 255),   # white
        (211, 211, 211),   # lightgray
        (0, 0, 255)        # blue
    ]

    with rasterio.open(tif_path) as dataset:
        data = dataset.read(1)

    img_rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(landcover_palette, start=1):
        img_rgb[data == i] = color

    img = Image.fromarray(img_rgb, mode='RGB')
    img = img.resize((output_width, output_height), Image.NEAREST)
    img.save(output_path)

    print(f"Land cover image generated at: {output_path}")
    return output_path

def generate_data_and_heightmap(coords: str,date) -> Tuple[str, str, str]:
    width_km = 36.576
    height_km = 24.384
    scale_elev = 152.4
    scale_land = 500
    output_width, output_height = 1200, 813
    output_dir = '../src/public/data'

    lon, lat = map(float, coords.split(','))
    lat_deg_km = 111.0
    lon_deg_km = 111.0 * np.cos(np.radians(lat))
    w_deg = width_km / lon_deg_km
    h_deg = height_km / lat_deg_km

    region = [
        [lon - w_deg / 2, lat - h_deg / 2],
        [lon + w_deg / 2, lat - h_deg / 2],
        [lon + w_deg / 2, lat + h_deg / 2],
        [lon - w_deg / 2, lat + h_deg / 2]
    ]
    
    print(region)
    try:
        initialize_earth_engine()
        elev_tif = get_elevation_data(region, scale_elev, output_dir)
        land_tif = get_landcover_data(date,region, scale_land, output_dir)
        heightmap = convert_to_heightmap(elev_tif, output_width, output_height, output_dir)
        landcover_img = convert_landcover_to_image(land_tif, output_width, output_height, output_dir)
    except Exception as e:
        print(f"Error generating images: {e}")
        exit(1)

    return heightmap, land_tif, landcover_img

if __name__ == "__main__":
    generate_data_and_heightmap("-118.54453,34.07022")
