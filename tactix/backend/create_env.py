# def generate_heightmap():
#     """Main function to run the script."""
#     parser = argparse.ArgumentParser(description='Extract elevation data and generate a heightmap.')
#     parser.add_argument('--center', type=str, default='-122.2,37.75',
#                         help='Center coordinates as longitude,latitude (default: -122.2,37.75)')
#     parser.add_argument('--width-km', type=float, default=20.0,
#                         help='Width of region in kilometers (default: 20.0)')
#     parser.add_argument('--height-km', type=float, default=10.0,
#                         help='Height of region in kilometers (default: 10.0)')
#     parser.add_argument('--scale', type=int, default=30,
#                         help='Resolution in meters (default: 30)')
#     parser.add_argument('--output-width', type=int, default=1200,
#                         help='Output heightmap width in pixels (default: 1200)')
#     parser.add_argument('--output-height', type=int, default=800,
#                         help='Output heightmap height in pixels (default: 800)')
#     parser.add_argument('--output-dir', type=str, default='output',
#                         help='Output directory (default: "output")')
    
#     args = parser.parse_args()
    
#     # Parse center coordinates
#     center_coords = list(map(float, args.center.split(',')))
#     if len(center_coords) != 2:
#         print("Error: Center should be specified as longitude,latitude")
#         exit(1)
    
#     # Convert width and height from km to degrees
#     # Approximate conversion: 1 degree of latitude = 111 km
#     # Longitude degrees vary by latitude, rough approximation
#     lat_degree_km = 111.0
#     lon_degree_km = 111.0 * np.cos(np.radians(center_coords[1]))
    
#     width_degrees = args.width_km / lon_degree_km
#     height_degrees = args.height_km / lat_degree_km
    
#     # Calculate region corners
#     min_lon = center_coords[0] - width_degrees / 2
#     max_lon = center_coords[0] + width_degrees / 2
#     min_lat = center_coords[1] - height_degrees / 2
#     max_lat = center_coords[1] + height_degrees / 2
    
#     # Convert to GEE format: [[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2]]
#     region = [
#         [min_lon, min_lat],  # bottom-left
#         [max_lon, min_lat],  # bottom-right
#         [max_lon, max_lat],  # top-right
#         [min_lon, max_lat]   # top-left
#     ]
    
#     # Initialize Earth Engine
#     initialize_earth_engine()
    
#     # Get elevation data
#     tif_path = get_elevation_data(region, args.scale, args.output_dir)
    
#     # Convert to heightmap
#     heightmap_path = convert_to_heightmap(tif_path, args.output_width, args.output_height, args.output_dir)
    
#     print("\nProcess completed successfully!")
#     print(f"Elevation data: {tif_path}")
#     print(f"Heightmap: {heightmap_path}")#!/usr/bin/env python
"""
Automated Elevation Data Extraction and Heightmap Generation
This script automates the process of:
1. Extracting elevation data from Google Earth Engine
2. Converting it to a heightmap image
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

def initialize_earth_engine():
    """Initialize the Earth Engine API."""
    try:
        creds = ee.ServiceAccountCredentials(
        'ee-shauryamathur2001@inferno-tactics.iam.gserviceaccount.com',
        './ee-shauryamathur2001-7b7c6b1280b5.json'
        )
        print(creds)
        ee.Initialize(credentials=creds, project='ee-shauryamathur2001')
        # ee.Initialize()
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Make sure you have authenticated with Earth Engine.")
        print("Run 'earthengine authenticate' if needed.")
        exit(1)

def get_elevation_data(region, scale=30, output_dir="."):
    """
    Extract elevation data from Google Earth Engine.
    
    Args:
        region: List of coordinates defining the region [[lon1, lat1], [lon2, lat2], ...].
                Or a GEE Geometry object.
        scale: Resolution in meters.
        output_dir: Directory to save the downloaded data.
        
    Returns:
        Path to the downloaded elevation TIF file.
    """
    # Convert region list to GEE Geometry if it's a list
    if isinstance(region, list):
        region = ee.Geometry.Polygon(region)
    
    # Get SRTM elevation data
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
    
    # Create a filename with timestamp
    timestamp = int(time.time())
    filename = f"elevation_data_{timestamp}"
    
    # Get the download URL (direct download approach)
    download_url = elevation.getDownloadURL({
        'name': filename,
        'scale': scale,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e13,
        'crs': 'EPSG:4326',
        'region': region
    })
    
    print(f"Download URL: {download_url}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download the file
    response = requests.get(download_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download data: {response.status_code}")
    
    # Extract the zip file
    tif_path = None
    with ZipFile(BytesIO(response.content)) as zip_file:
        for file in zip_file.namelist():
            if file.endswith('.tif'):
                zip_file.extract(file, output_dir)
                tif_path = os.path.join(output_dir, file)
                break
    
    if not tif_path:
        raise Exception("No TIF file found in the downloaded ZIP")
    
    print(f"Elevation data downloaded to: {tif_path}")
    return tif_path

def convert_to_heightmap(tif_path, output_width=1200, output_height=800, output_dir="."):
    """
    Convert the TIF file to a 16-bit RGBA heightmap PNG (R=high byte, G=low byte).
    Args:
        tif_path: Path to the elevation TIF file.
        output_width: Width of the output heightmap.
        output_height: Height of the output heightmap.
        output_dir: Directory to save the heightmap.
    Returns:
        Path to the generated heightmap PNG.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build output filename
    output_path = os.path.join(
        output_dir,
        f"heightmap_{output_width}x{output_height}_2.png"
    )

    # Read elevation from GeoTIFF
    with rasterio.open(tif_path) as dataset:
        data = dataset.read(1).astype(np.float32)

    # Compute min/max for normalization
    min_val = data.min()
    max_val = data.max()

    # Normalize to full uint16 range
    if max_val == min_val:
        data16 = np.zeros_like(data, dtype=np.uint16)
    else:
        data16 = ((data - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

    # Pack into RGBA: R=high byte, G=low byte, B=0, A=255
    h, w = data16.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (data16 >> 8) & 0xFF  # High byte
    rgba[..., 1] = data16 & 0xFF         # Low byte
    rgba[..., 2] = 0                     # Unused
    rgba[..., 3] = 255                   # Opaque

    # Create image and resize
    img = Image.fromarray(rgba, mode='RGBA')
    img = img.resize((output_width, output_height), Image.LANCZOS)
    img.save(output_path)

    print(f"Heightmap generated at: {output_path}")
    return output_path

def generate_heightmap(coords: str) -> str:
    """
    coords: "lon,lat" string
    returns path to generated PNG heightmap
    """
    # === PARAMETERS YOU CARE ABOUT ===
    width_km      = 36.576   # modelWidth 120000 ft ≈ 36.576 km
    height_km     = 24.384   # modelHeight 80000 ft ≈ 24.384 km
    # scale         = 30       # meters per pixel
    scale         = 152.4       # meters per pixel
    output_width  = 240     # px
    output_height = 160      # px (keeps aspect ratio: 24.384/36.576*1200≈800)
    output_dir    = '../src/public/data'
    # =================================

    # 1) parse center lon/lat
    lon_str, lat_str = coords.split(',')
    lon, lat = float(lon_str), float(lat_str)

    # 2) km → degrees
    lat_deg_km = 111.0
    lon_deg_km = 111.0 * np.cos(np.radians(lat))
    w_deg = width_km  / lon_deg_km
    h_deg = height_km / lat_deg_km

    # 3) build region polygon
    region = [
        [lon - w_deg/2, lat - h_deg/2],
        [lon + w_deg/2, lat - h_deg/2],
        [lon + w_deg/2, lat + h_deg/2],
        [lon - w_deg/2, lat + h_deg/2]
    ]

    # 4) run EE → GeoTIFF → PNG
    initialize_earth_engine()
    tif = get_elevation_data(region, scale, output_dir)
    try:
        png = convert_to_heightmap(tif, output_width, output_height, output_dir)
    except Exception as e:
        print(f"Error generating heightmap: {e}")
        exit(1)

    return png

if __name__ == "__main__":
    generate_heightmap()