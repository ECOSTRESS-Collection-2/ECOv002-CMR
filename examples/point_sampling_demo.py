#!/usr/bin/env python3
"""
Demonstration of point sampling from ECOSTRESS Collection 2 data.

This script samples Land Surface Temperature (LSTE) at a specific coordinate
for all available granules that contain that point.

AUTHENTICATION REQUIRED:
NASA Earthdata credentials required. Set up via ~/.netrc or environment variables.
"""

from datetime import date
import os
import sys
import pandas as pd
import requests
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
from sentinel_tiles import sentinel_tiles
from ECOv002_CMR import ECOSTRESS_CMR_search

def setup_earthdata_session():
    """Set up NASA Earthdata authenticated session."""
    netrc_path = Path.home() / ".netrc"
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    
    if not (username and password) and netrc_path.exists():
        try:
            import netrc
            netrc_auth = netrc.netrc(str(netrc_path))
            auth = netrc_auth.authenticators("urs.earthdata.nasa.gov")
            if auth:
                username, _, password = auth
        except:
            pass
    
    if not (username and password):
        print("ERROR: NASA Earthdata authentication required!")
        print("\nOption 1: Create ~/.netrc with:")
        print("    machine urs.earthdata.nasa.gov")
        print("        login YOUR_USERNAME")
        print("        password YOUR_PASSWORD")
        print("\nOption 2: Set environment variables:")
        print("    export EARTHDATA_USERNAME=YOUR_USERNAME")
        print("    export EARTHDATA_PASSWORD=YOUR_PASSWORD")
        sys.exit(1)
    
    session = requests.Session()
    session.auth = (username, password)
    print("✓ NASA Earthdata authentication configured\n")
    return session

def check_point_in_raster(session, url, lon, lat, max_retries=2):
    """
    Check if a point is within a raster's bounds and return value if so.
    Returns (value, metadata) or (None, reason).
    """
    for attempt in range(max_retries):
        try:
            response = session.get(url, allow_redirects=True, timeout=90)
            response.raise_for_status()
            
            with MemoryFile(response.content) as memfile:
                with memfile.open() as src:
                    # Check if point is within bounds
                    bounds = src.bounds
                    if not (bounds.left <= lon <= bounds.right and 
                           bounds.bottom <= lat <= bounds.top):
                        return None, "outside_bounds"
                    
                    # Get pixel coordinates
                    try:
                        py, px = src.index(lon, lat)
                    except:
                        return None, "coordinate_error"
                    
                    # Verify pixel is within image
                    if not (0 <= py < src.height and 0 <= px < src.width):
                        return None, "outside_image"
                    
                    # Read pixel value
                    window = Window(px, py, 1, 1)
                    value = src.read(1, window=window)[0, 0]
                    
                    # Check for nodata
                    nodata = src.nodata
                    if nodata is not None and value == nodata:
                        return None, "nodata"
                    
                    return value, {
                        'bounds': bounds,
                        'crs': str(src.crs),
                        'nodata': nodata
                    }
                    
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                continue
            return None, f"network_error: {str(e)}"
        except Exception as e:
            return None, f"error: {str(e)}"
    
    return None, "failed_after_retries"

def find_sentinel2_tile(lon, lat):
    """Find which Sentinel-2 tile contains the given coordinate."""
    try:
        # Convert lat/lon to MGRS coordinates and extract tile name (first 5 chars)
        # Format: grid zone (2 digits) + latitude band (1 letter) + 100km square (2 letters)
        mgrs = sentinel_tiles.toMGRS(lat, lon)
        tile_name = mgrs[:5]
        return tile_name
    except:
        return None

# Configuration
TARGET_LON = -118.2437  # Downtown LA longitude
TARGET_LAT = 34.0522    # Downtown LA latitude
PRODUCT = "L2T_LSTE"    # Land Surface Temperature & Emissivity
START_DATE = date(2025, 6, 1)
END_DATE = date(2025, 6, 30)

print("=" * 80)
print("ECOSTRESS POINT SAMPLING DEMONSTRATION")
print("=" * 80)
print(f"\nTarget coordinate: {TARGET_LAT:.6f}°N, {TARGET_LON:.6f}°W")
print(f"Product: {PRODUCT}")
print(f"Time period: {START_DATE} to {END_DATE}\n")

# Find Sentinel-2 tile containing the point
print("Finding Sentinel-2 tile containing target coordinate...")
tile = find_sentinel2_tile(TARGET_LON, TARGET_LAT)
if not tile:
    print("ERROR: Could not determine Sentinel-2 tile for coordinate")
    sys.exit(1)
print(f"✓ Sentinel-2 tile: {tile}\n")

# Set up authentication
session = setup_earthdata_session()

# Search CMR for granules in the tile
print(f"Searching NASA CMR for {PRODUCT} granules in tile {tile}...")
search_results = ECOSTRESS_CMR_search(
    product=PRODUCT,
    tile=tile,
    start_date=START_DATE,
    end_date=END_DATE
)

if search_results.empty:
    print(f"No {PRODUCT} granules found for tile {tile} in the specified time period.")
    sys.exit(0)

print(f"Found {len(search_results)} files across multiple granules\n")

# Filter to LST variable only (the main temperature product)
lst_files = search_results[
    (search_results['type'] == 'GeoTIFF Data') & 
    (search_results['variable'] == 'LST')
].copy()

if lst_files.empty:
    print("No LST (Land Surface Temperature) files found.")
    sys.exit(0)

print(f"Found {len(lst_files)} LST files to check\n")
print("Checking which granules contain the target coordinate...")
print("(This samples from each file to determine coverage)\n")

# Sample each LST file, checking if it contains the target point
results = []
granules_checked = set()
granules_with_data = set()

for idx, row in lst_files.iterrows():
    granule = row['granule']
    orbit = row['orbit']
    scene = row['scene']
    url = row['URL']
    
    # Track progress
    if granule not in granules_checked:
        granules_checked.add(granule)
        print(f"  Checking granule {len(granules_checked)}/{len(lst_files)}: {granule}", end="")
    
    # Check if point is in this raster and get value
    value, meta = check_point_in_raster(session, url, TARGET_LON, TARGET_LAT)
    
    if value is not None:
        # Extract timestamp from granule name
        # Format: ECOv002_L2T_LSTE_<orbit>_<scene>_<tile>_<timestamp>_<version>_<build>
        parts = granule.split('_')
        timestamp = parts[6] if len(parts) > 6 else "unknown"
        
        # Convert Kelvin to Celsius if value is in Kelvin range
        value_celsius = value - 273.15 if value > 200 else value
        
        results.append({
            'timestamp': timestamp,
            'granule': granule,
            'orbit': orbit,
            'scene': scene,
            'LST_kelvin': round(value, 2),
            'LST_celsius': round(value_celsius, 2),
            'lon': TARGET_LON,
            'lat': TARGET_LAT
        })
        
        granules_with_data.add(granule)
        print(f" ✓ {value:.2f} K ({value_celsius:.2f}°C)")
    else:
        if meta == "outside_bounds":
            print(f" - (outside swath)")
        elif meta == "nodata":
            print(f" - (no data)")
        else:
            print(f" - ({meta})")

print(f"\n✓ Found {len(results)} observations at target coordinate")
print(f"  ({len(granules_with_data)} granules contain the point)\n")

# Display and save results
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('timestamp')
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(df[['timestamp', 'orbit', 'scene', 'LST_kelvin', 'LST_celsius']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Number of observations: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nLand Surface Temperature:")
    print(f"  Mean: {df['LST_celsius'].mean():.2f}°C ({df['LST_kelvin'].mean():.2f} K)")
    print(f"  Min:  {df['LST_celsius'].min():.2f}°C ({df['LST_kelvin'].min():.2f} K)")
    print(f"  Max:  {df['LST_celsius'].max():.2f}°C ({df['LST_kelvin'].max():.2f} K)")
    print(f"  Std:  {df['LST_celsius'].std():.2f}°C ({df['LST_kelvin'].std():.2f} K)")
    
    # Save to CSV
    output_file = f"ecostress_LST_{tile}_{START_DATE}_{END_DATE}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"Successfully sampled ECOSTRESS LST at {TARGET_LAT:.6f}°N, {TARGET_LON:.6f}°W")
    print(f"for {len(df)} observations in June 2025.")
else:
    print("=" * 80)
    print("NO DATA FOUND")
    print("=" * 80)
    print(f"The target coordinate ({TARGET_LAT:.6f}°N, {TARGET_LON:.6f}°W) is not")
    print(f"covered by any ECOSTRESS granules in tile {tile} during the specified")
    print("time period. This can happen if:")
    print("  1. The satellite swath didn't cover this exact location")
    print("  2. Data was masked as cloud/water/invalid")
    print("  3. No observations were made during this time period")
