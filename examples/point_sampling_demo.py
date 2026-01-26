#!/usr/bin/env python3
"""
Demonstration of point sampling from ECOSTRESS Collection 2 data.

This script samples Land Surface Temperature (LSTE) at a specific coordinate
for all available granules that contain that point.

AUTHENTICATION REQUIRED:
NASA Earthdata credentials required. Set up via ~/.netrc or environment variables.
"""

from datetime import date
import sys
import pandas as pd
from ECOv002_CMR import (
    ECOSTRESS_CMR_search,
    setup_earthdata_session,
    sample_point_from_url,
    find_sentinel2_tile
)

# Configuration
TARGET_LON = -118.2437  # Downtown LA longitude
TARGET_LAT = 34.0522    # Downtown LA latitude
PRODUCTS = {
    "L2T_LSTE": "LST",      # Land Surface Temperature
    "L2T_STARS": ["NDVI", "albedo"]  # NDVI and Albedo from STARS (tiled)
}
START_DATE = date(2025, 6, 1)
END_DATE = date(2025, 6, 30)

print("=" * 80)
print("ECOSTRESS POINT SAMPLING DEMONSTRATION")
print("=" * 80)
print(f"\nTarget coordinate: {TARGET_LAT:.6f}°N, {TARGET_LON:.6f}°W")
print(f"Products: {', '.join(PRODUCTS.keys())}")
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

# Search CMR for all products and collect files by granule
print("Searching NASA CMR for ECOSTRESS granules...")
all_files = {}  # Dictionary to store files by (granule, variable)

for product, variables in PRODUCTS.items():
    if not isinstance(variables, list):
        variables = [variables]
    
    print(f"  Searching {product}...")
    search_results = ECOSTRESS_CMR_search(
        product=product,
        tile=tile,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    if search_results.empty:
        print(f"  ⚠ No {product} granules found")
        continue
    
    # Filter to desired variables
    for variable in variables:
        var_files = search_results[
            (search_results['type'] == 'GeoTIFF Data') & 
            (search_results['variable'] == variable)
        ].copy()
        
        if not var_files.empty:
            print(f"    Found {len(var_files)} {variable} files")
            for idx, row in var_files.iterrows():
                key = (row['granule'], variable)
                all_files[key] = {
                    'url': row['URL'],
                    'orbit': row['orbit'],
                    'scene': row['scene'],
                    'product': product
                }

# Identify unique granules across all products
unique_granules = list(set(granule for granule, _ in all_files.keys()))
print(f"\nFound {len(unique_granules)} unique granules across all products")
print(f"Checking which granules contain the target coordinate...")
print("(This samples from each file to determine coverage)\n")

# Check each granule
results = []
granules_checked = set()

for granule in unique_granules:
    granules_checked.add(granule)
    print(f"  Checking granule {len(granules_checked)}/{len(unique_granules)}: {granule}")
    
    # Collect all variables for this granule
    granule_data = {
        'granule': granule,
        'lon': TARGET_LON,
        'lat': TARGET_LAT
    }
    
    # Extract orbit, scene, timestamp from granule name
    # Different formats for different products:
    # L2T_LSTE: ECOv002_L2T_LSTE_<orbit>_<scene>_<tile>_<timestamp>_<version>_<build>
    # L2T_STARS: ECOv002_L2T_STARS_<tile>_<date>_<version>_<build>
    parts = granule.split('_')
    product = "_".join(parts[1:3])  # Extract product name (e.g., "L2T_LSTE" or "L2T_STARS")
    
    if product == "L2T_STARS":
        # STARS format: no orbit/scene, organized by tile and date
        if len(parts) > 4:
            granule_data['orbit'] = None
            granule_data['scene'] = None
            granule_data['timestamp'] = parts[4]  # Date in YYYYMMDD format
    else:
        # LSTE and other products: have orbit, scene, and timestamp
        if len(parts) > 6:
            granule_data['orbit'] = parts[3]
            granule_data['scene'] = parts[4]
            granule_data['timestamp'] = parts[6]
    
    has_data = False
    
    # Check each variable for this granule
    for variable in ['LST', 'NDVI', 'albedo']:
        key = (granule, variable)
        if key not in all_files:
            continue
        
        file_info = all_files[key]
        value, meta = sample_point_from_url(session, file_info['url'], TARGET_LON, TARGET_LAT)
        
        if value is not None:
            has_data = True
            if variable == 'LST':
                # Convert Kelvin to Celsius if value is in Kelvin range
                value_celsius = value - 273.15 if value > 200 else value
                granule_data['LST_kelvin'] = round(value, 2)
                granule_data['LST_celsius'] = round(value_celsius, 2)
                print(f"    LST: {value:.2f} K ({value_celsius:.2f}°C)")
            elif variable == 'NDVI':
                granule_data['NDVI'] = round(value, 4)
                print(f"    NDVI: {value:.4f}")
            elif variable == 'albedo':
                granule_data['Albedo'] = round(value, 4)
                print(f"    Albedo: {value:.4f}")
        else:
            display_var = 'Albedo' if variable == 'albedo' else variable
            if meta == "outside_bounds":
                print(f"    {display_var}: outside swath")
            elif meta == "nodata":
                print(f"    {display_var}: no data")
            else:
                print(f"    {display_var}: {meta}")
    
    if has_data:
        results.append(granule_data)

print(f"\n✓ Found {len(results)} observations at target coordinate")
print(f"  ({len(results)} granules contain the point)\n")

# Combine LST observations with daily NDVI and Albedo
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('timestamp')

    # Separate instantaneous observations (LST) and daily estimates (NDVI, Albedo)
    lst_df = df[df['timestamp'].str.contains('T')].copy()  # Instantaneous observations
    daily_df = df[~df['timestamp'].str.contains('T')].copy()  # Daily estimates

    # Create a lookup dictionary for NDVI and Albedo by date
    daily_lookup = {}
    for idx, row in daily_df.iterrows():
        date = row['timestamp']
        daily_lookup[date] = {
            'NDVI': row.get('NDVI', None),
            'Albedo': row.get('Albedo', None)
        }

    # Add NDVI and Albedo to each instantaneous observation based on date
    lst_df['date'] = lst_df['timestamp'].str[:8]
    lst_df['NDVI'] = lst_df['date'].map(lambda d: daily_lookup.get(d, {}).get('NDVI', None))
    lst_df['Albedo'] = lst_df['date'].map(lambda d: daily_lookup.get(d, {}).get('Albedo', None))
    
    # Drop the temporary date column
    merged_df = lst_df.drop(columns=['date'])

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Display columns dynamically based on what's available
    display_cols = ['timestamp', 'orbit', 'scene']
    if 'LST_celsius' in merged_df.columns:
        display_cols.extend(['LST_kelvin', 'LST_celsius'])
    if 'NDVI' in merged_df.columns:
        display_cols.append('NDVI')
    if 'Albedo' in merged_df.columns:
        display_cols.append('Albedo')

    print(merged_df[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Number of observations: {len(merged_df)}")
    print(f"Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")

    if 'LST_celsius' in merged_df.columns:
        print(f"\nLand Surface Temperature:")
        print(f"  Mean: {merged_df['LST_celsius'].mean():.2f}°C ({merged_df['LST_kelvin'].mean():.2f} K)")
        print(f"  Min:  {merged_df['LST_celsius'].min():.2f}°C ({merged_df['LST_kelvin'].min():.2f} K)")
        print(f"  Max:  {merged_df['LST_celsius'].max():.2f}°C ({merged_df['LST_kelvin'].max():.2f} K)")
        print(f"  Std:  {merged_df['LST_celsius'].std():.2f}°C ({merged_df['LST_kelvin'].std():.2f} K)")

    if 'NDVI' in merged_df.columns:
        print(f"\nNDVI (Normalized Difference Vegetation Index):")
        print(f"  Mean: {merged_df['NDVI'].mean():.4f}")
        print(f"  Min:  {merged_df['NDVI'].min():.4f}")
        print(f"  Max:  {merged_df['NDVI'].max():.4f}")
        print(f"  Std:  {merged_df['NDVI'].std():.4f}")

    if 'Albedo' in merged_df.columns:
        print(f"\nAlbedo:")
        print(f"  Mean: {merged_df['Albedo'].mean():.4f}")
        print(f"  Min:  {merged_df['Albedo'].min():.4f}")
        print(f"  Max:  {merged_df['Albedo'].max():.4f}")
        print(f"  Std:  {merged_df['Albedo'].std():.4f}")

    # Save to CSV
    output_file = f"ecostress_multivar_{tile}_{START_DATE}_{END_DATE}.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"Successfully sampled ECOSTRESS data at {TARGET_LAT:.6f}°N, {TARGET_LON:.6f}°W")
    print(f"for {len(merged_df)} observations in {START_DATE.strftime('%B %Y')}.")
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
