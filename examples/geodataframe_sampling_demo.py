#!/usr/bin/env python3
"""
Demonstration of batch point sampling for all available acquisitions in a date range.

This script shows how to sample ECOSTRESS data for multiple points across all
available acquisitions within a time period. This is the recommended approach
since you don't need to know ECOSTRESS acquisition times in advance.

AUTHENTICATION REQUIRED:
NASA Earthdata credentials required. Set up via ~/.netrc or environment variables.
"""

import geopandas as gpd
from datetime import date
from shapely.geometry import Point
from ECOv002_CMR import sample_points_over_date_range

# Create sample points across different locations in Los Angeles area
# Note: No datetime column needed - we just specify the date range
data = {
    'site_id': ['downtown', 'airport', 'pasadena', 'long_beach'],
    'site_name': ['Downtown LA', 'LAX Airport', 'Pasadena', 'Long Beach'],
    'geometry': [
        Point(-118.2437, 34.0522),   # Downtown LA
        Point(-118.4085, 33.9416),   # LAX
        Point(-118.1445, 34.1478),   # Pasadena
        Point(-118.1937, 33.7701),   # Long Beach
    ]
}

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')

print("=" * 80)
print("ECOSTRESS POINT SAMPLING - ALL AVAILABLE ACQUISITIONS")
print("=" * 80)
print(f"\nSampling {len(gdf)} points:\n")
print(gdf[['site_id', 'site_name']].to_string(index=False))

# Define products and variables to sample
products = {
    'L2T_LSTE': 'LST',                    # Land Surface Temperature
    'L2T_STARS': ['NDVI', 'albedo']      # NDVI and Albedo
}

# Define date range
start_date = date(2025, 6, 1)
end_date = date(2025, 6, 30)

print(f"\nSearching for all ECOSTRESS acquisitions from {start_date} to {end_date}")
print("=" * 80)

# Sample all points for all available acquisitions in the date range
results = sample_points_over_date_range(
    gdf=gdf,
    products=products,
    start_date=start_date,
    end_date=end_date,
    verbose=True
)

if not results.empty:
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Add temperature conversion if LST is present
    if 'LST' in results.columns:
        results['LST_celsius'] = results['LST'] - 273.15
    
    # Display results
    display_cols = ['site_id', 'site_name', 'timestamp']
    if 'LST_celsius' in results.columns:
        display_cols.append('LST_celsius')
    if 'NDVI' in results.columns:
        display_cols.append('NDVI')
    if 'albedo' in results.columns:
        display_cols.append('albedo')
    
    print(results[display_cols].to_string(index=False))
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total observations: {len(results)}")
    print(f"Unique sites sampled: {results['site_id'].nunique()}")
    print(f"Unique acquisition times: {results['timestamp'].nunique()}")
    
    # Show observations per site
    print(f"\nObservations per site:")
    site_counts = results.groupby('site_id').size().sort_values(ascending=False)
    for site, count in site_counts.items():
        print(f"  {site}: {count}")
    
    if 'LST_celsius' in results.columns:
        print(f"\nLand Surface Temperature:")
        print(f"  Mean: {results['LST_celsius'].mean():.2f}°C")
        print(f"  Min:  {results['LST_celsius'].min():.2f}°C")
        print(f"  Max:  {results['LST_celsius'].max():.2f}°C")
        print(f"  Std:  {results['LST_celsius'].std():.2f}°C")
    
    if 'NDVI' in results.columns:
        ndvi_data = results['NDVI'].dropna()
        if not ndvi_data.empty:
            print(f"\nNDVI:")
            print(f"  Mean: {ndvi_data.mean():.4f}")
            print(f"  Min:  {ndvi_data.min():.4f}")
            print(f"  Max:  {ndvi_data.max():.4f}")
    
    if 'albedo' in results.columns:
        albedo_data = results['albedo'].dropna()
        if not albedo_data.empty:
            print(f"\nAlbedo:")
            print(f"  Mean: {albedo_data.mean():.4f}")
            print(f"  Min:  {albedo_data.min():.4f}")
            print(f"  Max:  {albedo_data.max():.4f}")
    
    # Save results
    output_file = "ecostress_batch_sampling_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)
    print(f"Successfully sampled all ECOSTRESS acquisitions for {len(gdf)} input points")
    print(f"Found {len(results)} total observations across {results['timestamp'].nunique()} acquisition times")
    
else:
    print("\n" + "=" * 80)
    print("NO DATA FOUND")
    print("=" * 80)
    print("No ECOSTRESS data found for the specified points and time period.")
    print("Try adjusting:")
    print("  - Date range (check data availability at earthdata.nasa.gov)")
    print("  - Point locations (ensure they're in satellite coverage area)")
