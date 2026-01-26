#!/usr/bin/env python3
"""
Example: Loading points from CSV and sampling ECOSTRESS data.

This shows how to read point locations from a CSV file, convert to
GeoDataFrame, and sample all available ECOSTRESS acquisitions in a date range.

CSV format should include columns for:
- latitude, longitude (or x, y)
- Any other metadata you want to preserve (site names, IDs, etc.)

No datetime column needed - you just specify the date range to search!
"""

import pandas as pd
import geopandas as gpd
from datetime import date
from shapely.geometry import Point
from ECOv002_CMR import sample_points_over_date_range

# Example: Create sample CSV
sample_data = """site_id,site_name,latitude,longitude
site_001,Downtown LA,34.0522,-118.2437
site_002,LAX Airport,33.9416,-118.4085
site_003,Pasadena,34.1478,-118.1445
site_004,Long Beach,33.7701,-118.1937
"""

# Save sample CSV (in practice, you'd load your own file)
with open('sample_points.csv', 'w') as f:
    f.write(sample_data)

print("=" * 80)
print("ECOSTRESS SAMPLING FROM CSV - ALL AVAILABLE ACQUISITIONS")
print("=" * 80)

# Read CSV
df = pd.read_csv('sample_points.csv')
print("\nLoaded CSV data:")
print(df.to_string(index=False))

# Create Point geometries from lat/lon columns
df['geometry'] = df.apply(
    lambda row: Point(row['longitude'], row['latitude']), 
    axis=1
)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, crs='EPSG:4326')

# Drop the separate lat/lon columns (optional, geometry column has this info)
gdf = gdf.drop(columns=['latitude', 'longitude'])

print("\nConverted to GeoDataFrame:")
print(gdf.head())

# Define products to sample
products = {
    'L2T_LSTE': 'LST',              # Land Surface Temperature
    'L2T_STARS': ['NDVI', 'albedo'] # NDVI and Albedo
}

# Define date range to search
start_date = date(2025, 6, 1)
end_date = date(2025, 6, 30)

print("\n" + "=" * 80)
print("SAMPLING ALL ECOSTRESS ACQUISITIONS")
print("=" * 80)
print(f"Date range: {start_date} to {end_date}")

# Sample all available ECOSTRESS acquisitions in the date range
results = sample_points_over_date_range(
    gdf=gdf,
    products=products,
    start_date=start_date,
    end_date=end_date,
    verbose=True
)

if not results.empty:
    # Add temperature in Celsius
    if 'LST' in results.columns:
        results['LST_celsius'] = results['LST'] - 273.15
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Display key columns
    display_cols = ['site_id', 'site_name', 'timestamp']
    if 'LST_celsius' in results.columns:
        display_cols.append('LST_celsius')
    if 'NDVI' in results.columns:
        display_cols.append('NDVI')
    if 'albedo' in results.columns:
        display_cols.append('albedo')
    
    print(results[display_cols].to_string(index=False))
    
    # Save enriched results back to CSV
    output_file = 'ecostress_samples.csv'
    results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Input sites: {len(gdf)}")
    print(f"Total observations: {len(results)}")
    print(f"Sites with data: {results['site_id'].nunique()}")
    print(f"Unique acquisition times: {results['timestamp'].nunique()}")
    
    # Show observations per site
    print(f"\nObservations per site:")
    site_counts = results.groupby('site_id').size().sort_values(ascending=False)
    for site, count in site_counts.items():
        print(f"  {site}: {count}")
    
    if 'LST_celsius' in results.columns:
        print(f"\nTemperature range: {results['LST_celsius'].min():.1f}°C to {results['LST_celsius'].max():.1f}°C")
    
else:
    print("\n⚠ No ECOSTRESS data found for these points and dates")
    print("Try adjusting the date range or checking data availability")

# Cleanup sample file
import os
os.remove('sample_points.csv')

