"""
Test script to demonstrate drop_na parameter functionality.
"""
import geopandas as gpd
import pandas as pd
from ECOv002_calval_tables import load_metadata_ebc_filt
from ECOv002_CMR import sample_points_over_date_range

# Load site metadata
sites_df = load_metadata_ebc_filt()
test_sites = sites_df.iloc[:3]  # Use 3 sites for testing

print("=" * 80)
print("TESTING drop_na PARAMETER")
print("=" * 80)

# Test 1: With drop_na=True (default)
print("\nTest 1: drop_na=True (default) - Removes rows with no valid data")
print("-" * 80)
result_with_drop = sample_points_over_date_range(
    geometry=test_sites,
    start_date="2022-06-01",
    end_date="2022-06-20",
    layers=['ST_C', 'emissivity', 'NDVI', 'albedo', 'Ta_C', 'RH', 'SM'],
    drop_na=True,  # Default behavior
    verbose=True
)
print(f"\nResult: {len(result_with_drop)} observations")
print(f"Unique granules: {result_with_drop['granule'].nunique()}")

# Test 2: With drop_na=False
print("\n" + "=" * 80)
print("Test 2: drop_na=False - Keeps all rows including those with no data")
print("-" * 80)
result_without_drop = sample_points_over_date_range(
    geometry=test_sites,
    start_date="2022-06-01",
    end_date="2022-06-20",
    layers=['ST_C', 'emissivity', 'NDVI', 'albedo', 'Ta_C', 'RH', 'SM'],
    drop_na=False,
    verbose=True
)
print(f"\nResult: {len(result_without_drop)} observations")
print(f"Unique granules: {result_without_drop['granule'].nunique()}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"With drop_na=True:  {len(result_with_drop)} observations (default)")
print(f"With drop_na=False: {len(result_without_drop)} observations")
print(f"Rows filtered:      {len(result_without_drop) - len(result_with_drop)} rows with no valid data")

# Show example of dropped rows
if len(result_without_drop) > len(result_with_drop):
    print("\n" + "=" * 80)
    print("EXAMPLE OF FILTERED ROWS (rows with all NaN in sampled variables)")
    print("=" * 80)
    
    # Find rows that were dropped
    merged = result_without_drop.merge(
        result_with_drop[['granule', 'point_index']], 
        on=['granule', 'point_index'], 
        how='left', 
        indicator=True
    )
    dropped_rows = merged[merged['_merge'] == 'left_only']
    
    if len(dropped_rows) > 0:
        cols = ['timestamp', 'ST_C', 'emissivity', 'NDVI', 'albedo', 'Ta_C', 'RH', 'SM']
        available_cols = [c for c in cols if c in dropped_rows.columns]
        print(dropped_rows[available_cols].head())
