"""
Test script to compare parallel vs sequential execution performance.
"""
import time
import geopandas as gpd
import pandas as pd
from ECOv002_calval_tables import load_metadata_ebc_filt
from ECOv002_CMR import sample_points_over_date_range

# Load site metadata
sites_df = load_metadata_ebc_filt()
test_sites = sites_df.iloc[:3]  # Use 3 sites for testing

print("=" * 80)
print("PERFORMANCE COMPARISON: Sequential vs Parallel Execution")
print("=" * 80)
print(f"\nTest configuration:")
print(f"  - Points: {len(test_sites)}")
print(f"  - Date range: 2022-06-01 to 2022-06-20")
print(f"  - Layers: ST_C, emissivity, NDVI, albedo, Ta_C, RH, SM")

# Test 1: Sequential execution (max_workers=1)
print("\n" + "=" * 80)
print("Test 1: SEQUENTIAL EXECUTION (max_workers=1)")
print("=" * 80)
start_time = time.time()
result_seq = sample_points_over_date_range(
    geometry=test_sites,
    start_date="2022-06-01",
    end_date="2022-06-20",
    layers=['ST_C', 'emissivity', 'NDVI', 'albedo', 'Ta_C', 'RH', 'SM'],
    max_workers=1,
    verbose=False
)
seq_time = time.time() - start_time

print(f"\n  Results: {len(result_seq)} observations")
print(f"  Execution time: {seq_time:.2f} seconds")

# Test 2: Parallel execution (max_workers=10, default)
print("\n" + "=" * 80)
print("Test 2: PARALLEL EXECUTION (max_workers=10, default)")
print("=" * 80)
start_time = time.time()
result_par = sample_points_over_date_range(
    geometry=test_sites,
    start_date="2022-06-01",
    end_date="2022-06-20",
    layers=['ST_C', 'emissivity', 'NDVI', 'albedo', 'Ta_C', 'RH', 'SM'],
    max_workers=10,
    verbose=False
)
par_time = time.time() - start_time

print(f"\n  Results: {len(result_par)} observations")
print(f"  Execution time: {par_time:.2f} seconds")

# Summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Sequential execution: {seq_time:.2f}s")
print(f"Parallel execution:   {par_time:.2f}s")
print(f"Speedup:              {seq_time/par_time:.2f}x")
print(f"Time saved:           {seq_time - par_time:.2f}s ({(1 - par_time/seq_time)*100:.1f}% reduction)")

# Verify results match
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
print(f"Sequential observations: {len(result_seq)}")
print(f"Parallel observations:   {len(result_par)}")
print(f"Results match: {len(result_seq) == len(result_par)}")
