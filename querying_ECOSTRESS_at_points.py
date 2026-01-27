import geopandas as gpd
import pandas as pd
from ECOv002_calval_tables import load_calval_table, load_metadata_ebc_filt
from ECOv002_CMR import sample_points_over_date_range

# Load site metadata
sites_df = load_metadata_ebc_filt()
print(f"Loaded {len(sites_df)} sites")
print(f"Site columns: {sites_df.columns.tolist()}\n")

# Test: Sample a single site using a GeoDataFrame row
# This will propagate all metadata (Site ID, Name, elevation, etc.) to the output
print("=" * 60)
print("Testing: Single site using gdf.iloc[0]")
print("=" * 60)
result = sample_points_over_date_range(
    geometry=sites_df.iloc[:3],  # Single row with all metadata
    start_date="2022-06-01",
    end_date="2022-06-20",
    layers=['ST_C', 'emissivity', 'NDVI', 'albedo', 'Ta_C', 'RH', 'SM']  # Added emissivity, air temp, humidity, soil moisture
)

print(f"\nResult shape: {result.shape}")
print(f"Available columns: {result.columns.tolist()}")
print(f"\n" + "="*80)
print("RESULTS: Multi-product point sampling")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 4)

# Display all columns
result_display = result.copy()
if 'timestamp' in result_display.columns:
    result_display['date'] = result_display['timestamp'].str[:8]
    # Reorder to put timestamp and date first
    cols = ['timestamp', 'date'] + [c for c in result_display.columns if c not in ['timestamp', 'date']]
    result_display = result_display[cols]
print(result_display.to_string(index=False))

print("\n" + "="*80)
print("DATA AVAILABILITY ANALYSIS")
print("="*80)
print(f"LST (L2T_LSTE) observations: {result['ST_C'].notna().sum()}")
print(f"Emissivity (L2T_LSTE): {result['emissivity'].notna().sum()} ({result['emissivity'].notna().sum()/len(result)*100:.1f}%)")
print(f"STARS (L2T_STARS) coverage: {result['NDVI'].notna().sum()} ({result['NDVI'].notna().sum()/len(result)*100:.1f}%)")
print(f"Air temp (L3T_MET): {result['Ta_C'].notna().sum()} ({result['Ta_C'].notna().sum()/len(result)*100:.1f}%)")
print(f"Humidity (L3T_MET): {result['RH'].notna().sum()} ({result['RH'].notna().sum()/len(result)*100:.1f}%)")
print(f"Soil moisture (L3T_SM): {result['SM'].notna().sum()} ({result['SM'].notna().sum()/len(result)*100:.1f}%)")
print(f"\nNote: Different products have different temporal resolutions and coverage.")
print(f"L3T products (MET, SM) may have sparser temporal sampling than L2T products.")
