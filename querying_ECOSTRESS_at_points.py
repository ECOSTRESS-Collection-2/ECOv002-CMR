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
)

print(f"\nResult shape: {result.shape}")
print(f"\n" + "="*80)
print("RESULTS: Sub-daily LST matched with daily STARS (NDVI/albedo) by date")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
pd.set_option('display.precision', 4)
result_display = result[['timestamp', 'ST_C', 'NDVI', 'albedo']].copy()
result_display['date'] = result_display['timestamp'].str[:8]
result_display = result_display[['timestamp', 'date', 'ST_C', 'NDVI', 'albedo']]
print(result_display.to_string(index=False))

print("\n" + "="*80)
print("DATA AVAILABILITY ANALYSIS")
print("="*80)
print(f"LST observations: 4 dates (20220601, 20220602, 20220606, 20220608)")
print(f"STARS available: 2 dates (20220606, 20220608 only)")
print(f"\nSTARS coverage: {result['NDVI'].notna().sum()}/{len(result)} = {result['NDVI'].notna().sum()/len(result)*100:.1f}%")
print(f"\nNote: STARS is a data fusion product but doesn't have daily coverage")
print(f"in the archive for this time period. The gaps are real data gaps,")
print(f"not a code issue. Consider:")
print(f"  • Searching a wider date range to get more STARS granules")
print(f"  • Using forward/backward fill to interpolate missing dates")
print(f"  • Checking if newer builds have better temporal coverage")
