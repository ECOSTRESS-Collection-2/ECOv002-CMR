"""Debug script to see what STARS granules are found."""
import pandas as pd
from ECOv002_CMR import ECOSTRESS_CMR_search

# Search for STARS products in the same date range
tile = "18SUE"
start_date = "2022-06-01"
end_date = "2022-06-10"

print("="*80)
print(f"Searching for L2T_STARS in tile {tile} from {start_date} to {end_date}")
print("="*80)

results = ECOSTRESS_CMR_search(
    product='L2T_STARS',
    tile=tile,
    start_date=start_date,
    end_date=end_date
)

if results.empty:
    print("No STARS granules found!")
else:
    print(f"\nFound {len(results)} total files")
    print(f"\nUnique granules: {results['granule'].nunique()}")
    print("\nGranules found:")
    for granule in results['granule'].unique():
        print(f"  {granule}")
    
    print("\nBreakdown by variable:")
    geotiff_data = results[results['type'] == 'GeoTIFF Data']
    for var in geotiff_data['variable'].unique():
        var_files = geotiff_data[geotiff_data['variable'] == var]
        print(f"  {var}: {len(var_files)} files")
        for granule in var_files['granule'].unique():
            print(f"    - {granule}")

# Also check LST for comparison
print("\n" + "="*80)
print(f"Searching for L2T_LSTE in tile {tile} from {start_date} to {end_date}")
print("="*80)

lst_results = ECOSTRESS_CMR_search(
    product='L2T_LSTE',
    tile=tile,
    start_date=start_date,
    end_date=end_date
)

if lst_results.empty:
    print("No LST granules found!")
else:
    print(f"\nFound {len(lst_results)} total files")
    print(f"\nUnique LST granules: {lst_results['granule'].nunique()}")
    print("\nLST Granules (dates):")
    for granule in sorted(lst_results['granule'].unique()):
        # Extract date from granule name
        parts = granule.split('_')
        if len(parts) > 6:
            timestamp = parts[6]
            date = timestamp[:8]
            print(f"  {granule}")
            print(f"    Date: {date}")
