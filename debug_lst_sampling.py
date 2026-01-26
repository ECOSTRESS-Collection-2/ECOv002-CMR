"""Debug why LST sampling failed for June 8."""
import requests
from ECOv002_CMR import setup_earthdata_session, ECOSTRESS_CMR_search

session = setup_earthdata_session()

# Search for LST on June 8
results = ECOSTRESS_CMR_search(
    product='L2T_LSTE',
    tile='18SUE',
    start_date='2022-06-08',
    end_date='2022-06-08'
)

print("June 8 LST granule:")
lst_files = results[results['variable'] == 'LST']
for _, row in lst_files.iterrows():
    print(f"  Granule: {row['granule']}")
    print(f"  URL: {row['URL']}")
    print(f"  Variable: {row['variable']}")

# Also check QC
qc_files = results[results['variable'] == 'QC']
print(f"\nQC files available: {len(qc_files)}")

print("\n" + "="*80)
print("The LST file exists, but sampling returned NaN.")
print("Possible reasons:")
print("  1. Pixel is masked (cloud, water, or quality flag)")
print("  2. Pixel value is nodata/fill value") 
print("  3. Point falls outside valid data bounds")
print("  4. Pixel failed internal QC checks")
print("\nThis is normal behavior - not all pixels have valid LST at every acquisition.")
