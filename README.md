# Common Metadata Repository (CMR) Search for ECOSTRESS Collection 2

![CI](https://github.com/ECOSTRESS-Collection-2/ECOv002-CMR/actions/workflows/ci.yml/badge.svg)

The `ECOv002-CMR` Python package is a utility for searching and downloading ECOSTRESS Collection 2 tiled data product granules using the [Common Metadata Repository (CMR) API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html).

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## Pre-Requisites

This package uses [wget](https://www.gnu.org/software/wget/) for file transfers.

On macOS, install [wget](https://formulae.brew.sh/formula/wget) with [Homebrew](https://brew.sh/):

```
brew install wget
```

## Installation

Install the [ECOv002-CMR](https://pypi.org/project/ECOv002-CMR/) package, with a dash in the name, from PyPi using pip:

```
pip install ECOv002-CMR
```

## Authentication

ECOSTRESS data requires NASA Earthdata authentication. Set up your credentials using one of these methods:

**Option 1: Create a `.netrc` file** (recommended)

```bash
cat > ~/.netrc << EOF
machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
EOF
chmod 600 ~/.netrc
```

**Option 2: Environment variables**

```bash
export EARTHDATA_USERNAME=YOUR_USERNAME
export EARTHDATA_PASSWORD=YOUR_PASSWORD
```

Register for a free account at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/) if you don't have one.

## Usage

Import the `ECOv002_CMR` package, with an underscore in the name:

```python
import ECOv002_CMR
from datetime import date
```

### Search for ECOSTRESS Granules

Use `ECOSTRESS_CMR_search()` to query the CMR API and get available granules without downloading:

```python
from ECOv002_CMR import ECOSTRESS_CMR_search

# Search for Land Surface Temperature data
results = ECOSTRESS_CMR_search(
    product="L2T_LSTE",              # Product name
    tile="11SLT",                    # Sentinel-2 tile ID
    start_date=date(2025, 6, 1),     # Start date
    end_date=date(2025, 6, 30)       # End date (optional, defaults to start_date)
)

# Results is a pandas DataFrame with columns:
# - product, variable, orbit, scene, tile
# - type (e.g., 'GeoTIFF Data', 'JSON Metadata')
# - granule, filename, URL

print(f"Found {len(results)} files")
print(results[['variable', 'orbit', 'scene', 'URL']].head())

# Filter for specific variables
lst_files = results[
    (results['type'] == 'GeoTIFF Data') & 
    (results['variable'] == 'LST')
]

# Filter for specific orbit/scene (optional)
specific_granule = ECOSTRESS_CMR_search(
    product="L2T_LSTE",
    tile="11SLT",
    start_date=date(2025, 6, 1),
    orbit=52314,                     # Filter by orbit number
    scene=3                          # Filter by scene number
)
```

### Download an ECOSTRESS Granule

Use `download_ECOSTRESS_granule()` to download all files for a specific granule:

```python
from ECOv002_CMR import download_ECOSTRESS_granule

# Download a granule by date
granule = download_ECOSTRESS_granule(
    product="L2T_LSTE",
    tile="10UEV",
    aquisition_date=date(2023, 8, 1),
    parent_directory="./data"        # Where to save (default: ~/data/ECOSTRESS)
)

# Download specific orbit/scene
granule = download_ECOSTRESS_granule(
    product="L2T_LSTE",
    tile="10UEV",
    aquisition_date=date(2023, 8, 1),
    orbit=52314,
    scene=3
)

# The returned object is an ECOSTRESSGranule instance
print(granule)
```

### Available Products

The package supports these ECOSTRESS Collection 2 products:

| Product Code | Description | Key Variables |
|-------------|-------------|---------------|
| `L2T_LSTE` | Land Surface Temperature & Emissivity | LST, LST_err, QC, EmisWB |
| `L2T_STARS` | Surface Reflectance & Albedo | NDVI, EVI, albedo, reflectance bands |
| `L3T_MET` | Meteorological Data | temperature, pressure, humidity |
| `L3T_SM` | Soil Moisture | soil moisture estimates |
| `L3T_SEB` | Surface Energy Balance | latent heat, sensible heat, net radiation |
| `L3T_JET` | Evapotranspiration Ensemble | ensemble ET products |
| `L4T_ESI` | Evaporative Stress Index | ESI, anomaly |
| `L4T_WUE` | Water Use Efficiency | WUE metrics |

### Finding Your Tile

ECOSTRESS uses the Sentinel-2 MGRS tiling system. Find your tile using coordinates:

```python
from sentinel_tiles import sentinel_tiles
from shapely.geometry import Point

# Find tile for a coordinate (lon, lat)
point = Point(-118.2437, 34.0522)  # Los Angeles
tile = sentinel_tiles.nearest(point)
print(f"Tile: {tile}")  # Output: 11SLT
```

Or browse the [Sentinel-2 tiling grid](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2/data-products).

## Point Sampling

### Sample at Single Point

Extract values at specific coordinates from ECOSTRESS rasters:

```python
from ECOv002_CMR import (
    setup_earthdata_session,
    sample_point_from_url,
    ECOSTRESS_CMR_search
)
from datetime import date

# Set up authenticated session
session = setup_earthdata_session()

# Search for data
results = ECOSTRESS_CMR_search(
    product="L2T_LSTE",
    tile="11SLT",
    start_date=date(2025, 6, 15)
)

# Get URL for LST variable
lst_files = results[
    (results['type'] == 'GeoTIFF Data') & 
    (results['variable'] == 'LST')
]
url = lst_files.iloc[0]['URL']

# Sample at coordinate (lon, lat in WGS84)
value, metadata = sample_point_from_url(
    session=session,
    url=url,
    lon=-118.2437,  # Downtown LA
    lat=34.0522
)

if value is not None:
    print(f"LST: {value:.2f} K ({value - 273.15:.2f}°C)")
else:
    print(f"Sampling failed: {metadata}")
```

### Batch Sampling for All Available Acquisitions (Recommended)

Sample multiple points across all ECOSTRESS acquisitions in a date range. This is the recommended approach since you don't need to know acquisition times in advance:

```python
import geopandas as gpd
from datetime import date
from shapely.geometry import Point
from ECOv002_CMR import sample_points_over_date_range

# Create GeoDataFrame with points (no datetime needed!)
data = {
    'site_id': ['site_A', 'site_B', 'site_C'],
    'geometry': [
        Point(-118.24, 34.05),   # Downtown LA
        Point(-118.41, 33.94),   # LAX Airport
        Point(-118.14, 34.15)    # Pasadena
    ]
}
gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')

# Define products and variables to sample
products = {
    'L2T_LSTE': 'LST',                    # Land Surface Temperature
    'L2T_STARS': ['NDVI', 'albedo']      # NDVI and Albedo
}

# Sample all available acquisitions in June 2025
results = sample_points_over_date_range(
    gdf=gdf,
    products=products,
    start_date=date(2025, 6, 1),
    end_date=date(2025, 6, 30),
    verbose=True
)

# Results DataFrame contains all point-acquisition combinations
# Each row is one observation at one point at one acquisition time
print(results[['site_id', 'timestamp', 'LST', 'NDVI', 'albedo']])

# Convert temperature to Celsius
results['LST_celsius'] = results['LST'] - 273.15

# See how many acquisitions covered each site
print(results.groupby('site_id').size())

# Save results
results.to_csv('ecostress_samples.csv', index=False)
```

The `sample_points_over_date_range` function:
- Searches for all ECOSTRESS granules in the date range
- Automatically determines Sentinel-2 tiles from point coordinates
- Samples all found granules at each point
- Returns one row per point-acquisition combination
- Much more efficient than searching by specific datetimes

### Batch Sampling with Specific Datetimes

If you do know specific acquisition times you want to sample:

```python
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point
from ECOv002_CMR import sample_points_from_geodataframe

# Create GeoDataFrame with points and datetimes
data = {
    'site_id': ['site_A', 'site_B', 'site_C'],
    'datetime': [
        datetime(2025, 6, 15, 12, 0),
        datetime(2025, 6, 20, 14, 0),
        datetime(2025, 6, 25, 10, 0)
    ],
    'geometry': [
        Point(-118.24, 34.05),   # Downtown LA
        Point(-118.41, 33.94),   # LAX Airport
        Point(-118.14, 34.15)    # Pasadena
    ]
}
gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')

# Define products and variables to sample
products = {
    'L2T_LSTE': 'LST',                    # Land Surface Temperature
    'L2T_STARS': ['NDVI', 'albedo']      # NDVI and Albedo
}

# Sample all points
results = sample_points_from_geodataframe(
    gdf=gdf,
    products=products,
    date_buffer_days=3,  # Search ±3 days around each datetime
    verbose=True
)

# Results DataFrame contains sampled values
print(results[['site_id', 'timestamp', 'LST', 'NDVI', 'albedo']])

# Convert temperature to Celsius
results['LST_celsius'] = results['LST'] - 273.15

# Save results
results.to_csv('ecostress_samples.csv', index=False)
```

This approach uses `date_buffer_days` to search for granules near each specified datetime. Useful when you have specific target times but want to allow some flexibility.

See [examples/geodataframe_sampling_demo.py](examples/geodataframe_sampling_demo.py) for the recommended date-range approach.

### Working with Multiple Variables

Search across multiple products and combine results:

```python
from ECOv002_CMR import ECOSTRESS_CMR_search

# Products and their variables
products = {
    "L2T_LSTE": ["LST", "QC"],
    "L2T_STARS": ["NDVI", "albedo"]
}

tile = "11SLT"
start = date(2025, 6, 1)
end = date(2025, 6, 30)

all_files = {}

for product, variables in products.items():
    results = ECOSTRESS_CMR_search(
        product=product,
        tile=tile,
        start_date=start,
        end_date=end
    )
    
    for variable in variables:
        var_files = results[
            (results['type'] == 'GeoTIFF Data') & 
            (results['variable'] == variable)
        ]
        all_files[variable] = var_files
        print(f"{variable}: {len(var_files)} files")
```

### Advanced Point Sampling Options

### Advanced Point Sampling Options

**Custom tile specification:**

```python
# Pre-specify tiles instead of auto-detecting
gdf['tile'] = ['11SLT', '11SLT', '11SLT']

results = sample_points_from_geodataframe(
    gdf=gdf,
    products=products,
    tile_col='tile'  # Use pre-specified tiles
)
```

**Find Sentinel-2 tile for a coordinate:**

```python
from ECOv002_CMR import find_sentinel2_tile

tile = find_sentinel2_tile(lon=-118.24, lat=34.05)
print(f"Tile: {tile}")  # Output: 11SLT
```

**Custom datetime column names:**

```python
results = sample_points_from_geodataframe(
    gdf=gdf,
    products=products,
    geometry_col='location',  # Custom geometry column
    datetime_col='timestamp'  # Custom datetime column
)
```

See [examples/point_sampling_demo.py](examples/point_sampling_demo.py) for single-point sampling workflow.

### Date Formats

Dates can be specified as `date` objects or strings:

```python
from datetime import date
from dateutil import parser

# Using date objects (recommended)
results = ECOSTRESS_CMR_search(
    product="L2T_LSTE",
    tile="11SLT",
    start_date=date(2025, 6, 1),
    end_date=date(2025, 6, 30)
)

# Using date strings
results = ECOSTRESS_CMR_search(
    product="L2T_LSTE",
    tile="11SLT",
    start_date="2025-06-01",
    end_date="2025-06-30"
)
```

### Error Handling

```python
from ECOv002_CMR import ECOSTRESS_CMR_search

try:
    results = ECOSTRESS_CMR_search(
        product="L2T_LSTE",
        tile="11SLT",
        start_date=date(2025, 6, 1)
    )
    
    if results.empty:
        print("No granules found for specified criteria")
    else:
        print(f"Found {len(results)} files")
        
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Search failed: {e}")
```
