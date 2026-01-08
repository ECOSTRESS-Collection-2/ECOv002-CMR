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

### Point Sampling Example

Extract values at specific coordinates:

```python
import requests
import rasterio
from rasterio.io import MemoryFile

# Set up authenticated session
session = requests.Session()
session.auth = (username, password)  # From .netrc or env vars

# Get URL from search results
url = results.iloc[0]['URL']

# Download and read raster
response = session.get(url, allow_redirects=True)
with MemoryFile(response.content) as memfile:
    with memfile.open() as src:
        # Sample at coordinate
        lon, lat = -118.2437, 34.0522
        for val in src.sample([(lon, lat)]):
            print(f"Value: {val[0]}")
```

See [examples/point_sampling_demo.py](examples/point_sampling_demo.py) for a complete point sampling workflow.

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
