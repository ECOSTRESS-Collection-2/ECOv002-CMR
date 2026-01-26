# ECOSTRESS Examples

This directory contains example scripts demonstrating various ways to work with ECOSTRESS Collection 2 data.

## Examples Overview

### 1. Point Sampling Demo (`point_sampling_demo.py`)

Basic demonstration of sampling ECOSTRESS data at a single coordinate across multiple granules and variables.

**Features:**
- Samples LST, NDVI, and albedo at a single point (Downtown LA)
- Searches across a date range (June 2025)
- Demonstrates working with multiple products (L2T_LSTE, L2T_STARS)
- Shows how to handle instantaneous vs daily data

**Usage:**
```bash
python examples/point_sampling_demo.py
```

### 2. GeoDataFrame Sampling Demo (`geodataframe_sampling_demo.py`)

**Recommended approach:** Shows how to sample all available ECOSTRESS acquisitions within a date range at multiple points.

**Features:**
- Provide points WITHOUT needing to know acquisition times
- Searches for all available granules in date range
- Samples all found acquisitions at each point
- Returns one row per point-acquisition combination
- Most efficient for discovering what data is available

**Usage:**
```bash
python examples/geodataframe_sampling_demo.py
```

### 3. CSV to GeoDataFrame Demo (`csv_to_geodataframe_demo.py`)

Complete workflow for loading points from CSV and sampling ECOSTRESS data.

**Features:**
- Reads site locations from CSV file
- Converts lat/lon to GeoDataFrame
- Samples ECOSTRESS variables
- Exports enriched CSV with sampled values

**Usage:**
```bash
python examples/csv_to_geodataframe_demo.py
```

### Authentication Required

ECOSTRESS data is hosted in NASA's protected Earthdata Cloud and requires authentication.

#### Option 1: Using .netrc file (Recommended)

Create or edit `~/.netrc` with your NASA Earthdata credentials:

```bash
cat > ~/.netrc << EOF
machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
EOF

chmod 600 ~/.netrc
```

#### Option 2: Using Environment Variables

```bash
export EARTHDATA_USERNAME=YOUR_USERNAME
export EARTHDATA_PASSWORD=YOUR_PASSWORD
```

#### Get NASA Earthdata Credentials

If you don't have credentials yet:
1. Create a free account at https://urs.earthdata.nasa.gov/users/new
2. Approve the LP DAAC applications

### Running the Demo

```bash
python examples/point_sampling_demo.py
```

The script will:
1. Search CMR for ECOSTRESS L2T_LSTE granules over Los Angeles for June 2025
2. Sample land surface temperature at a point location using HTTP range requests
3. Display results and statistics
4. Save results to CSV

### How It Works

The script uses **Cloud-Optimized GeoTIFFs (COGs)** with **HTTP range requests** to read only the specific pixels needed, transferring only ~few KB per sample instead of downloading entire files (~tens of MB each).

This technique enables efficient point queries and time-series analysis without local storage requirements.
