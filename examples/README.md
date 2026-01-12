# ECOSTRESS Examples

## Point Sampling Demo

The `point_sampling_demo.py` script demonstrates efficient point sampling from ECOSTRESS Collection 2 Cloud-Optimized GeoTIFFs without downloading entire files.

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
