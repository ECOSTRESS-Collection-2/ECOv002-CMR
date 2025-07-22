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

## Usage

Import the `ECOv002_CMR` package, with an underscore in the name:

```
import ECOv002_CMR
```

### Download an ECOSTRESS Granule

Use the `download_ECOSTRESS_granule` function to download a granule by specifying the product, tile, and acquisition date (optionally orbit and scene):

```python
from ECOv002_CMR import download_ECOSTRESS_granule
from datetime import date

# Example: Download an ECOSTRESS granule
granule = download_ECOSTRESS_granule(
    product="ECO2LSTE.002",
    tile="10UEV",
    aquisition_date=date(2023, 8, 1)
)

# The returned object is an ECOSTRESSGranule instance
print(granule)
```
