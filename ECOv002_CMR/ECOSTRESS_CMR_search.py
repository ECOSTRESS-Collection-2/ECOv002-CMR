from typing import Union
import requests
import pandas as pd
import json
import posixpath
from datetime import date
from dateutil import parser

from sentinel_tiles import sentinel_tiles

from .constants import *
from .ECOSTRESS_CMR_search_links import ECOSTRESS_CMR_search_links
from .interpret_ECOSTRESS_URLs import interpret_ECOSTRESS_URLs

def ECOSTRESS_CMR_search(
        product: str, 
        tile: str, 
        start_date: Union[date, str], 
        end_date: Union[date, str] = None,
        orbit: int = None,
        scene: int = None,
        CMR_search_URL: str = CMR_SEARCH_URL) -> pd.DataFrame:
    """
    Searches the CMR API for ECOSTRESS granules and constructs a DataFrame with granule information.

    This function utilizes the `ECOSTRESS_CMR_search_links` function to retrieve URLs of 
    ECOSTRESS granules from the CMR API. It then parses these URLs and extracts relevant 
    information like product type, variable, orbit, scene, tile, file type, granule name, 
    and filename to construct a pandas DataFrame.

    Args:
        concept_ID: The concept ID of the ECOSTRESS collection to search (e.g., 'C2082256699-ECOSTRESS').
        tile: The Sentinel-2 tile identifier for the area of interest (e.g., '10UEV').
        start_date: The start date of the search period in YYYY-MM-DD format (e.g., '2023-08-01').
        end_date: The end date of the search period in YYYY-MM-DD format (e.g., '2023-08-15').
        CMR_search_URL: The base URL for the CMR search API. Defaults to the constant 
                        CMR_SEARCH_URL defined in constants.py.

    Returns:
        A pandas DataFrame containing information about the ECOSTRESS granules. The DataFrame has 
        the following columns:

            - product: The ECOSTRESS product type (e.g., 'ECO1BGEO').
            - variable: The measured variable (e.g., 'L2_LSTE_Day_Structure').
            - orbit: The orbit number of the granule.
            - scene: The scene number of the granule.
            - tile: The Sentinel-2 tile identifier.
            - type: The file type (e.g., 'GeoTIFF Data', 'JSON Metadata').
            - granule: The full granule name.
            - filename: The filename of the granule.
            - URL: The URL of the granule.

    Raises:
        ValueError: If an unknown file type is encountered in the URLs.

    Example:
        >>> df = ECOSTRESS_CMR_search(
        ...     concept_ID='C2082256699-ECOSTRESS', 
        ...     tile='10UEV', 
        ...     start_date='2023-08-01', 
        ...     end_date='2023-08-15'
        ... )
        >>> print(df)
    """
    # Convert start_date and end_date to date objects if they are strings
    if isinstance(start_date, str):
        start_date = parser.parse(start_date).date()

    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, str):
        end_date = parser.parse(end_date).date()

    if product not in CONCEPT_IDS:
        raise ValueError(f"Unknown product type: {product}")
    
    concept_ID = CONCEPT_IDS[product]

    # Get the URLs of ECOSTRESS granules using the helper function
    URLs = ECOSTRESS_CMR_search_links(
        concept_ID=concept_ID, 
        tile=tile, 
        start_date=start_date.strftime("%Y-%m-%d"), 
        end_date=end_date.strftime("%Y-%m-%d"), 
        CMR_search_URL=CMR_search_URL
    )

    df = interpret_ECOSTRESS_URLs(
        URLs=URLs,
        orbit=orbit,
        scene=scene
    )

    return df