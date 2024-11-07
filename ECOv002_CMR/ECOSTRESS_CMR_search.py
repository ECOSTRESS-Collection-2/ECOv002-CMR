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
from .process_orbit_scene_tile_URLs import process_orbit_scene_tile_URLs
from .process_tile_URLs import process_tile_URLs

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
          product                  variable  orbit  scene   tile            type  \
        0  ECO1BGEO  L2_LSTE_Day_Structure   1234   5678  10UEV  GeoTIFF Data   
        1  ECO1BGEO  L2_LSTE_Day_Structure   1234   5678  10UEV  JSON Metadata   
        ...
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

    # records = []

    # for URL in URLs:
    #     filename = posixpath.basename(URL)
    #     variable = ""
        
    #     # Determine file type and granule name based on filename
    #     if filename.endswith(".json"):
    #         granule_name = filename.split(".")[0]
    #         type = "JSON Metadata"
    #     elif filename.endswith(".tif"):
    #         type = "GeoTIFF Data"
    #         granule_name = "_".join(filename.split("_")[:-1])
    #     elif filename.endswith(".jpeg"):
    #         type = "GeoJPEG Preview"
    #         granule_name = "_".join(filename.split("_")[:-1])
    #     elif filename.endswith(".jpeg.aux.xml"):
    #         type = "GeoJPEG Metadata"
    #         granule_name = "_".join(filename.split("_")[:-2])
    #     else:
    #         raise ValueError(f"Unknown file type for {filename}")

    #     # Extract variable name from filename for data files
    #     if filename.endswith((".tif", ".jpeg", ".jpeg.aux.xml")):
    #         variable = "_".join(filename.split(".")[0].split("_")[9:])

    #     # Parse granule metadata from granule name
    #     try:
    #         product = "_".join(granule_name.split("_")[1:3])
    #         orbit = int(granule_name.split("_")[3])
    #         scene = int(granule_name.split("_")[4])
    #         tile = granule_name.split("_")[5]
    #         # Add extracted information to the records list
    #         records.append({
    #             "product": product,
    #             "variable": variable,
    #             "orbit": orbit, 
    #             "scene": scene, 
    #             "tile": tile, 
    #             "type": type,
    #             "granule": granule_name,
    #             "filename": filename,
    #             "URL": URL
    #         })
    #     except (IndexError, ValueError) as e:
    #         print(e)
    #         print(f"Filename {filename} does not match expected pattern and was skipped.")

    # # Create a pandas DataFrame from the records
    # df = pd.DataFrame(records, columns=[
    #     "product", 
    #     "variable", 
    #     "orbit", 
    #     "scene",
    #     "tile", 
    #     "type", 
    #     "granule", 
    #     "filename", 
    #     "URL"
    # ])

    if product == "L2T_STARS":
        df = process_tile_URLs(URLs)
    else:
        df = process_orbit_scene_tile_URLs(
            URLs=URLs,
            orbit=orbit,
            scene=scene
        )

    return df