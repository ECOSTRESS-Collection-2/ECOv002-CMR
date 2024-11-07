from typing import List, Union
import requests
from datetime import date
from dateutil import parser

from sentinel_tiles import sentinel_tiles

from .constants import CMR_SEARCH_URL  # Assuming CMR_SEARCH_URL is defined in constants.py


def ECOSTRESS_CMR_search_links(
        concept_ID: str, 
        tile: str, 
        start_date: Union[date, str], 
        end_date: Union[date, str],
        CMR_search_URL: str = CMR_SEARCH_URL) -> List[str]:
    """
    Searches the CMR API for ECOSTRESS granules matching the given parameters and extracts all relevant URLs.

    This function constructs a query to the NASA Common Metadata Repository (CMR) API to find 
    ECOSTRESS data granules that intersect with a specified Sentinel-2 tile and fall within a 
    given date range. It then extracts the URLs of these granules from the API response.

    Args:
        concept_ID: The concept ID of the ECOSTRESS collection to search (e.g., 'C2082256699-ECOSTRESS').
        tile: The Sentinel-2 tile identifier for the area of interest (e.g., '10UEV').
        start_date: The start date of the search period in YYYY-MM-DD format (e.g., '2023-08-01').
        end_date: The end date of the search period in YYYY-MM-DD format (e.g., '2023-08-15').
        CMR_search_URL: The base URL for the CMR search API. Defaults to the constant 
                        CMR_SEARCH_URL defined in constants.py.

    Returns:
        A list of URLs pointing to the matching ECOSTRESS granules. These URLs may 
        refer to different file types like GeoTIFF (.tif), JSON metadata (.json), 
        auxiliary XML (.aux.xml), and JPEG preview images (.jpeg).

    Raises:
        requests.exceptions.HTTPError: If the CMR API request fails (e.g., due to a 
                                        network error or an invalid request).

    Example:
        >>> links = ECOSTRESS_CMR_search_links(
        ...     concept_ID='C2082256699-ECOSTRESS', 
        ...     tile='10UEV', 
        ...     start_date='2023-08-01', 
        ...     end_date='2023-08-15'
        ... )
        >>> print(links)
    """

    if isinstance(start_date, str):
        start_date = parser.parse(start_date).date()

    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, str):
        end_date = parser.parse(end_date).date()

    # Get the centroid coordinates of the Sentinel-2 tile
    geometry = sentinel_tiles.grid(tile)
    centroid = geometry.centroid_latlon
    lon, lat = centroid.x, centroid.y

    # Construct the CMR API request URL and parameters
    granule_search_URL = f"{CMR_search_URL}granules.json"
    params = {
        "concept_id": concept_ID,
        "bounding_box": f"{lon},{lat},{lon},{lat}",  # Point search using the tile's centroid
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",  # ISO 8601 format for date/time
        "page_size": 2000  # Retrieve up to 2000 granules per page
    }

    # Send the request to the CMR API
    response = requests.get(granule_search_URL, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 Not Found)

    # Extract URLs from the JSON response
    data = response.json()
    URLs = []
    for entry in data.get('feed', {}).get('entry', []):  # Safely navigate the JSON structure
        for link in entry.get('links', []):
            URL = link.get('href')
            # Filter for URLs that start with "https" and end with specific file extensions
            if URL and URL.startswith("https") and URL.endswith((".tif", ".json", ".aux.xml", ".jpeg")):  
                URLs.append(URL)

    return URLs
