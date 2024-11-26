from typing import Union
from os.path import join
from datetime import date
import posixpath
import pandas as pd

from ECOv002_granules import ECOSTRESSGranule, open_granule

from .constants import *
from .granule_ID import GranuleID
from .download_file import download_file
from .download_ECOSTRESS_granule_files import download_ECOSTRESS_granule_files

def download_ECOSTRESS_granule(
        product: str, 
        tile: str, 
        aquisition_date: Union[date, str], 
        orbit: int = None,
        scene: int = None,
        parent_directory: str = ".",
        CMR_file_listing_df: pd.DataFrame = None,
        CMR_search_URL: str = CMR_SEARCH_URL) -> ECOSTRESSGranule:
    """
    Downloads an ECOSTRESS granule based on the provided parameters and returns the granule object.

    Parameters:
    - product (str): The product type to download.
    - tile (str): The tile identifier.
    - aquisition_date (Union[date, str]): The date of acquisition.
    - orbit (int, optional): The orbit number. Defaults to None.
    - scene (int, optional): The scene number. Defaults to None.
    - parent_directory (str, optional): The directory to save the downloaded files. Defaults to ".".
    - CMR_file_listing_df (pd.DataFrame, optional): DataFrame containing file listings from CMR. Defaults to None.
    - CMR_search_URL (str, optional): The URL for CMR search. Defaults to CMR_SEARCH_URL.

    Returns:
    - ECOSTRESSGranule: The downloaded ECOSTRESS granule object.
    """
    directory = download_ECOSTRESS_granule_files(
        product=product,
        tile=tile,
        aquisition_date=aquisition_date,
        orbit=orbit,
        scene=scene,
        parent_directory=parent_directory,
        CMR_file_listing_df=CMR_file_listing_df,
        CMR_search_URL=CMR_search_URL
    )

    granule = open_granule(directory)

    return granule