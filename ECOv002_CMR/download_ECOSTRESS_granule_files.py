from typing import Union
from os.path import join
from datetime import date
import posixpath
import pandas as pd

from .constants import *
from .granule_ID import GranuleID
from .download_file import download_file
from .ECOSTRESS_CMR_search import ECOSTRESS_CMR_search

def get_granule_from_listing(listing: pd.DataFrame) -> GranuleID:
    granule_IDs = list(listing.granule.unique())

    if len(granule_IDs) != 1:
        raise ValueError(f"there are {len(granule_IDs)} found in listing")
    
    granule_ID = GranuleID(granule_IDs[0])

    return granule_ID

def download_ECOSTRESS_granule_files(
        product: str, 
        tile: str, 
        aquisition_date: Union[date, str], 
        orbit: int = None,
        scene: int = None,
        parent_directory: str = ".",
        CMR_file_listing_df: pd.DataFrame = None,
        CMR_search_URL: str = CMR_SEARCH_URL) -> str:
    # run CMR search to list all files in ECOSTRESS granule
    if CMR_file_listing_df is None:
        CMR_file_listing_df = ECOSTRESS_CMR_search(
            product=product, 
            orbit=orbit,
            scene=scene,
            tile=tile, 
            start_date=aquisition_date, 
            end_date=aquisition_date
        )

    # check that there is exactly one granule listed and get granule ID
    granule_ID = get_granule_from_listing(CMR_file_listing_df)

    if orbit != granule_ID.orbit:
        raise ValueError(f"given orbit {orbit} does not match listed orbit {granule_ID.orbit}")
    
    if scene != granule_ID.scene:
        raise ValueError(f"given scene {scene} does not match listed scene {granule_ID.scene}")

    if tile != granule_ID.tile:
        raise ValueError(f"given tile {tile} does not match listed tile {granule_ID.tile}")
    
    # construct granule directory path
    directory = join(parent_directory, str(granule_ID))

    for URL in list(CMR_file_listing_df.URL):
        filename = join(directory, posixpath.basename(URL))
        download_file(URL, filename)
    
    return directory