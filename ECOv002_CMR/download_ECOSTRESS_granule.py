import argparse
import sys
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
from .ECOSTRESS_CMR_search import ECOSTRESS_CMR_search

def download_ECOSTRESS_granule(
        product: str, 
        tile: str, 
        aquisition_date: Union[date, str], 
        orbit: int = None,
        scene: int = None,
        parent_directory: str = DOWNLOAD_DIRECTORY,
        CMR_file_listing_df: pd.DataFrame = None,
        CMR_search_URL: str = CMR_SEARCH_URL) -> ECOSTRESSGranule:
    """
    Downloads an ECOSTRESS granule based on the provided parameters and returns the granule object.
    """
    # Pre-fetch the CMR file listing if not supplied
    if CMR_file_listing_df is None:
        CMR_file_listing_df = ECOSTRESS_CMR_search(
            product=product,
            tile=tile,
            acquisition_date=aquisition_date,
            orbit=orbit,
            scene=scene,
            CMR_search_URL=CMR_search_URL
        )

    # Disambiguate multiple matching revisions/versions returned by CMR in Collection 2
    if CMR_file_listing_df is not None and not CMR_file_listing_df.empty:
        if 'granule_ID' in CMR_file_listing_df.columns:
            granule_IDs = CMR_file_listing_df['granule_ID'].unique()
            if len(granule_IDs) > 1:
                # Select the latest version/revision deterministically
                latest_id = sorted(granule_IDs)[-1]
                CMR_file_listing_df = CMR_file_listing_df[
                    CMR_file_listing_df['granule_ID'] == latest_id
                ].copy()

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