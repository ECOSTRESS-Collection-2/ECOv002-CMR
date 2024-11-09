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
    