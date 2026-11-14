import logging
from os.path import join, expanduser
from os import makedirs

import numpy as np

from ECOv002_CMR import download_ECOSTRESS_granule
from ECOv002_granules import open_granule

# Disable logger output
# logging.getLogger().handlers = []

logger = logging.getLogger(__name__)

# Set working directory
working_directory = expanduser(join("~", "data", "ECOv002_testing"))
makedirs(working_directory, exist_ok=True)

logger.info("retrieving collection 2 L2T LSTE granule")
# Retrieve LST LSTE granule from CMR API for target date
L2T_LSTE_granule = download_ECOSTRESS_granule(
    product="L2T_LSTE",
    tile="11SPS",
    aquisition_date="2026-10-27",
    parent_directory=working_directory
)

# Retrieve L2T STARS granule from CMR API as prior
L2T_STARS_granule = download_ECOSTRESS_granule(
    product="L2T_STARS",
    tile="11SPS",
    aquisition_date="2026-10-23",
    parent_directory=working_directory
)

# Retrieve the published L2T STARS granule for the target date to compare against
L2T_STARS_expected_output = download_ECOSTRESS_granule(
    product="L2T_STARS",
    tile="11SPS",
    aquisition_date="2026-10-27",
    parent_directory=working_directory
)
