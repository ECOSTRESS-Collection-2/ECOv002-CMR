from .product_name_from_filename import product_name_from_filename
from .granule_ID import GranuleID

def granule_ID_from_filename(filename: str) -> GranuleID:
    """
    Extracts the granule name from an ECOSTRESS filename.

    Args:
        filename: The name of the ECOSTRESS granule file.

    Returns:
        The granule name extracted from the filename.
    """
    product = product_name_from_filename(filename)
    filename_base = filename.split(".")[0]

    if product == "L2T_STARS":
        granule_ID = "_".join(filename_base.split("_")[:7])
    else:
        granule_ID = "_".join(filename_base.split("_")[:9])

    return GranuleID(granule_ID)
