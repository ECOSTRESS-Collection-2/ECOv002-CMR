from os.path import basename

def product_name_from_filename(filename: str) -> str:
    """
    Extracts the product name from an ECOSTRESS filename.

    Args:
        filename: The name of the ECOSTRESS granule file.

    Returns:
        The product name extracted from the filename.
    """
    return "_".join(basename(filename).split("_")[1:3])
