from .product_name_from_filename import product_name_from_filename

def variable_from_filename(filename):
    """Extracts the variable name from an ECOSTRESS granule filename.

    Args:
        filename (str): The filename of the ECOSTRESS granule.

    Returns:
        str: The variable name extracted from the filename.
    """
    product = product_name_from_filename(filename)
    
    if product == "L2T_STARS":
        variable = "_".join(filename.split(".")[0].split("_")[7:])
    else:
        variable = "_".join(filename.split(".")[0].split("_")[9:])
    
    return variable
