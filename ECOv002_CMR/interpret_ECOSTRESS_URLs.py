from typing import List
import posixpath

import pandas as pd

from .variable_from_filename import variable_from_filename
from .granule_ID_from_filename import granule_ID_from_filename

def interpret_ECOSTRESS_URLs(
        URLs: List[str],
        orbit: int = None,
        scene: int = None) -> pd.DataFrame:
    records = []

    for URL in URLs:
        filename = posixpath.basename(URL)
        variable = ""

        granule_ID = granule_ID_from_filename(filename)
        
        # Determine file type and granule name based on filename
        if filename.endswith(".json"):
            type = "JSON Metadata"
        elif filename.endswith(".tif"):
            type = "GeoTIFF Data"
        elif filename.endswith(".jpeg"):
            type = "GeoJPEG Preview"
        elif filename.endswith(".jpeg.aux.xml"):
            type = "GeoJPEG Metadata"
        else:
            raise ValueError(f"Unknown file type for {filename}")

        # Extract variable name from filename for data files
        if filename.endswith((".tif", ".jpeg", ".jpeg.aux.xml")):
            variable = variable_from_filename(filename)

        records.append({
            "product": granule_ID.product,
            "variable": variable,
            "orbit": granule_ID.orbit, 
            "scene": granule_ID.scene, 
            "tile": granule_ID.tile, 
            "type": type,
            "granule": str(granule_ID),
            "filename": filename,
            "URL": URL
        })

    # Create a pandas DataFrame from the records
    df = pd.DataFrame(records, columns=[
        "product", 
        "variable", 
        "orbit", 
        "scene",
        "tile", 
        "type", 
        "granule", 
        "filename", 
        "URL"
    ])

    if orbit is not None:
        df = df[df['orbit'] == orbit]

    if scene is not None:
        df = df[df['scene'] == scene]

    return df
