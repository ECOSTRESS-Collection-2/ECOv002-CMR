from typing import List
import posixpath

import pandas as pd

def process_orbit_scene_tile_URLs(
        URLs: List[str],
        orbit: int = None,
        scene: int = None) -> pd.DataFrame:
    records = []

    for URL in URLs:
        filename = posixpath.basename(URL)
        variable = ""
        
        # Determine file type and granule name based on filename
        if filename.endswith(".json"):
            granule_name = filename.split(".")[0]
            type = "JSON Metadata"
        elif filename.endswith(".tif"):
            type = "GeoTIFF Data"
            granule_name = "_".join(filename.split("_")[:-1])
        elif filename.endswith(".jpeg"):
            type = "GeoJPEG Preview"
            granule_name = "_".join(filename.split("_")[:-1])
        elif filename.endswith(".jpeg.aux.xml"):
            type = "GeoJPEG Metadata"
            granule_name = "_".join(filename.split("_")[:-2])
        else:
            raise ValueError(f"Unknown file type for {filename}")

        # Extract variable name from filename for data files
        if filename.endswith((".tif", ".jpeg", ".jpeg.aux.xml")):
            variable = "_".join(filename.split(".")[0].split("_")[9:])

        # Parse granule metadata from granule name
        try:
            product = "_".join(granule_name.split("_")[1:3])
            orbit = int(granule_name.split("_")[3])
            scene = int(granule_name.split("_")[4])
            tile = granule_name.split("_")[5]
            # Add extracted information to the records list
            records.append({
                "product": product,
                "variable": variable,
                "orbit": orbit, 
                "scene": scene, 
                "tile": tile, 
                "type": type,
                "granule": granule_name,
                "filename": filename,
                "URL": URL
            })
        except (IndexError, ValueError) as e:
            print(e)
            print(f"Filename {filename} does not match expected pattern and was skipped.")

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
