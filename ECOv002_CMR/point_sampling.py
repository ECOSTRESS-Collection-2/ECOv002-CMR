"""
Point sampling from ECOSTRESS Collection 2 data.

This module provides functions for sampling ECOSTRESS raster data at specific
geographic points, supporting both single-point queries and batch processing
of GeoDataFrames.
"""

from datetime import date, datetime
from typing import Optional, Union, List, Dict, Tuple, Any
import logging

import pandas as pd
import geopandas as gpd
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.warp import transform, transform_bounds
from rasterio.crs import CRS
from shapely.geometry import Point
from sentinel_tiles import sentinel_tiles

from .ECOSTRESS_CMR_search import ECOSTRESS_CMR_search
from .authentication import setup_earthdata_session

logger = logging.getLogger(__name__)

# Layer name to (product, variable) mapping
LAYER_MAPPING = {
    'ST_C': ('L2T_LSTE', 'LST'),           # Surface Temperature in Celsius (converted from Kelvin)
    'LST': ('L2T_LSTE', 'LST'),            # Land Surface Temperature (Kelvin)
    'LST_err': ('L2T_LSTE', 'LST_err'),    # LST uncertainty
    'QC': ('L2T_LSTE', 'QC'),              # Quality control
    'emissivity': ('L2T_LSTE', 'EmisWB'),  # Wideband emissivity (alias for EmisWB)
    'EmisWB': ('L2T_LSTE', 'EmisWB'),      # Wideband emissivity
    'NDVI': ('L2T_STARS', 'NDVI'),         # Normalized Difference Vegetation Index
    'EVI': ('L2T_STARS', 'EVI'),           # Enhanced Vegetation Index
    'albedo': ('L2T_STARS', 'albedo'),     # Surface albedo
    'SAVI': ('L2T_STARS', 'SAVI'),         # Soil Adjusted Vegetation Index
    'Ta_C': ('L3T_MET', 'Ta'),             # Air Temperature in Celsius (native units)
    'Ta': ('L3T_MET', 'Ta'),               # Air Temperature in Celsius (alias for Ta_C)
    'RH': ('L3T_MET', 'RH'),               # Relative humidity
    'SM': ('L3T_SM', 'SM'),                # Soil moisture
}

DEFAULT_LAYERS = ['ST_C', 'NDVI', 'albedo']


def sample_point_from_url(
    session: requests.Session,
    url: str,
    lon: float,
    lat: float,
    max_retries: int = 2,
    timeout: int = 90
) -> Tuple[Optional[float], Union[Dict[str, Any], str]]:
    """
    Sample a single point from a raster file accessed via URL.
    
    Downloads the raster to memory, checks if the point falls within bounds,
    transforms coordinates if needed, and extracts the pixel value.
    
    Parameters
    ----------
    session : requests.Session
        Authenticated session for downloading data.
    url : str
        URL of the GeoTIFF file to sample.
    lon : float
        Longitude of the point in WGS84 (EPSG:4326).
    lat : float
        Latitude of the point in WGS84 (EPSG:4326).
    max_retries : int, default 2
        Number of retry attempts for network failures.
    timeout : int, default 90
        Request timeout in seconds.
    
    Returns
    -------
    tuple
        (value, metadata) if successful, where:
        - value: float pixel value at the point
        - metadata: dict with 'bounds', 'crs', 'nodata'
        
        (None, reason) if unsuccessful, where:
        - reason: string describing why sampling failed
    
    Examples
    --------
    >>> from ECOv002_CMR import setup_earthdata_session, sample_point_from_url
    >>> session = setup_earthdata_session()
    >>> url = "https://data.lpdaac.earthdatacloud.nasa.gov/..."
    >>> value, meta = sample_point_from_url(session, url, -118.24, 34.05)
    >>> if value is not None:
    ...     print(f"Sampled value: {value}")
    ... else:
    ...     print(f"Sampling failed: {meta}")
    """
    for attempt in range(max_retries):
        try:
            response = session.get(url, allow_redirects=True, timeout=timeout)
            response.raise_for_status()
            
            with MemoryFile(response.content) as memfile:
                with memfile.open() as src:
                    # Transform bounds to WGS84 for comparison with input lon/lat
                    bounds_wgs84 = transform_bounds(
                        src.crs, 
                        CRS.from_epsg(4326), 
                        src.bounds.left, 
                        src.bounds.bottom, 
                        src.bounds.right, 
                        src.bounds.top
                    )
                    
                    # Check if point is within bounds (both in WGS84)
                    if not (bounds_wgs84[0] <= lon <= bounds_wgs84[2] and 
                           bounds_wgs84[1] <= lat <= bounds_wgs84[3]):
                        return None, "outside_bounds"
                    
                    # Transform point from WGS84 to raster CRS for pixel indexing
                    try:
                        xs, ys = transform(
                            CRS.from_epsg(4326),
                            src.crs,
                            [lon],
                            [lat]
                        )
                        x_proj, y_proj = xs[0], ys[0]
                    except Exception as e:
                        return None, f"coordinate_transform_error: {e}"
                    
                    # Get pixel coordinates using transformed coordinates
                    try:
                        py, px = src.index(x_proj, y_proj)
                    except Exception as e:
                        return None, f"coordinate_error: {e}"
                    
                    # Verify pixel is within image
                    if not (0 <= py < src.height and 0 <= px < src.width):
                        return None, "outside_image"
                    
                    # Read pixel value
                    window = Window(px, py, 1, 1)
                    value = src.read(1, window=window)[0, 0]
                    
                    # Check for nodata
                    nodata = src.nodata
                    if nodata is not None and value == nodata:
                        return None, "nodata"
                    
                    return value, {
                        'bounds': bounds_wgs84,
                        'crs': str(src.crs),
                        'nodata': nodata
                    }
                    
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                continue
            return None, f"network_error: {str(e)}"
        except Exception as e:
            return None, f"error: {str(e)}"
    
    return None, "failed_after_retries"


def find_sentinel2_tile(lon: float, lat: float) -> Optional[str]:
    """
    Find which Sentinel-2 tile contains the given coordinate.
    
    Parameters
    ----------
    lon : float
        Longitude in WGS84 (EPSG:4326).
    lat : float
        Latitude in WGS84 (EPSG:4326).
    
    Returns
    -------
    str or None
        Sentinel-2 tile name (e.g., '11SLT') or None if not found.
    
    Examples
    --------
    >>> from ECOv002_CMR import find_sentinel2_tile
    >>> tile = find_sentinel2_tile(-118.24, 34.05)
    >>> print(f"Tile: {tile}")
    Tile: 11SLT
    """
    try:
        point = Point(lon, lat)
        tile_name = sentinel_tiles.nearest(point)
        return tile_name
    except Exception as e:
        logger.error(f"Failed to find Sentinel-2 tile: {e}")
        return None


def sample_points_over_date_range(
    geometry: Union[gpd.GeoDataFrame, gpd.GeoSeries, pd.Series, List[Point], Point],
    start_date: Union[date, str],
    end_date: Union[date, str],
    layers: Optional[List[str]] = None,
    tile: Optional[Union[str, List[str]]] = None,
    session: Optional[requests.Session] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Sample ECOSTRESS data at multiple points for all available acquisitions in a date range.
    
    This function searches for all ECOSTRESS granules available within the specified
    date range that cover the input points, then samples all found granules at each point.
    This is the recommended approach since ECOSTRESS acquisition times are not known
    in advance.
    
    Parameters
    ----------
    geometry : GeoDataFrame, GeoSeries, Series, list of Point, or Point
        Point geometries to sample. Can be:
        - GeoDataFrame with point geometries (uses 'geometry' column)
        - GeoSeries of point geometries
        - pandas Series (single row from GeoDataFrame, with 'geometry' key)
        - List of shapely Point objects
        - Single shapely Point object
        Must be in or will be converted to EPSG:4326 (WGS84).
        When a Series is provided, all non-geometry columns are propagated as metadata.
    start_date : date or str
        Start date for searching granules.
    end_date : date or str
        End date for searching granules.
    layers : list of str, optional
        Layer names to sample. Defaults to ['ST_C', 'NDVI', 'albedo'].
        Available layers:
        - 'ST_C': Surface Temperature in Celsius (L2T_LSTE)
        - 'LST': Land Surface Temperature in Kelvin (L2T_LSTE)
        - 'LST_err': LST uncertainty (L2T_LSTE)
        - 'QC': Quality control flags (L2T_LSTE)
        - 'emissivity': Wideband emissivity (L2T_LSTE) - recommended
        - 'EmisWB': Wideband emissivity (L2T_LSTE) - alias for emissivity
        - 'NDVI': Normalized Difference Vegetation Index (L2T_STARS)
        - 'EVI': Enhanced Vegetation Index (L2T_STARS)
        - 'albedo': Surface albedo (L2T_STARS)
        - 'SAVI': Soil Adjusted Vegetation Index (L2T_STARS)
        - 'Ta_C': Air Temperature in Celsius (L3T_MET) - recommended
        - 'Ta': Air Temperature in Celsius (L3T_MET) - alias for Ta_C
        - 'RH': Relative humidity (L3T_MET)
        - 'SM': Soil moisture (L3T_SM)
    tile : str or list of str, optional
        Sentinel-2 tile name(s). If None, automatically determined from coordinates.
        Can be a single tile ('11SLT') or list of tiles for multiple points.
    session : requests.Session, optional
        Authenticated session for NASA Earthdata. If None, will attempt
        to set up using credentials from environment or .netrc.
    verbose : bool, default True
        Whether to print progress messages.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sampled values for all available acquisitions. 
        Each row represents one point-acquisition combination.
        Contains original GeoDataFrame columns plus:
        - granule: Granule identifier
        - timestamp: Acquisition datetime
        - tile: Sentinel-2 tile
        - One column per requested variable (LST, NDVI, etc.)
    
    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:4326, or required columns are missing.
    RuntimeError
        If authentication fails.
    
    Examples
    --------
    Create a GeoDataFrame with sample points (no datetime needed):
    
    >>> import geopandas as gpd
    >>> from datetime import date
    >>> from shapely.geometry import Point
    >>> 
    >>> data = {
    ...     'site_id': ['LA_downtown', 'LA_airport'],
    ...     'geometry': [Point(-118.24, 34.05), Point(-118.40, 33.94)]
    ... }
    >>> gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    
    Sample all ECOSTRESS data in June 2025 (using defaults):
    
    >>> from ECOv002_CMR import sample_points_over_date_range
    >>> results = sample_points_over_date_range(
    ...     geometry=gdf,
    ...     start_date=date(2025, 6, 1),
    ...     end_date=date(2025, 6, 30)
    ... )
    >>> # Results show all available acquisitions sampled at both points
    >>> # Default layers: ST_C (Surface Temp in Celsius), NDVI, albedo
    >>> print(results[['site_id', 'timestamp', 'ST_C', 'NDVI', 'albedo']])
    
    Or use a list of points directly:
    
    >>> points = [Point(-118.24, 34.05), Point(-118.40, 33.94)]
    >>> results = sample_points_over_date_range(
    ...     geometry=points,
    ...     start_date=date(2025, 6, 1),
    ...     end_date=date(2025, 6, 30),
    ... )
    
    Or use a single point:
    
    >>> single_point = Point(-118.24, 34.05)
    >>> results = sample_points_over_date_range(
    ...     geometry=single_point,
    ...     start_date=date(2025, 6, 1),
    ...     end_date=date(2025, 6, 30),
    ...     layers=['ST_C', 'NDVI', 'EVI']  # Custom layers
    ... )
    
    Or use a single row from a GeoDataFrame (metadata will be propagated):
    
    >>> # Sample just one site, keeping all its metadata
    >>> results = sample_points_over_date_range(
    ...     geometry=gdf.iloc[0],  # Single row with site_id and other columns
    ...     start_date=date(2025, 6, 1),
    ...     end_date=date(2025, 6, 30)
    ... )
    >>> # Output includes 'site_id' and other columns from the input row
    >>> print(results[['site_id', 'timestamp', 'ST_C']])
    """
    # Set default layers if not provided
    if layers is None:
        layers = DEFAULT_LAYERS.copy()
    
    # Convert geometry input to standardized format
    points_data = []  # List of (index, point, metadata_dict)
    
    if isinstance(geometry, gpd.GeoDataFrame):
        # Extract from GeoDataFrame
        if geometry.crs is None:
            raise ValueError("GeoDataFrame must have a defined CRS")
        if not geometry.crs.equals(CRS.from_epsg(4326)):
            if verbose:
                logger.info(f"Reprojecting GeoDataFrame from {geometry.crs} to EPSG:4326")
            geometry = geometry.to_crs('EPSG:4326')
        
        for idx, row in geometry.iterrows():
            point = row.get('geometry') or row.get(geometry.index.name)
            if not isinstance(point, Point):
                logger.warning(f"Row {idx}: geometry is not a Point, skipping")
                continue
            # Copy all non-geometry columns as metadata
            metadata = {col: row[col] for col in geometry.columns if col != 'geometry'}
            points_data.append((idx, point, metadata))
    
    elif isinstance(geometry, gpd.GeoSeries):
        # Extract from GeoSeries
        if geometry.crs is None:
            raise ValueError("GeoSeries must have a defined CRS")
        if not geometry.crs.equals(CRS.from_epsg(4326)):
            if verbose:
                logger.info(f"Reprojecting GeoSeries from {geometry.crs} to EPSG:4326")
            geometry = geometry.to_crs('EPSG:4326')
        
        for idx, point in geometry.items():
            if not isinstance(point, Point):
                logger.warning(f"Index {idx}: geometry is not a Point, skipping")
                continue
            points_data.append((idx, point, {}))
    
    elif isinstance(geometry, pd.Series):
        # Single row from GeoDataFrame (pandas Series with geometry)
        if 'geometry' not in geometry:
            raise ValueError("Series must have a 'geometry' key")
        
        point = geometry['geometry']
        if not isinstance(point, Point):
            raise ValueError(f"Series geometry is not a Point: {type(point)}")
        
        # Check if point needs reprojection (if it's a GeoSeries with CRS info)
        if isinstance(geometry, gpd.GeoSeries) and geometry.crs is not None:
            if not geometry.crs.equals(CRS.from_epsg(4326)):
                if verbose:
                    logger.info(f"Reprojecting point from {geometry.crs} to EPSG:4326")
                geometry = geometry.to_crs('EPSG:4326')
                point = geometry['geometry']
        
        # Copy all non-geometry columns as metadata
        metadata = {key: geometry[key] for key in geometry.index if key != 'geometry'}
        points_data.append((0, point, metadata))
    
    elif isinstance(geometry, list):
        # List of Point objects
        for idx, point in enumerate(geometry):
            if not isinstance(point, Point):
                logger.warning(f"Index {idx}: not a Point object, skipping")
                continue
            points_data.append((idx, point, {}))
    
    elif isinstance(geometry, Point):
        # Single Point object
        points_data.append((0, geometry, {}))
    
    else:
        raise ValueError(
            "geometry must be a GeoDataFrame, GeoSeries, Series (single row), list of Point objects, or a single Point"
        )
    
    if not points_data:
        raise ValueError("No valid point geometries found in input")
    
    # Map layer names to products/variables
    products = {}  # {product: [variable1, variable2, ...]}
    layer_to_output_name = {}  # Map layer name to output column name
    
    for layer in layers:
        if layer not in LAYER_MAPPING:
            logger.warning(f"Unknown layer '{layer}', skipping")
            continue
        
        product, variable = LAYER_MAPPING[layer]
        if product not in products:
            products[product] = []
        if variable not in products[product]:
            products[product].append(variable)
        
        # For ST_C, we'll convert from LST (Kelvin) after sampling
        # For Ta_C/Ta and emissivity/EmisWB, the source is already correct, just rename
        if layer == 'ST_C':
            layer_to_output_name['LST'] = 'ST_C'
        elif layer in ['Ta_C', 'Ta']:
            layer_to_output_name['Ta'] = layer  # Rename Ta variable to match requested layer name
        elif layer in ['emissivity', 'EmisWB']:
            layer_to_output_name['EmisWB'] = layer  # Rename EmisWB variable to match requested layer name
        else:
            layer_to_output_name[variable] = layer
    
    # Parse dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    
    # Set up authentication
    if session is None:
        if verbose:
            logger.info("Setting up NASA Earthdata authentication...")
        session = setup_earthdata_session()
    
    # Determine tiles for each point
    point_tiles = {}  # Map point index to tile
    
    # Handle tile parameter
    if isinstance(tile, str):
        # Single tile for all points
        for idx, _, _ in points_data:
            point_tiles[idx] = tile
    elif isinstance(tile, list):
        # List of tiles (one per point)
        if len(tile) != len(points_data):
            raise ValueError(f"tile list length ({len(tile)}) must match number of points ({len(points_data)})")
        for (idx, _, _), t in zip(points_data, tile):
            point_tiles[idx] = t
    else:
        # Automatically determine tiles from coordinates
        for idx, point, _ in points_data:
            lon, lat = point.x, point.y
            determined_tile = find_sentinel2_tile(lon, lat)
            if not determined_tile:
                logger.warning(f"Point {idx}: could not determine Sentinel-2 tile, skipping")
                continue
            point_tiles[idx] = determined_tile
    
    tiles_to_search = set(point_tiles.values())
    
    if not tiles_to_search:
        logger.warning("No valid tiles found for any points")
        return pd.DataFrame()
    
    if verbose:
        logger.info(f"Searching {len(tiles_to_search)} tile(s): {', '.join(sorted(tiles_to_search))}")
        logger.info(f"Date range: {start_date} to {end_date}")
    
    # Search CMR for all granules across all tiles and products
    all_granule_files = {}  # {tile: {(granule, variable): {url, orbit, scene, product}}}
    
    for tile in tiles_to_search:
        if verbose:
            logger.info(f"\nSearching tile {tile}...")
        
        tile_files = {}
        
        for product, variables in products.items():
            if not isinstance(variables, list):
                variables = [variables]
            
            try:
                search_results = ECOSTRESS_CMR_search(
                    product=product,
                    tile=tile,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if search_results.empty:
                    if verbose:
                        logger.info(f"  No {product} granules found")
                    continue
                
                # Filter to desired variables
                for variable in variables:
                    var_files = search_results[
                        (search_results['type'] == 'GeoTIFF Data') & 
                        (search_results['variable'] == variable)
                    ]
                    
                    if not var_files.empty:
                        if verbose:
                            logger.info(f"  Found {len(var_files)} {variable} files")
                        
                        for _, file_row in var_files.iterrows():
                            key = (file_row['granule'], variable)
                            tile_files[key] = {
                                'url': file_row['URL'],
                                'orbit': file_row.get('orbit'),
                                'scene': file_row.get('scene'),
                                'product': product
                            }
            
            except Exception as e:
                logger.error(f"  Error searching {product}: {e}")
                continue
        
        all_granule_files[tile] = tile_files
    
    # Get unique granules per tile
    granules_by_tile = {}
    for tile, tile_files in all_granule_files.items():
        granules = list(set(granule for granule, _ in tile_files.keys()))
        granules_by_tile[tile] = granules
        if verbose:
            logger.info(f"\nTile {tile}: {len(granules)} unique granules to sample")
    
    if verbose:
        logger.info(f"\nSampling all point-granule combinations...")
    
    # Sample each point at each granule in its tile
    all_results = []
    
    for idx, point, metadata in points_data:
        if idx not in point_tiles:
            continue
        
        lon, lat = point.x, point.y
        point_tile = point_tiles[idx]
        tile_files = all_granule_files.get(point_tile, {})
        granules = granules_by_tile.get(point_tile, [])
        
        if not granules:
            if verbose:
                logger.info(f"  Point {idx}: No granules available")
            continue
        
        if verbose:
            logger.info(f"  Point {idx} ({lat:.4f}, {lon:.4f}): Sampling {len(granules)} granule(s)")
        
        # Separate LST, STARS, and other daily products (L3T, L4T) by date
        lst_granules = {}  # {date: [(granule, timestamp, orbit, scene), ...]}
        stars_granules = {}  # {date: granule}
        daily_granules = {}  # {product: {date: granule}} for L3T/L4T products
        
        for granule in granules:
            parts = granule.split('_')
            product = "_".join(parts[1:3]) if len(parts) > 2 else None
            
            if product == 'L2T_STARS':
                # STARS granule: ECO_L2T_STARS_<tile>_<date>_<build>
                if len(parts) > 4:
                    date_str = parts[4]  # YYYYMMDD
                    stars_granules[date_str] = granule
            elif product == 'L2T_LSTE':
                # LST granule: ECO_L2T_LSTE_<orbit>_<scene>_<tile>_<timestamp>_<build>
                if len(parts) > 6:
                    timestamp = parts[6]
                    date_str = timestamp[:8]  # Extract YYYYMMDD from YYYYMMDDTHHMMSS
                    orbit = parts[3] if len(parts) > 3 else None
                    scene = parts[4] if len(parts) > 4 else None
                    
                    if date_str not in lst_granules:
                        lst_granules[date_str] = []
                    lst_granules[date_str].append((granule, timestamp, orbit, scene))
            elif product and (product.startswith('L3T_') or product.startswith('L4T_')):
                # L3T/L4T products: ECO_L3T_<product>_<orbit>_<scene>_<tile>_<timestamp>_<build>
                # Similar structure to L2T_LSTE with orbit/scene/tile/timestamp
                if len(parts) > 6:
                    timestamp = parts[6]
                    date_str = timestamp[:8]  # Extract YYYYMMDD from YYYYMMDDTHHMMSS
                    
                    if product not in daily_granules:
                        daily_granules[product] = {}
                    daily_granules[product][date_str] = granule
        
        # Process LST granules and match with STARS data by date
        for date_str, lst_list in lst_granules.items():
            stars_granule = stars_granules.get(date_str)
            
            for granule, timestamp, orbit, scene in lst_list:
                granule_data = {
                    'point_index': idx,
                    'granule': granule,
                    'lon': lon,
                    'lat': lat,
                    'tile': point_tile,
                    'timestamp': timestamp,
                    'orbit': orbit,
                    'scene': scene
                }
                
                # Copy metadata from original input
                granule_data.update(metadata)
                
                has_data = False
                
                # Sample LST variables from the LST granule
                for variable in products.get('L2T_LSTE', []):
                    key = (granule, variable)
                    if key not in tile_files:
                        continue
                    
                    file_info = tile_files[key]
                    value, meta = sample_point_from_url(
                        session, 
                        file_info['url'], 
                        lon, 
                        lat
                    )
                    
                    if value is not None:
                        has_data = True
                        granule_data[variable] = value
                
                # Sample STARS variables from the STARS granule for this date
                if stars_granule:
                    for variable in products.get('L2T_STARS', []):
                        key = (stars_granule, variable)
                        if key not in tile_files:
                            continue
                        
                        file_info = tile_files[key]
                        value, meta = sample_point_from_url(
                            session, 
                            file_info['url'], 
                            lon, 
                            lat
                        )
                        
                        if value is not None:
                            has_data = True
                            granule_data[variable] = value
                
                # Sample L3T/L4T products for this date
                for product, date_granules in daily_granules.items():
                    if date_str in date_granules:
                        product_granule = date_granules[date_str]
                        for variable in products.get(product, []):
                            key = (product_granule, variable)
                            if key not in tile_files:
                                continue
                            
                            file_info = tile_files[key]
                            value, meta = sample_point_from_url(
                                session, 
                                file_info['url'], 
                                lon, 
                                lat
                            )
                            
                            if value is not None:
                                has_data = True
                                granule_data[variable] = value
                
                if has_data:
                    all_results.append(granule_data)
        
        # Also process STARS-only granules (dates with STARS but no LST)
        stars_only_dates = set(stars_granules.keys()) - set(lst_granules.keys())
        for date_str in stars_only_dates:
            stars_granule = stars_granules[date_str]
            
            granule_data = {
                'point_index': idx,
                'granule': stars_granule,
                'lon': lon,
                'lat': lat,
                'tile': point_tile,
                'timestamp': date_str,
                'orbit': None,
                'scene': None
            }
            
            # Copy metadata from original input
            granule_data.update(metadata)
            
            has_data = False
            
            # Sample STARS variables
            for variable in products.get('L2T_STARS', []):
                key = (stars_granule, variable)
                if key not in tile_files:
                    continue
                
                file_info = tile_files[key]
                value, meta = sample_point_from_url(
                    session, 
                    file_info['url'], 
                    lon, 
                    lat
                )
                
                if value is not None:
                    has_data = True
                    granule_data[variable] = value
            
            # Sample L3T/L4T products for this date
            for product, date_granules in daily_granules.items():
                if date_str in date_granules:
                    product_granule = date_granules[date_str]
                    for variable in products.get(product, []):
                        key = (product_granule, variable)
                        if key not in tile_files:
                            continue
                        
                        file_info = tile_files[key]
                        value, meta = sample_point_from_url(
                            session, 
                            file_info['url'], 
                            lon, 
                            lat
                        )
                        
                        if value is not None:
                            has_data = True
                            granule_data[variable] = value
            
            if has_data:
                all_results.append(granule_data)
    
    if not all_results:
        logger.warning("No data sampled from any point-granule combinations")
        return pd.DataFrame()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Apply layer name mappings and conversions
    for var_name, layer_name in layer_to_output_name.items():
        if var_name in results_df.columns:
            if layer_name == 'ST_C' and var_name == 'LST':
                # Convert Kelvin to Celsius
                results_df[layer_name] = results_df[var_name] - 273.15
                # Keep LST in Kelvin only if explicitly requested
                if 'LST' not in layers:
                    results_df = results_df.drop(columns=[var_name])
            elif layer_name != var_name:
                # Rename column (no conversion needed for Ta - already Celsius)
                results_df = results_df.rename(columns={var_name: layer_name})
    
    if verbose:
        logger.info(f"\n✓ Sampled {len(results_df)} observations from {len(points_data)} input points")
        logger.info(f"  Unique granules sampled: {results_df['granule'].nunique()}")
        logger.info(f"  Points with data: {results_df['point_index'].nunique()}")
    
    return results_df


def sample_points_from_geodataframe(
    geometry: Union[gpd.GeoDataFrame, gpd.GeoSeries, List[Point]],
    layers: Optional[List[str]] = None,
    datetime_col: str = 'datetime',
    tile_col: Optional[str] = None,
    session: Optional[requests.Session] = None,
    date_buffer_days: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Sample ECOSTRESS data at multiple points with known acquisition times.
    
    Takes points with associated datetimes and samples ECOSTRESS data.
    For unknown acquisition times, use sample_points_over_date_range instead.
    
    Parameters
    ----------
    geometry : GeoDataFrame, GeoSeries, or list of Point
        Point geometries to sample. Can be:
        - GeoDataFrame with point geometry column and datetime column
        - GeoSeries of Point geometries (requires datetime column if GeoDataFrame parent)
        - List of shapely Point objects (requires datetime specification)
    layers : list of str, optional
        Layer names to sample. Defaults to ['ST_C', 'NDVI', 'albedo'].
        Available: ST_C, LST, NDVI, EVI, albedo, SAVI, QC, LST_err, EmisWB
    datetime_col : str, default 'datetime'
        Name of the datetime column (only used if geometry is a GeoDataFrame).
    tile_col : str, optional
        Name of column containing Sentinel-2 tile names. If None, tiles
        will be automatically determined from point coordinates.
    session : requests.Session, optional
        Authenticated session for NASA Earthdata. If None, will attempt
        to set up using credentials from environment or .netrc.
    date_buffer_days : int, default 0
        Number of days before/after each datetime to include in search.
    verbose : bool, default True
        Whether to print progress messages.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sampled values. Contains original data columns
        plus one column per requested layer, along with metadata columns
        (granule, orbit, scene, timestamp). ST_C values are in Celsius.
    
    Raises
    ------
    ValueError
        If geometry is not in EPSG:4326, or required columns are missing.
    RuntimeError
        If authentication fails.
    
    Examples
    --------
    Sample with GeoDataFrame (default layers):
    
    >>> import geopandas as gpd
    >>> from datetime import datetime
    >>> from shapely.geometry import Point
    >>> 
    >>> data = {
    ...     'datetime': [datetime(2025, 6, 15), datetime(2025, 6, 20)],
    ...     'site_id': ['LA_downtown', 'LA_airport'],
    ...     'geometry': [Point(-118.24, 34.05), Point(-118.40, 33.94)]
    ... }
    >>> gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    >>> 
    >>> from ECOv002_CMR import sample_points_from_geodataframe
    >>> results = sample_points_from_geodataframe(geometry=gdf)
    >>> print(results[['site_id', 'datetime', 'ST_C', 'NDVI', 'albedo']])
    
    Sample with custom layers:
    
    >>> results = sample_points_from_geodataframe(
    ...     geometry=gdf,
    ...     layers=['ST_C', 'EVI', 'QC']
    ... )
    """
    # Default layers
    if layers is None:
        layers = DEFAULT_LAYERS
    
    # Normalize geometry input
    if isinstance(geometry, list):
        # List of Point objects - convert to GeoSeries
        gdf = gpd.GeoDataFrame({'geometry': geometry}, crs='EPSG:4326')
    elif isinstance(geometry, gpd.GeoSeries):
        # GeoSeries - convert to GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': geometry}, crs=geometry.crs)
    elif isinstance(geometry, gpd.GeoDataFrame):
        gdf = geometry
    else:
        raise TypeError(
            f"geometry must be GeoDataFrame, GeoSeries, or list of Point, got {type(geometry)}"
        )
    
    # Validate inputs
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a defined CRS")
    
    if not gdf.crs.equals(CRS.from_epsg(4326)):
        logger.info(f"Reprojecting GeoDataFrame from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    if datetime_col not in gdf.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in GeoDataFrame")
    
    # Build products dictionary from layer names
    products = {}
    for layer in layers:
        if layer not in LAYER_MAPPING:
            raise ValueError(
                f"Unknown layer '{layer}'. Available: {', '.join(LAYER_MAPPING.keys())}"
            )
        product, variable = LAYER_MAPPING[layer]
        if product not in products:
            products[product] = []
        if isinstance(products[product], str):
            products[product] = [products[product]]
        if variable not in products[product]:
            products[product].append(variable)
    
    # Convert single variables to strings
    for product in products:
        if len(products[product]) == 1:
            products[product] = products[product][0]
    
    # Set up authentication
    if session is None:
        if verbose:
            logger.info("Setting up NASA Earthdata authentication...")
        session = setup_earthdata_session()
    
    # Prepare results list
    all_results = []
    
    # Process each point
    for idx, row in gdf.iterrows():
        point = row['geometry']
        dt = row[datetime_col]
        
        if not isinstance(point, Point):
            logger.warning(f"Row {idx}: geometry is not a Point, skipping")
            continue
        
        lon, lat = point.x, point.y
        
        # Convert datetime to date for CMR search
        if isinstance(dt, datetime):
            search_date = dt.date()
        elif isinstance(dt, date):
            search_date = dt
        else:
            try:
                search_date = pd.to_datetime(dt).date()
            except:
                logger.warning(f"Row {idx}: cannot parse datetime '{dt}', skipping")
                continue
        
        # Determine tile
        if tile_col and tile_col in row:
            tile = row[tile_col]
        else:
            tile = find_sentinel2_tile(lon, lat)
            if not tile:
                logger.warning(f"Row {idx}: could not determine Sentinel-2 tile, skipping")
                continue
        
        if verbose:
            logger.info(f"Processing row {idx}: ({lat:.4f}, {lon:.4f}) on {search_date} in tile {tile}")
        
        # Calculate date range
        from datetime import timedelta
        start_date = search_date - timedelta(days=date_buffer_days)
        end_date = search_date + timedelta(days=date_buffer_days)
        
        # Search CMR for each product and collect files
        point_files = {}  # {(granule, variable): {url, orbit, scene, product}}
        
        for product, variables in products.items():
            if not isinstance(variables, list):
                variables = [variables]
            
            try:
                search_results = ECOSTRESS_CMR_search(
                    product=product,
                    tile=tile,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if search_results.empty:
                    if verbose:
                        logger.info(f"  No {product} granules found")
                    continue
                
                # Filter to desired variables
                for variable in variables:
                    var_files = search_results[
                        (search_results['type'] == 'GeoTIFF Data') & 
                        (search_results['variable'] == variable)
                    ]
                    
                    for _, file_row in var_files.iterrows():
                        key = (file_row['granule'], variable)
                        point_files[key] = {
                            'url': file_row['URL'],
                            'orbit': file_row.get('orbit'),
                            'scene': file_row.get('scene'),
                            'product': product
                        }
                
            except Exception as e:
                logger.error(f"  Error searching {product}: {e}")
                continue
        
        if not point_files:
            if verbose:
                logger.info(f"  No granules found for this point")
            continue
        
        # Sample from each granule
        unique_granules = list(set(granule for granule, _ in point_files.keys()))
        
        for granule in unique_granules:
            granule_data = {
                'gdf_index': idx,
                'granule': granule,
                'lon': lon,
                'lat': lat,
                'tile': tile,
                'search_date': search_date
            }
            
            # Copy original row data
            for col in gdf.columns:
                if col != geometry_col and col not in granule_data:
                    granule_data[col] = row[col]
            
            # Extract timestamp from granule name
            parts = granule.split('_')
            if len(parts) > 6:
                granule_data['timestamp'] = parts[6] if 'T' in parts[6] else parts[4]
            
            has_data = False
            
            # Sample each variable for this granule
            for variable in [v for vars in products.values() for v in (vars if isinstance(vars, list) else [vars])]:
                key = (granule, variable)
                if key not in point_files:
                    continue
                
                file_info = point_files[key]
                value, meta = sample_point_from_url(
                    session, 
                    file_info['url'], 
                    lon, 
                    lat
                )
                
                if value is not None:
                    has_data = True
                    granule_data[variable] = value
                    if verbose:
                        logger.info(f"    {variable}: {value:.4f}")
                else:
                    if verbose and meta not in ["outside_bounds", "nodata"]:
                        logger.info(f"    {variable}: {meta}")
            
            if has_data:
                all_results.append(granule_data)
    
    if not all_results:
        logger.warning("No data sampled from any points")
        return pd.DataFrame()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Apply layer renaming and conversions
    # Map product variables back to layer names
    variable_to_layer = {}
    for layer in layers:
        product, variable = LAYER_MAPPING[layer]
        variable_to_layer[variable] = layer
    
    # Rename columns and apply conversions
    rename_map = {}
    for variable, layer in variable_to_layer.items():
        if variable in results_df.columns and variable != layer:
            rename_map[variable] = layer
    
    if rename_map:
        results_df = results_df.rename(columns=rename_map)
    
    # Convert LST to Celsius for ST_C layer
    if 'ST_C' in results_df.columns:
        results_df['ST_C'] = results_df['ST_C'] - 273.15
    
    if verbose:
        logger.info(f"\nSampled {len(results_df)} observations from {len(gdf)} input points")
    
    return results_df
