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
    gdf: gpd.GeoDataFrame,
    products: Dict[str, Union[str, List[str]]],
    start_date: Union[date, str],
    end_date: Union[date, str],
    geometry_col: str = 'geometry',
    tile_col: Optional[str] = None,
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
    gdf : gpd.GeoDataFrame
        GeoDataFrame with point geometries. No datetime column needed.
        Must be in EPSG:4326 (WGS84) coordinate system.
    products : dict
        Dictionary mapping product names to variables.
        Format: {'product_name': 'variable'} or {'product_name': ['var1', 'var2']}
        Example: {'L2T_LSTE': 'LST', 'L2T_STARS': ['NDVI', 'albedo']}
    start_date : date or str
        Start date for searching granules.
    end_date : date or str
        End date for searching granules.
    geometry_col : str, default 'geometry'
        Name of the geometry column in the GeoDataFrame.
    tile_col : str, optional
        Name of column containing Sentinel-2 tile names. If None, tiles
        will be automatically determined from point coordinates.
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
    
    Sample all ECOSTRESS data in June 2025:
    
    >>> from ECOv002_CMR import sample_points_over_date_range
    >>> products = {
    ...     'L2T_LSTE': 'LST',
    ...     'L2T_STARS': ['NDVI', 'albedo']
    ... }
    >>> results = sample_points_over_date_range(
    ...     gdf=gdf,
    ...     products=products,
    ...     start_date=date(2025, 6, 1),
    ...     end_date=date(2025, 6, 30)
    ... )
    >>> # Results show all available acquisitions sampled at both points
    >>> print(results[['site_id', 'timestamp', 'LST', 'NDVI', 'albedo']])
    """
    # Validate inputs
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a defined CRS")
    
    if not gdf.crs.equals(CRS.from_epsg(4326)):
        logger.info(f"Reprojecting GeoDataFrame from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    if geometry_col not in gdf.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not found in GeoDataFrame")
    
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
    
    # Determine unique tiles needed
    tiles_to_search = set()
    point_tiles = {}  # Map point index to tile
    
    for idx, row in gdf.iterrows():
        point = row[geometry_col]
        if not isinstance(point, Point):
            logger.warning(f"Row {idx}: geometry is not a Point, skipping")
            continue
        
        lon, lat = point.x, point.y
        
        # Determine tile
        if tile_col and tile_col in row:
            tile = row[tile_col]
        else:
            tile = find_sentinel2_tile(lon, lat)
            if not tile:
                logger.warning(f"Row {idx}: could not determine Sentinel-2 tile, skipping")
                continue
        
        point_tiles[idx] = tile
        tiles_to_search.add(tile)
    
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
    
    for idx, row in gdf.iterrows():
        if idx not in point_tiles:
            continue
        
        point = row[geometry_col]
        lon, lat = point.x, point.y
        tile = point_tiles[idx]
        tile_files = all_granule_files.get(tile, {})
        granules = granules_by_tile.get(tile, [])
        
        if not granules:
            if verbose:
                logger.info(f"  Point {idx}: No granules available")
            continue
        
        if verbose:
            logger.info(f"  Point {idx} ({lat:.4f}, {lon:.4f}): Sampling {len(granules)} granule(s)")
        
        for granule in granules:
            granule_data = {
                'gdf_index': idx,
                'granule': granule,
                'lon': lon,
                'lat': lat,
                'tile': tile
            }
            
            # Copy original row data
            for col in gdf.columns:
                if col != geometry_col and col not in granule_data:
                    granule_data[col] = row[col]
            
            # Extract timestamp from granule name
            parts = granule.split('_')
            if len(parts) > 6:
                granule_data['timestamp'] = parts[6] if 'T' in parts[6] else parts[4]
                granule_data['orbit'] = parts[3] if len(parts) > 3 else None
                granule_data['scene'] = parts[4] if len(parts) > 4 else None
            
            has_data = False
            
            # Sample each variable for this granule
            for variable in [v for vars in products.values() for v in (vars if isinstance(vars, list) else [vars])]:
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
            
            if has_data:
                all_results.append(granule_data)
    
    if not all_results:
        logger.warning("No data sampled from any point-granule combinations")
        return pd.DataFrame()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    if verbose:
        logger.info(f"\n✓ Sampled {len(results_df)} observations from {len(gdf)} input points")
        logger.info(f"  Unique granules sampled: {results_df['granule'].nunique()}")
        logger.info(f"  Points with data: {results_df['gdf_index'].nunique()}")
    
    return results_df


def sample_points_from_geodataframe(
    gdf: gpd.GeoDataFrame,
    products: Dict[str, Union[str, List[str]]],
    geometry_col: str = 'geometry',
    datetime_col: str = 'datetime',
    tile_col: Optional[str] = None,
    session: Optional[requests.Session] = None,
    date_buffer_days: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Sample ECOSTRESS data at multiple points from a GeoDataFrame.
    
    Takes a GeoDataFrame with point geometries and datetimes, searches for
    ECOSTRESS granules covering each point, and samples requested variables.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with point geometries and datetime information.
        Must be in EPSG:4326 (WGS84) coordinate system.
    products : dict
        Dictionary mapping product names to variables.
        Format: {'product_name': 'variable'} or {'product_name': ['var1', 'var2']}
        Example: {'L2T_LSTE': 'LST', 'L2T_STARS': ['NDVI', 'albedo']}
    geometry_col : str, default 'geometry'
        Name of the geometry column in the GeoDataFrame.
    datetime_col : str, default 'datetime'
        Name of the datetime column in the GeoDataFrame.
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
        DataFrame with sampled values. Contains original GeoDataFrame columns
        plus one column per requested variable, along with metadata columns
        (granule, orbit, scene, timestamp).
    
    Raises
    ------
    ValueError
        If GeoDataFrame is not in EPSG:4326, or required columns are missing.
    RuntimeError
        If authentication fails.
    
    Examples
    --------
    Create a GeoDataFrame with sample points:
    
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
    
    Sample ECOSTRESS data:
    
    >>> from ECOv002_CMR import sample_points_from_geodataframe
    >>> products = {
    ...     'L2T_LSTE': 'LST',
    ...     'L2T_STARS': ['NDVI', 'albedo']
    ... }
    >>> results = sample_points_from_geodataframe(gdf, products)
    >>> print(results[['site_id', 'datetime', 'LST', 'NDVI', 'albedo']])
    """
    # Validate inputs
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a defined CRS")
    
    if not gdf.crs.equals(CRS.from_epsg(4326)):
        logger.info(f"Reprojecting GeoDataFrame from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs('EPSG:4326')
    
    if geometry_col not in gdf.columns:
        raise ValueError(f"Geometry column '{geometry_col}' not found in GeoDataFrame")
    
    if datetime_col not in gdf.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in GeoDataFrame")
    
    # Set up authentication
    if session is None:
        if verbose:
            logger.info("Setting up NASA Earthdata authentication...")
        session = setup_earthdata_session()
    
    # Prepare results list
    all_results = []
    
    # Process each point
    for idx, row in gdf.iterrows():
        point = row[geometry_col]
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
    
    if verbose:
        logger.info(f"\nSampled {len(results_df)} observations from {len(gdf)} input points")
    
    return results_df
