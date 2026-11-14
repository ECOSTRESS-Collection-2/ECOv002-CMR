import argparse
import sys
from typing import List, Union
import logging
import requests
import pandas as pd
import json
import posixpath
from datetime import date
from dateutil import parser

from sentinel_tiles import sentinel_tiles

from .constants import *
from .ECOSTRESS_CMR_search_links import ECOSTRESS_CMR_search_links
from .interpret_ECOSTRESS_URLs import interpret_ECOSTRESS_URLs

logger = logging.getLogger(__name__)

def ECOSTRESS_CMR_search(
        product: Union[str, List[str]],
        tile: Union[str, List[str]],
        start_date: Union[date, str] = None, 
        end_date: Union[date, str] = None,
        acquisition_date: Union[date, str] = None,
        orbit: int = None,
        scene: int = None,
        CMR_search_URL: str = CMR_SEARCH_URL) -> pd.DataFrame:
    """
    Search the CMR API for ECOSTRESS granules and return parsed results.

    Args:
        product: ECOSTRESS product code or list of product codes. Allowed values are:
            L2T_LSTE, L2T_STARS, L3T_MET, L3T_SM, L3T_SEB, L3T_JET, L4T_ESI, L4T_WUE.
        tile: Sentinel-2 tile identifier or list of tile identifiers (for example, "11SLT").
        start_date: Start date as a date object or parsable date string.
        end_date: End date as a date object or parsable date string. Defaults to start_date.
        acquisition_date: Alternative single date string or object; sets start/end date if provided.
        orbit: Optional orbit number filter.
        scene: Optional scene number filter.
        CMR_search_URL: Base URL for CMR search.

    Returns:
        pandas.DataFrame with matched granule metadata.

    Raises:
        ValueError: If any product is invalid or no tile is provided.
    """
    # Handle acquisition_date shortcut if passed
    if acquisition_date is not None:
        if isinstance(acquisition_date, str):
            acquisition_date = parser.parse(acquisition_date).date()
        start_date = acquisition_date
        end_date = acquisition_date

    # Convert start_date and end_date to date objects if they are strings
    if start_date is None:
        raise ValueError("Either start_date or acquisition_date must be provided")

    if isinstance(start_date, str):
        start_date = parser.parse(start_date).date()

    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, str):
        end_date = parser.parse(end_date).date()

    if isinstance(product, str):
        products = [product]
    else:
        products = list(product)

    if isinstance(tile, str):
        tiles = [tile]
    else:
        tiles = list(tile)

    if len(products) == 0:
        raise ValueError("At least one product must be provided")

    if len(tiles) == 0:
        raise ValueError("At least one tile must be provided")

    invalid_products = sorted({p for p in products if p not in CONCEPT_IDS})
    if invalid_products:
        allowed_products = ", ".join(sorted(CONCEPT_IDS))
        invalid_products_text = ", ".join(invalid_products)
        raise ValueError(f"Unknown product type(s): {invalid_products_text}. Allowed products: {allowed_products}")

    products = list(dict.fromkeys(products))
    tiles = list(dict.fromkeys(tiles))

    frames = []
    for item in products:
        concept_ID = CONCEPT_IDS[item]
        for item_tile in tiles:
            # Get the URLs of ECOSTRESS granules using the helper function
            URLs = ECOSTRESS_CMR_search_links(
                concept_ID=concept_ID,
                tile=item_tile,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                CMR_search_URL=CMR_search_URL
            )

            df = interpret_ECOSTRESS_URLs(
                URLs=URLs,
                orbit=orbit,
                scene=scene
            )

            if not df.empty:
                frames.append(df)

    if len(frames) == 0:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    """Command-line interface wrapper for ECOSTRESS_CMR_search."""
    parser = argparse.ArgumentParser(
        description="Search the NASA CMR API for ECOSTRESS granules and map them to Sentinel-2 tiles."
    )
    
    allowed_product_codes = sorted(CONCEPT_IDS)
    allowed_products = ", ".join(allowed_product_codes)

    # Required arguments
    parser.add_argument("-p", "--product", required=True, type=str, nargs="+",
                        choices=allowed_product_codes,
                        help=f"One or more ECOSTRESS product codes. Allowed values: {allowed_products}.")
    parser.add_argument("-t", "--tile", required=True, type=str, nargs="+",
                        help="One or more Sentinel-2 tile identifiers (e.g., '10UEV').")
    parser.add_argument("-s", "--start-date", required=True, type=str,
                        metavar="YYYY-MM-DD",
                        help="Start date of the search period (YYYY-MM-DD).")
    
    # Optional arguments
    parser.add_argument("-e", "--end-date", type=str, default=None,
                        metavar="YYYY-MM-DD",
                        help="End date of the search period (YYYY-MM-DD). Defaults to start-date.")
    parser.add_argument("--orbit", type=int, default=None, 
                        help="Optional orbit number filter.")
    parser.add_argument("--scene", type=int, default=None, 
                        help="Optional scene number filter.")
    parser.add_argument("-o", "--output", type=str, default=None, 
                        help="Path to save the output CSV file. If not provided, results print to console.")
    parser.add_argument("--cmr-url", type=str, default=CMR_SEARCH_URL, 
                        help="Custom base URL for the CMR search API.")

    args = parser.parse_args()

    selected_products = list(dict.fromkeys(args.product))
    selected_tiles = list(dict.fromkeys(args.tile))
    tile_label = "tile" if len(selected_tiles) == 1 else "tiles"
    date_text = (
        f"on {args.start_date}"
        if args.end_date is None or args.end_date == args.start_date
        else f"from {args.start_date} to {args.end_date}"
    )

    try:
        logger.info(
            f"searching for {', '.join(selected_products)} on {tile_label} {', '.join(selected_tiles)} {date_text}"
        )
        
        df = ECOSTRESS_CMR_search(
            product=selected_products,
            tile=selected_tiles,
            start_date=args.start_date,
            end_date=args.end_date,
            CMR_search_URL=args.cmr_url
        )

        if df.empty:
            logger.info("No matching granules found for the given criteria.")
            sys.exit(0)

        logger.info(f"Success! Found {len(df)} matching items.")

        # Handle output target
        if args.output:
            df.to_csv(args.output, index=False)
            logger.info(f"Results successfully saved to: {args.output}")
        else:
            if "granule" not in df.columns:
                raise ValueError("Expected 'granule' column in search results.")

            # Preserve first-seen ordering while removing duplicates.
            granules = df["granule"].dropna().astype(str).tolist()
            ordered_unique_granules = list(dict.fromkeys(granules))

            logger.info("Granules:")
            for granule in ordered_unique_granules:
                logger.info(granule)

    except Exception as err:
        logger.info(f"Error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()