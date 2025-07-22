
import os
from os import makedirs
from os.path import exists, getsize, dirname, abspath, expanduser, join
import logging
import posixpath
import requests
from tqdm.notebook import tqdm

import colored_logging as cl

from .exceptions import *

logger = logging.getLogger(__name__)

def expand_filename(filename: str) -> str:
    """
    Expands the given filename to an absolute path.

    Args:
        filename (str): The filename to expand.

    Returns:
        str: The expanded absolute path of the filename.
    """
    return abspath(expanduser(filename))

def download_file(
        URL: str, 
        filename: str = None,
        granule_directory: str = None) -> str:
    """
    Downloads a file from the given URL to the specified filename.

    Args:
        URL (str): The URL of the file to download.
        filename (str, optional): The local filename to save the downloaded file. Defaults to None.
        granule_directory (str, optional): The directory to save the file if filename is not provided. Defaults to None.

    Returns:
        str: The path to the downloaded file.

    Raises:
        ECOSTRESSDownloadFailed: If the download fails.
    """
    # If filename is not provided, construct it using the granule_directory and the basename of the URL
    if filename is None:
        filename = join(granule_directory, posixpath.basename(URL))

    abs_filename = expand_filename(filename)

    # Check if the file exists and is zero-size, if so, remove it
    if exists(abs_filename) and getsize(abs_filename) == 0:
        logger.warning(f"removing zero-size corrupted ECOSTRESS file: {filename}")
        os.remove(abs_filename)

    # If the file already exists, log the information and return the filename
    if exists(abs_filename):
        logger.info(f"file already downloaded: {cl.file(filename)}")
        return filename

    # Log the download start information
    logger.info(f"downloading: {cl.URL(URL)} -> {cl.file(filename)}")
    directory = dirname(abs_filename)
    makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    partial_filename = f"{abs_filename}.download"

    try:
        with requests.get(URL, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(partial_filename, 'wb') as f, tqdm(
                desc=posixpath.basename(filename),
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except Exception as e:
        if exists(partial_filename):
            if getsize(partial_filename) == 0:
                logger.warning(f"removing zero-size corrupted ECOSTRESS file: {partial_filename}")
                os.remove(partial_filename)
        raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}\n{e}")

    # Check if the partial file was not created, raise an exception
    if not exists(partial_filename):
        raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}")
    # Check if the partial file is zero-size, remove it and raise an exception
    elif exists(partial_filename) and getsize(partial_filename) == 0:
        logger.warning(f"removing zero-size corrupted ECOSTRESS file: {partial_filename}")
        os.remove(partial_filename)
        raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}")

    # Move the partial file to the final filename
    os.replace(partial_filename, abs_filename)

    # Verify if the final file exists, if not, raise an exception
    if not exists(abs_filename):
        raise ECOSTRESSDownloadFailed(f"failed to download file: {filename}")

    logger.info(f"completed download: {cl.file(filename)}")
    return filename
