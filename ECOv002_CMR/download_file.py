import os
from os import makedirs, system
from os.path import exists, getsize, dirname, abspath, expanduser

import logging
from shutil import move
import posixpath

import colored_logging as cl

from .exceptions import *
from .timer import Timer

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

    # Check if the file exists and is zero-size, if so, remove it
    if exists(expand_filename(filename)) and getsize(expand_filename(filename)) == 0:
        logger.warning(f"removing zero-size corrupted ECOSTRESS file: {filename}")
        os.remove(filename)

    # If the file already exists, log the information and return the filename
    if exists(expand_filename(filename)):
        logger.info(f"file already downloaded: {cl.file(filename)}")
        return filename

    # Log the download start information
    logger.info(f"downloading: {cl.URL(URL)} -> {cl.file(filename)}")
    directory = dirname(expand_filename(filename))
    makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    partial_filename = f"{expand_filename(filename)}.download"
    command = f'wget -c -O "{partial_filename}" "{URL}"'
    timer = Timer()
    system(command)  # Execute the download command
    logger.info(f"completed download in {cl.time(timer)} seconds: " + cl.file(filename))

    # Check if the partial file was not created, raise an exception
    if not exists(partial_filename):
        raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}")
    # Check if the partial file is zero-size, remove it and raise an exception
    elif exists(partial_filename) and getsize(partial_filename) == 0:
        logger.warning(f"removing zero-size corrupted ECOSTRESS file: {partial_filename}")
        os.remove(partial_filename)
        raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}")

    # Move the partial file to the final filename
    move(partial_filename, expand_filename(filename))

    # Verify if the final file exists, if not, raise an exception
    if not exists(expand_filename(filename)):
        raise ECOSTRESSDownloadFailed(f"failed to download file: {filename}")

    return filename
