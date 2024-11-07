import os
from os import makedirs, system
from os.path import exists, getsize, dirname, abspath

import logging
from shutil import move
import posixpath

import colored_logging as cl

from .exceptions import *
from .timer import Timer

logger = logging.getLogger(__name__)

def download_file(
          URL: str, 
          filename: str = None,
          granule_directory: str = None) -> str:
        if filename is None:
             filename = abspath(posixpath.basename(URL))

        if exists(filename) and getsize(filename) == 0:
            logger.warning(f"removing zero-size corrupted ECOSTRESS file: {filename}")
            os.remove(filename)

        if exists(filename):
            logger.info(f"file already downloaded: {cl.file(filename)}")
            return filename

        logger.info(f"downloading: {cl.URL(URL)} -> {cl.file(filename)}")
        directory = dirname(filename)
        makedirs(directory, exist_ok=True)
        partial_filename = f"{filename}.download"
        command = f'wget -c -O "{partial_filename}" "{URL}"'
        timer = Timer()
        system(command)
        logger.info(f"completed download in {cl.time(timer)} seconds: " + cl.file(filename))

        if not exists(partial_filename):
            raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}")
        elif exists(partial_filename) and getsize(partial_filename) == 0:
            logger.warning(f"removing zero-size corrupted ECOSTRESS file: {partial_filename}")
            os.remove(partial_filename)
            raise ECOSTRESSDownloadFailed(f"unable to download URL: {URL}")

        move(partial_filename, filename)

        if not exists(filename):
            raise ECOSTRESSDownloadFailed(f"failed to download file: {filename}")

        return filename
