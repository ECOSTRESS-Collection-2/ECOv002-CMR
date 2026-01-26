"""
NASA Earthdata authentication utilities.

This module provides functions for setting up authenticated sessions
to access NASA Earthdata services.
"""

import os
from pathlib import Path
from typing import Optional
import requests
import logging

logger = logging.getLogger(__name__)


def setup_earthdata_session(
    username: Optional[str] = None,
    password: Optional[str] = None,
    require_auth: bool = True
) -> requests.Session:
    """
    Set up NASA Earthdata authenticated session.
    
    Credentials are sourced from (in order of priority):
    1. Parameters passed directly to this function
    2. Environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
    3. ~/.netrc file
    
    Parameters
    ----------
    username : str, optional
        NASA Earthdata username. If not provided, will check environment
        variables and .netrc file.
    password : str, optional
        NASA Earthdata password. If not provided, will check environment
        variables and .netrc file.
    require_auth : bool, default True
        If True, raises an exception when credentials cannot be found.
        If False, returns an unauthenticated session.
    
    Returns
    -------
    requests.Session
        Authenticated session object ready for NASA Earthdata requests.
    
    Raises
    ------
    RuntimeError
        If require_auth is True and credentials cannot be found.
    
    Examples
    --------
    Set up using environment variables:
    
    >>> import os
    >>> os.environ['EARTHDATA_USERNAME'] = 'myusername'
    >>> os.environ['EARTHDATA_PASSWORD'] = 'mypassword'
    >>> session = setup_earthdata_session()
    
    Or using .netrc file (~/.netrc):
    
    ```
    machine urs.earthdata.nasa.gov
        login myusername
        password mypassword
    ```
    
    >>> session = setup_earthdata_session()
    
    Or passing credentials directly:
    
    >>> session = setup_earthdata_session(username='myuser', password='mypass')
    """
    # Priority 1: Use provided credentials
    if not (username and password):
        # Priority 2: Check environment variables
        username = username or os.environ.get("EARTHDATA_USERNAME")
        password = password or os.environ.get("EARTHDATA_PASSWORD")
    
    # Priority 3: Check .netrc file
    if not (username and password):
        netrc_path = Path.home() / ".netrc"
        if netrc_path.exists():
            try:
                import netrc
                netrc_auth = netrc.netrc(str(netrc_path))
                auth = netrc_auth.authenticators("urs.earthdata.nasa.gov")
                if auth:
                    username, _, password = auth
                    logger.info("Credentials loaded from ~/.netrc")
            except Exception as e:
                logger.warning(f"Failed to read .netrc: {e}")
    
    # Create session
    session = requests.Session()
    
    if username and password:
        session.auth = (username, password)
        logger.info("NASA Earthdata authentication configured")
    elif require_auth:
        raise RuntimeError(
            "NASA Earthdata authentication required!\n\n"
            "Option 1: Create ~/.netrc with:\n"
            "    machine urs.earthdata.nasa.gov\n"
            "        login YOUR_USERNAME\n"
            "        password YOUR_PASSWORD\n\n"
            "Option 2: Set environment variables:\n"
            "    export EARTHDATA_USERNAME=YOUR_USERNAME\n"
            "    export EARTHDATA_PASSWORD=YOUR_PASSWORD\n\n"
            "Option 3: Pass credentials directly:\n"
            "    setup_earthdata_session(username='...', password='...')"
        )
    else:
        logger.warning("No NASA Earthdata credentials found, session is unauthenticated")
    
    return session
