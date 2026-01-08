from typing import Tuple

def product_name_from_granule_ID(granule_ID: str) -> str:
    """
    Extracts the product name from an ECOSTRESS granule ID.

    Args:
        granule_ID: The granule identifier.

    Returns:
        The product name extracted from the granule ID.
    """
    return "_".join(granule_ID.split("_")[1:3])

class GranuleID:
    def __init__(self, granule_ID: str):
        """
        Initializes an ECOSTRESS GranuleID object.

        Args:
            granule_ID: The granule identifier.
        """
        self.granule_ID = granule_ID

        self.product = product_name_from_granule_ID(granule_ID)

        self.orbit = None
        self.scene = None

        if self.product == "L2T_STARS":
            self._parse_STARS_granule_ID()
        else:
            self._parse_granule_ID()

    def _parse_granule_ID(self):
        """
        Parse orbit, scene, and tile information from a standard ECOSTRESS granule ID.
        
        Extracts orbit number, scene number, and tile identifier from the granule ID
        by splitting on underscores and parsing the appropriate segments.
        """
        self.orbit = int(self.granule_ID.split("_")[3])
        self.scene = int(self.granule_ID.split("_")[4])
        self.tile = self.granule_ID.split("_")[5]

    def _parse_STARS_granule_ID(self):
        """
        Parse tile information from an ECOSTRESS L2T_STARS granule ID.
        
        STARS products have a different granule ID structure and only require
        extracting the tile identifier.
        """
        self.tile = self.granule_ID.split("_")[3]

    def __str__(self) -> str:
        """
        Return string representation of the granule ID.
        
        Returns:
            The granule identifier string.
        """
        return self.granule_ID

    def __repr__(self) -> str:
        """
        Return official string representation of the granule ID.
        
        Returns:
            The granule identifier string.
        """
        return self.granule_ID
    
    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying granule_ID string.
        
        Allows GranuleID objects to respond to string methods like split, replace, etc.
        
        Args:
            attr: The attribute name to access.
            
        Returns:
            The attribute from the granule_ID string.
        """
        return getattr(self.granule_ID, attr)
