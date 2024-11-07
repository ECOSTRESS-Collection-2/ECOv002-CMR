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
        self.orbit = int(self.granule_ID.split("_")[3])
        self.scene = int(self.granule_ID.split("_")[4])
        self.tile = self.granule_ID.split("_")[5]

    def _parse_STARS_granule_ID(self):
        self.tile = self.granule_ID.split("_")[3]

    def __str__(self) -> str:
        return self.granule_ID

    def __repr__(self) -> str:
        return self.granule_ID
    
    def __getattr__(self, attr):
        return getattr(self.granule_ID, attr)
