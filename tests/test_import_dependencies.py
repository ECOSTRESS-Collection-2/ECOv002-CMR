import pytest

# List of dependencies
dependencies = [
    "colored_logging",
    "ECOv002_granules",
    "numpy",
    "pandas",
    "rasters",
    "requests",
    "sentinel_tiles"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
