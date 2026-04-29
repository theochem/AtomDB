import pytest
from importlib import import_module
from importlib_resources import files
import os

from atomdb import Species


TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])

@pytest.mark.slow
@pytest.mark.parametrize(
    "element, charge, mult, dataset", [("C", 0, 3, "orca")]
)
def test_compile(element, charge, mult, dataset):
    submodule = import_module(f"atomdb.datasets.orca.run")
    # Compile the Species instance
    species = submodule.run(element, charge, mult, 0, dataset, TEST_DATAPATH)
    assert isinstance(species, Species)


