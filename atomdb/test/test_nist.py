import pytest
import os
import numpy as np
from importlib_resources import files
from atomdb.api import load
from atomdb.utils import angstrom, amu, cminv, ev

# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


TEST_CASES_MAKE_PROMOLECULE = [
    pytest.param(
        {
            "dataset": "nist",
            "elem": "H",
            "atnum": 1,
            "obasis_name": None,
            "nelec": 1,
            "nspin": 1,
            "nexc": 0,
            "atmass": 1.007975 * amu,
            # "cov_radius": {"cordero": 0.31 * angstrom, "bragg": np.nan, "slater": 0.25 * angstrom},
            # "vdw_radius": {
            #     "bondi": 1.2 * angstrom,
            #     "truhlar": np.nan,
            #     "rt": 1.1 * angstrom,
            #     "batsanov": np.nan,
            #     "dreiding": 3.195 * angstrom / 2,
            #     "uff": 2.886 * angstrom / 2,
            #     "mm3": 1.62 * angstrom,
            # },
            "at_radius": {"wc": 0.529192875 * angstrom, "cr": 0.53 * angstrom},
            "polarizability": {"crc": 0.666793 * angstrom**3, "chu": 4.5},
            "dispersion_c6": {"chu": 6.499026705},
            "energy": -109678.77174307 * cminv,
            "ip": 13.598443 * ev,
            "mu": -7.18 * ev,
            "eta": 12.84 * ev,
        },
        id="H neutral",
    ),
    pytest.param(
        {
            "dataset": "nist",
            "elem": "C",
            "atnum": 6,
            "obasis_name": None,
            "nelec": 6,
            "nspin": 2,
            "nexc": 0,
            "atmass": 12.0106 * amu,
            "cov_radius": {
                "cordero": 0.7445337366 * angstrom,
                "bragg": 0.77 * angstrom,
                "slater": 0.7 * angstrom,
            },
            # "vdw_radius": {
            #     "bondi": 1.7 * angstrom,
            #     "truhlar": np.nan,
            #     "rt": 1.77 * angstrom,
            #     "batsanov": 1.7 * angstrom,
            #     "dreiding": 3.8983 * angstrom / 2,
            #     "uff": 3.851 * angstrom / 2,
            #     "mm3": 2.04 * angstrom,
            # },
            "at_radius": {"wc": 0.62 * angstrom, "cr": 0.67 * angstrom},
            "polarizability": {"crc": 1.76 * angstrom**3, "chu": 12.0},
            "dispersion_c6": {"chu": 46.6},
            "energy": -8308396.1899999995 * cminv,
            "ip": 11.2603 * ev,
            "mu": -6.26 * ev,
            "eta": 10 * ev,
        },
        id="C neutral",
    ),
    pytest.param(
        {
            "dataset": "nist",
            "elem": "C",
            "atnum": 6,
            "obasis_name": None,
            "nelec": 5,
            "nspin": 1,
            "nexc": 0,
            "atmass": 12.0106 * amu,
            "cov_radius": None,
            "vdw_radius": None,
            "at_radius": None,
            "polarizability": None,
            "dispersion_c6": None,
            "energy": -8217575.77 * cminv,
            "ip": 24.3833 * ev,
            "mu": -17.82 * ev,
            "eta": 13.12 * ev,
        },
        id="C cation",
    ),
    pytest.param(
        {
            "dataset": "nist",
            "elem": "C",
            "atnum": 6,
            "obasis_name": None,
            "nelec": 7,
            "nspin": 3,
            "nexc": 0,
            "atmass": 12.0106 * amu,
            "cov_radius": None,
            "vdw_radius": None,
            "at_radius": None,
            "polarizability": None,
            "dispersion_c6": None,
            "energy": None,
            "ip": 1.262118 * ev,
            "mu": 2.08 * ev,
            "eta": 8.02 * ev,
        },
        id="C anion",
    ),
]


@pytest.mark.parametrize("case", TEST_CASES_MAKE_PROMOLECULE)
def test_nist_data(case):
    """
    Test getting the attributes of the atom for Hydrogen and Carbon.
    """
    elem = case.get("elem")
    charge = case.get("atnum") - case.get("nelec")
    mult = case.get("nspin") + 1
    sp = load(elem, charge, mult, dataset="nist", datapath=TEST_DATAPATH)

    for attr, value in case.items():
        assert getattr(sp, attr) == value, f"{elem} {attr} is not as expected."
