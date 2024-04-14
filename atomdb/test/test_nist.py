from importlib_resources import files

import os

import pytest

from atomdb import load
from atomdb.utils import ANGSTROM, AMU, CMINV, EV

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
            "atmass": 1.007975 * AMU,
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
            "at_radius": {"wc": 0.529192875 * ANGSTROM, "cr": 0.53 * ANGSTROM},
            "polarizability": {"crc": 0.666793 * ANGSTROM**3, "chu": 4.5},
            "dispersion_c6": {"chu": 6.499026705},
            "energy": -109678.77174307 * CMINV,
            "ip": 13.598443 * EV,
            "mu": -7.18 * EV,
            "eta": 12.84 * EV,
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
            "atmass": 12.0106 * AMU,
            "cov_radius": {
                "cordero": 0.7445337366 * ANGSTROM,
                "bragg": 0.77 * ANGSTROM,
                "slater": 0.7 * ANGSTROM,
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
            "at_radius": {"wc": 0.62 * ANGSTROM, "cr": 0.67 * ANGSTROM},
            "polarizability": {"crc": 1.76 * ANGSTROM**3, "chu": 12.0},
            "dispersion_c6": {"chu": 46.6},
            "energy": -8308396.1899999995 * CMINV,
            "ip": 11.2603 * EV,
            "mu": -6.26 * EV,
            "eta": 10 * EV,
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
            "atmass": 12.0106 * AMU,
            "cov_radius": None,
            "vdw_radius": None,
            "at_radius": None,
            "polarizability": None,
            "dispersion_c6": None,
            "energy": -8217575.77 * CMINV,
            "ip": 24.3833 * EV,
            "mu": -17.82 * EV,
            "eta": 13.12 * EV,
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
            "atmass": 12.0106 * AMU,
            "cov_radius": None,
            "vdw_radius": None,
            "at_radius": None,
            "polarizability": None,
            "dispersion_c6": None,
            "energy": None,
            "ip": 1.262118 * EV,
            "mu": 2.08 * EV,
            "eta": 8.02 * EV,
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
        data_value = getattr(sp, attr)
        # try:
        #     data_value = getattr(sp, attr)
        # except AttributeError:
        #     data_value = getattr(sp._data, attr)
        assert data_value == value, f"{elem} {attr} is not as expected."
