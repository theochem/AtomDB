import pytest
import numpy as np
from atomdb.periodic import _num2sym, _sym2num, _name2num, _num2name, Element
from atomdb.utils import angstrom, amu


def test_num2sym():
    assert _num2sym[1] == "H"
    assert _num2sym[6] == "C"
    assert _num2sym[26] == "Fe"


def test_sym2num():
    assert _sym2num["H"] == 1
    assert _sym2num["C"] == 6
    assert _sym2num["Fe"] == 26


def _name2num():
    assert _sym2num["hydrogen"] == 1
    assert _sym2num["carbon"] == 6
    assert _sym2num["iron"] == 26

    assert _sym2num["Hydrogen"] == 1
    assert _sym2num["Carbon"] == 6
    assert _sym2num["Iron"] == 26


def _num2name():
    assert _sym2num[1] == "Hydrogen"
    assert _sym2num[6] == "Carbon"
    assert _sym2num[26] == "Iron"


def test_element_invalid_element():
    with pytest.raises(Exception):
        Element("InvalidElement")


def test_element_invalid_symbol():
    with pytest.raises(Exception):
        Element("AA")


def test_element_invalid_atnum():
    with pytest.raises(Exception):
        Element(-2)


def test_get_attributes():
    """
    Test getting the attributes of the atom for Hydrogen and Carbon.
    """

    # Hydrogen data
    h_data = {
        "atnum": 1,
        "atsym": "H",
        "atname": "Hydrogen",
        "group": 1,
        "period": 1,
        "cov_radius": {"cordero": 0.31 * angstrom, "bragg": np.nan, "slater": 0.25 * angstrom},
        "vdw_radius": {
            "bondi": 1.2 * angstrom,
            "truhlar": np.nan,
            "rt": 1.1 * angstrom,
            "batsanov": np.nan,
            "dreiding": 3.195 * angstrom / 2,
            "uff": 2.886 * angstrom / 2,
            "mm3": 1.62 * angstrom,
        },
        "at_radius": {"wc": 0.529192875 * angstrom, "cr": 0.53 * angstrom},
        "eneg": {"pauling": 2.2},
        "pold": {"crc": 0.666793 * angstrom**3, "chu": 4.5},
        "c6": {"chu": 6.499026705},
        "mass": {"stb": 1.007975 * amu},
    }

    # Carbon data
    c_data = {
        "atnum": 6,
        "atsym": "C",
        "atname": "Carbon",
        "group": 14,
        "period": 2,
        "cov_radius": {
            "cordero": 0.7445337366 * angstrom,
            "bragg": 0.77 * angstrom,
            "slater": 0.7 * angstrom,
        },
        "vdw_radius": {
            "bondi": 1.7 * angstrom,
            "truhlar": np.nan,
            "rt": 1.77 * angstrom,
            "batsanov": 1.7 * angstrom,
            "dreiding": 3.8983 * angstrom / 2,
            "uff": 3.851 * angstrom / 2,
            "mm3": 2.04 * angstrom,
        },
        "at_radius": {"wc": 0.62 * angstrom, "cr": 0.67 * angstrom},
        "eneg": {"pauling": 2.55},
        "pold": {"crc": 1.76 * angstrom**3, "chu": 12.0},
        "c6": {"chu": 46.6},
        "mass": {"stb": 12.0106 * amu},
    }

    # for each element, check if the attributes are as expected
    for element, data in [("H", h_data), ("C", c_data)]:
        atom = Element(data["atnum"])
        for attr, value in data.items():
            assert getattr(atom, attr) == value, f"{element} {attr} is not as expected."
