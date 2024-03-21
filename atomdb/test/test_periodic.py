import pytest
import numpy as np
from atomdb.periodic import num2sym, sym2num, name2num, num2name, Element


def test_num2sym():
    assert num2sym[1] == "H"
    assert num2sym[6] == "C"
    assert num2sym[26] == "Fe"


def test_sym2num():
    assert sym2num["H"] == 1
    assert sym2num["C"] == 6
    assert sym2num["Fe"] == 26


def name2num():
    assert sym2num["hydrogen"] == 1
    assert sym2num["carbon"] == 6
    assert sym2num["iron"] == 26

    assert sym2num["Hydrogen"] == 1
    assert sym2num["Carbon"] == 6
    assert sym2num["Iron"] == 26


def num2name():
    assert sym2num[1] == "Hydrogen"
    assert sym2num[6] == "Carbon"
    assert sym2num[26] == "Iron"


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
    # will test getting the attributes of the atom for Hydrogen and Carbon

    atom = Element(1)
    assert atom.atnum == 1
    assert atom.atsym == "H"
    assert atom.atname == "Hydrogen"

    h_data = {
        "atnum": 1,
        "atsym": "H",
        "atname": "Hydrogen",
        "group": 1,
        "period": 1,
        "cov_radius": {"cordero": 0.31, "bragg": np.nan, "slater": 0.25},
        "vdw_radius": {
            "bondi": 1.2,
            "truhlar": np.nan,
            "rt": 1.1,
            "batsanov": np.nan,
            "dreiding": 3.195,
            "uff": 2.886,
            "mm3": 1.62,
        },
        "at_radius": {"wc": 0.529192875, "cr": 0.53},
        "eneg": {"pauling": 2.2},
        "pold": {"crc": 0.666793, "chu": 4.5},
        "c6": {"chu": 6.499026705},
        "mass": {"stb": 1.007975},
    }
    for i in h_data:
        assert getattr(atom, i) == h_data[i]

    atom = Element(6)
    c_data = {
        "atnum": 6,
        "atsym": "C",
        "atname": "Carbon",
        "group": 14,
        "period": 2,
        "cov_radius": {"cordero": 0.7445337366, "bragg": 0.77, "slater": 0.7},
        "vdw_radius": {
            "bondi": 1.7,
            "truhlar": np.nan,
            "rt": 1.77,
            "batsanov": 1.7,
            "dreiding": 3.8983,
            "uff": 3.851,
            "mm3": 2.04,
        },
        "at_radius": {"wc": 0.62, "cr": 0.67},
        "eneg": {"pauling": 2.55},
        "pold": {"crc": 1.76, "chu": 12.0},
        "c6": {"chu": 46.6},
        "mass": {"stb": 12.0106},
    }

    for i in c_data:
        assert getattr(atom, i) == c_data[i]
