import pytest

from atomdb import Element, element_number, element_symbol, element_name

from atomdb.utils import ANGSTROM, AMU


def testelement_symbol():
    assert element_symbol(1) == "H"
    assert element_symbol(6) == "C"
    assert element_symbol(26) == "Fe"


def test_sym2num():
    assert element_number("H") == 1
    assert element_number("C") == 6
    assert element_number("Fe") == 26

    assert element_number("hydrogen") == 1
    assert element_number("carbon") == 6
    assert element_number("iron") == 26

    assert element_number("Hydrogen") == 1
    assert element_number("Carbon") == 6
    assert element_number("Iron") == 26


def test__num2name():
    assert element_name(1) == "Hydrogen"
    assert element_name(6) == "Carbon"
    assert element_name(26) == "Iron"


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
        "symbol": "H",
        "name": "Hydrogen",
        "group": 1,
        "period": 1,
        "cov_radius": {
            "cordero": 0.31 * ANGSTROM,
            "slater": 0.25 * ANGSTROM,
        },
        "vdw_radius": {
            "bondi": 1.2 * ANGSTROM,
            "rt": 1.1 * ANGSTROM,
            "dreiding": 3.195 * ANGSTROM / 2,
            "uff": 2.886 * ANGSTROM / 2,
            "mm3": 1.62 * ANGSTROM,
        },
        "at_radius": {"wc": 0.529192875 * ANGSTROM, "cr": 0.53 * ANGSTROM},
        "eneg": {"pauling": 2.2},
        "pold": {"crc": 0.666793 * ANGSTROM**3, "chu": 4.5},
        "c6": {"chu": 6.499026705},
        "mass": {"stb": 1.007975 * AMU, "nist": 1.007825032 * AMU},
    }

    # Carbon data
    c_data = {
        "atnum": 6,
        "symbol": "C",
        "name": "Carbon",
        "group": 14,
        "period": 2,
        "cov_radius": {
            "cordero": 0.7445337366 * ANGSTROM,
            "bragg": 0.77 * ANGSTROM,
            "slater": 0.7 * ANGSTROM,
        },
        "vdw_radius": {
            "bondi": 1.7 * ANGSTROM,
            "rt": 1.77 * ANGSTROM,
            "batsanov": 1.7 * ANGSTROM,
            "dreiding": 3.8983 * ANGSTROM / 2,
            "uff": 3.851 * ANGSTROM / 2,
            "mm3": 2.04 * ANGSTROM,
        },
        "at_radius": {"wc": 0.62 * ANGSTROM, "cr": 0.67 * ANGSTROM},
        "eneg": {"pauling": 2.55},
        "pold": {"crc": 1.76 * ANGSTROM**3, "chu": 12.0},
        "c6": {"chu": 46.6},
        "mass": {"stb": 12.0106 * AMU, "nist": 12.000 * AMU},
    }

    # for each element, check if the attributes are as expected
    for element, data in [("H", h_data), ("C", c_data)]:
        atom = Element(data["atnum"])
        for attr, value in data.items():
            assert getattr(atom, attr) == value, f"{element} {attr} is not as expected."
