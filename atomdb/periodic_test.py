from enum import IntEnum
from numbers import Integral
import tables as pt
import numpy as np
from importlib_resources import files


__all__ = [
    "PROPERTY_NAME_MAP",
    "get_scalar_data",
    "element_symbol_map",
    "ElementAttr",
]


class ElementAttr(IntEnum):
    atnum = 0
    name = 1


elements_hdf5_file = files("atomdb.data").joinpath("elements_data.h5")
ELEMENTS_H5FILE = pt.open_file(elements_hdf5_file, mode="r")

PROPERTY_NAME_MAP = {
    "atmass": "atmass",
    "cov_radius": "cov_radius",
    "vdw_radius": "vdw_radius",
    "at_radius": "at_radius",
    "polarizability": "polarizability",
    "dispersion_c6": "dispersion_c6",
    "dispersion": "dispersion_c6",  # fields in run
    "elem": "symbol",
    "atnum": "atnum",
    "name": "name",
    "mult": "mult",
}


def get_scalar_data(prop_name, atnum, nelec):
    """
    Get a scalar property value for a given element.

    Args:
        prop_name (str): Property name to retrieve.
        atnum (int): Atomic number of the element.
        nelec (int): Number of electrons in the element.

    Returns:
        int | float | str | dict[str, float] | None:
            - int, float, or str for single-valued properties.
            - dict for properties with multiple sources.
            - None
    """

    charge = atnum - nelec

    if charge != 0 and prop_name not in ["atmass", "elem", "atnum", "name"]:
        return None

    # get the element group
    element_group = f"/Elements/{atnum:03d}"

    table_name = PROPERTY_NAME_MAP[prop_name]
    table_path = f"{element_group}/{table_name}"

    # get the table node from the HDF5 file
    table = ELEMENTS_H5FILE.get_node(table_path)

    # Handle basic properties (single column --> no sources)
    if len(table.colnames) == 1 and table.colnames[0] == "value":
        value = table[0]["value"]
        # if the value is an int, return it as an int
        if isinstance(value, Integral):
            return int(value)
        # if the value is a string, decode from bytes
        elif isinstance(value, bytes):
            return value.decode("utf-8")
    else:
        # handle properties with multiple sources
        result = {}
        for row in table:
            source = row["source"].decode("utf-8")
            value = row["value"]
            # exclude none values
            if not np.isnan(value):
                result[source] = float(value)
        return result if result else None


def map_element_symbol():
    """
    Build a mapping of element symbols to their atomic number and name.

    Returns:
        dict[str, tuple[int, str]]:
            Dictionary mapping element symbol → (atomic_number, name).
    """
    element_symbol_map = {}
    for element_group in ELEMENTS_H5FILE.root.Elements:
        symbol = element_group.symbol[0]["value"].decode("utf-8").strip()
        atnum = element_group.atnum[0]["value"]
        name = element_group.name[0]["value"].decode("utf-8").strip()
        element_symbol_map[symbol] = (atnum, name)

    return element_symbol_map


element_symbol_map = map_element_symbol()
