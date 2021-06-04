# This file is part of AtomDB.
#
# AtomDB is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# AtomDB is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with AtomDB. If not, see <http://www.gnu.org/licenses/>.

r"""AtomDB utilities."""


from json import JSONEncoder, dumps

from os.path import abspath, join

from sys import platform

from numpy import ndarray, exp, log

from scipy.interpolate import interp1d

from msgpack import Packer, Unpacker

from atomdb.config import DATAPATH


__all__ = [
    "get_element_number",
    "get_element_symbol",
    "get_file",
    "get_data_file",
    "get_raw_data_file",
    "ndarray_to_bytes",
    "pack_msg",
    "unpack_msg",
    "dump_json",
    "interp1d",
    "interp1d_log",
    "cubic_interp",
]


ELEMENTS = (
    "\0", "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
r"""Tuple of the symbols for each of the 118 elements. The zeroth element is a placeholder."""


def get_element_number(elem):
    r"""Return the element number from the given element symbol."""
    return ELEMENTS.index(elem) if isinstance(elem, str) else elem


def get_element_symbol(elem):
    r"""Return the element symbol from the given element number."""
    return elem if isinstance(elem, str) else ELEMENTS[elem]


def get_file(name):
    r"""Get a file from the `DATAPATH`."""
    return abspath(join(DATAPATH, name))


def get_data_file(dataset, elem, nelec, nspin, nexc, suffix):
    r"""Get a compiled data file from the `DATAPATH`."""
    return get_file(
        f"{dataset}/data/{get_element_symbol(elem)}_N{nelec}_S{nspin}_E{nexc}.{suffix}"
    )


def get_raw_data_file(dataset, elem, nelec, nspin, nexc, suffix):
    r"""Get a raw data file from the `DATAPATH`."""
    return get_file(
        f"{dataset}/raw_data/{get_element_symbol(elem)}_N{nelec}_S{nspin}_E{nexc}_{suffix}"
    )


if platform == 'darwin':
    def ndarray_to_bytes(array):
        r"""Convert a numpy.ndarray instance to bytes."""
        return array.tobytes()
else:
    def ndarray_to_bytes(array):
        r"""Convert a numpy.ndarray instance to bytes."""
        return array.data if array.flags['C_CONTIGUOUS'] else array.tobytes()


def pack_msg(obj):
    r"""Pack an object to MessagePack binary format."""
    return Packer(use_bin_type=True).pack(obj)


def unpack_msg(obj):
    r"""Unpack an object from MessagePack binary format."""
    return Unpacker(obj, use_list=False, strict_map_key=True).unpack()


class NDEncoder(JSONEncoder):
    r"""JSON encoder that handles `numpy.ndarray` objects."""

    def default(self, obj):
        r"""Default encode function."""
        return obj.tolist() if isinstance(obj, ndarray) else JSONEncoder.default(self, obj)


def dump_json(obj):
    r"""Return the JSON representation of a species entry."""
    return dumps(obj.todict(), cls=NDEncoder, sort_keys=True, indent=2)


class interp1d_log(interp1d):
    r"""Interpolate over a 1-D grid."""

    def __init__(self, x, y, **kwargs):
        r"""Initialize the interp1d_log instance."""
        interp1d.__init__(self, x, log(y), **kwargs)

    def __call__(self, x):
        r"""Compute the interpolation at some x-values."""
        return exp(interp1d.__call__(self, x))


def cubic_interp(x, y, log=False):
    r"""Create an interpolated cubic spline for the given data."""
    return (interp1d_log if log else interp1d)(
        x, y, kind="cubic", copy=False, fill_value="extrapolate", assume_sorted=True,
    )
