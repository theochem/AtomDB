from os import path

import numpy as np

from scipy.interpolate import interp1d


__all__ = [
    "cubic_interpolation",
    "get_element",
    "get_datafile",
    "get_raw_datafile",
]


DATAPATH = path.abspath(path.join(path.dirname(__file__), "data/"))


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


class interp1d_log(interp1d):
    r"""Interpolate over a 1-D grid."""

    def __init__(self, x, y, **kwargs):
        r"""Initialize the interp1d_log instance."""
        interp1d.__init__(self, x, np.log(y), **kwargs)

    def __call__(self, x):
        r"""Compute the interpolation at some x-values."""
        return np.exp(interp1d.__call__(self, x))


def cubic_interpolation(x, y, log=False):
    r"""Create an interpolated cubic spline for the given data."""
    return (interp1d_log if log else interp1d)(
        x, y, kind="cubic", copy=True, fill_value="extrapolate", assume_sorted=True,
    )


def get_element(elem):
    r""" """
    return elem if isinstance(elem, str) else ELEMENTS[elem]


def get_datafile(name):
    r""" """
    return path.abspath(path.join(DATAPATH, name))


def get_raw_datafile(suffix, dataset, elem, basis, nelec, nspin):
    r""" """
    return get_datafile(f"raw/{dataset}/{get_element(elem)}_N{nelec}_S{nspin}_{basis}.{suffix}")
