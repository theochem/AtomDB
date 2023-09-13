# -*- coding: utf-8 -*-
# GOpt is a Python library for optimizing molecular structures
# and determining chemical reaction pathways.
#
# Copyright (C) 2020-2023 The QC-Devs Team
#
# This file is part of GOpt.
#
# GOpt is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# GOpt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Properties for the elemants of the Periodic Table.

This module contains dictionaries containing the van der Waals and covalent
radius for the elements of the periodic table.It also loads from IOdata
the dictionaries necesary to convert from Z to element symbol and viceversa.

The dictionaries loaded in the module are:

    num2sym (fom iodata):
        Mode of use: num2sym[Z] returns the chemical symbol corresponding
        to the chemical element of atomic charge Z.

    sym2num (from iodata):
        Mode of use: sym2num[Sym] returns the atomic charge Z corresponding
        to the chemical element of chemical symbol Sym

    num2vdw:
        Mode of use: num2vdw[Z] returns the van der Waals radius
        corresponding to the chemical element of atomic charge Z.

    num2cov:
        Mode of use: num2cov[Z] returns the covalent radius
        corresponding to the chemical element of atomic charge Z.

    sym2vdw:
        Mode of use: sym2vdw[Z] returns the van der Waals radius
        corresponding to the chemical element of symbol sym.

    sym2cov:
        Mode of use: sym2cov[Z] returns the covalent radius
        corresponding to the chemical element of symbol sym.

The values for the van der Waals radii were taken from:
    S. Alvares, Dalton Trans., 2013,42, 8617-8636 https://doi.org/10.1039/C3DT50599E

The values for the covalent radii were taken from:
    B. Cordero,.V. Gómez, A. E. Platero-Prats, M. Revés,J. Echeverría,
    E. Cremades, F Barragána, S. Alvarez, Dalton Trans., 2008, 2832-2838
    https://doi.org/10.1039/B801115J


Relevant notes:

    - The C covalent radius value was taken as reported for Csp3

    - For Mn (Z=25), Fe (Z=26), and Co (Z=27) the covalent radii values were
      taken as reported for the low spin electronic configuration.

      The vdw radii values for Z = 61, 84, 85, 86, 87, 88, 8, 99 are missing in
      the original dataset. Sensible default values were given to each of
      these cases. For Z = 61 the value corresponding to Z = 62 (2.9 Angstrom)
      was used. For Z = 84, 85, 86, 87 and 88 the average value of Z = 83 and
      89 was used (2.67 Angstrom). For 99 the value corresponding to
      Z = 98 (2.7 Angstrom) was used.

      The covalent radii values for Z = 97, 98, 99 are missing in the original
      dataset. For these cases the value corresponding to Z = 96 (1.69 Angstrom)
      was used.
"""


from typing import Dict
from iodata.utils import angstrom
from iodata.periodic import num2sym, sym2num


__all__ = [
    "num2sym",
    "sym2num",
    "num2vdw",
    "num2cov",
    "sym2vdw",
    "sym2cov",
    "num2vdw_check",
    "sym2vdw_check",
    "num2cov_check",
    "sym2cov_check",
    "num2Ar",
    "sym2Ar",
]

"""
The missing wdw radii in the database were populated as following:
  - For Z = 61 the value corresponding to Z = 62 (2.9 Angstrom) was used.
  - For Z = 84, 85, 86, 87 and 88 the average value of Z = 83 and 89 (2.67 Angstrom) was used .
  - For 99 the value corresponding toZ = 98 (2.7 Angstrom) was used.

The missing cov radii in the database were populated as following:
  -  For Z = 97, 98, 99 the value corresponding to Z = 96 (1.69 Angstrom) was used.
"""

_missing_vdw = [61, 84, 85, 86, 87, 88, 99]
_missing_cov = [97, 98, 99]

num2vdw: Dict[int, float] = {
    1: 1.2,
    2: 1.43,
    3: 2.12,
    4: 1.98,
    5: 1.91,
    6: 1.77,
    7: 1.66,
    8: 1.5,
    9: 1.46,
    10: 1.58,
    11: 2.5,
    12: 2.51,
    13: 2.25,
    14: 2.19,
    15: 1.9,
    16: 1.89,
    17: 1.82,
    18: 1.83,
    19: 2.73,
    20: 2.62,
    21: 2.58,
    22: 2.46,
    23: 2.42,
    24: 2.45,
    25: 2.45,
    26: 2.44,
    27: 2.4,
    28: 2.4,
    29: 2.38,
    30: 2.39,
    31: 2.32,
    32: 2.29,
    33: 1.88,
    34: 1.82,
    35: 1.86,
    36: 2.25,
    37: 3.21,
    38: 2.84,
    39: 2.75,
    40: 2.52,
    41: 2.56,
    42: 2.45,
    43: 2.44,
    44: 2.46,
    45: 2.44,
    46: 2.15,
    47: 2.53,
    48: 2.49,
    49: 2.43,
    50: 2.42,
    51: 2.47,
    52: 1.99,
    53: 2.04,
    54: 2.06,
    55: 3.48,
    56: 3.03,
    57: 2.98,
    58: 2.88,
    59: 2.92,
    60: 2.95,
    61: 2.9,
    62: 2.9,
    63: 2.87,
    64: 2.83,
    65: 2.79,
    66: 2.87,
    67: 2.81,
    68: 2.83,
    69: 2.79,
    70: 2.8,
    71: 2.74,
    72: 2.63,
    73: 2.53,
    74: 2.57,
    75: 2.49,
    76: 2.48,
    77: 2.41,
    78: 2.29,
    79: 2.32,
    80: 2.45,
    81: 2.47,
    82: 2.6,
    83: 2.54,
    84: 2.67,
    85: 2.67,
    86: 2.67,
    87: 2.67,
    88: 2.67,
    89: 2.8,
    90: 2.93,
    91: 2.88,
    92: 2.82,
    93: 2.81,
    94: 2.83,
    95: 3.05,
    96: 3.4,
    97: 3.05,
    98: 2.7,
    99: 2.7,
}

num2cov: Dict[int, float] = {
    1: 0.31,
    2: 0.28,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    19: 2.03,
    20: 1.76,
    21: 1.7,
    22: 1.6,
    23: 1.53,
    24: 1.39,
    25: 1.39,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.2,
    33: 1.19,
    34: 1.2,
    35: 1.2,
    36: 1.16,
    37: 2.2,
    38: 1.95,
    39: 1.9,
    40: 1.75,
    41: 1.64,
    42: 1.54,
    43: 1.47,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    49: 1.42,
    50: 1.39,
    51: 1.39,
    52: 1.38,
    53: 1.39,
    54: 1.4,
    55: 2.44,
    56: 2.15,
    57: 2.07,
    58: 2.04,
    59: 2.03,
    60: 2.01,
    61: 1.99,
    62: 1.98,
    63: 1.98,
    64: 1.96,
    65: 1.94,
    66: 1.92,
    67: 1.92,
    68: 1.89,
    69: 1.9,
    70: 1.87,
    71: 1.87,
    72: 1.75,
    73: 1.7,
    74: 1.62,
    75: 1.51,
    76: 1.44,
    77: 1.41,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    81: 1.45,
    82: 1.46,
    83: 1.48,
    84: 1.4,
    85: 1.5,
    86: 1.5,
    87: 2.6,
    88: 2.21,
    89: 2.15,
    90: 2.06,
    91: 2.0,
    92: 1.96,
    93: 1.9,
    94: 1.87,
    95: 1.8,
    96: 1.69,
    97: 1.69,
    98: 1.69,
    99: 1.69,
}

# Relative atomic mass (Ar) of the elements.
#
#  The data was taken from the following source:
# Prohaska, T., Irrgeher, J., Benefield, J., Böhlke, J., Chesson, L., Coplen, T., Ding, T.,
# Dunn, P., Gröning, M., Holden, N., Meijer, H., Moossen, H., Possolo, A., Takahashi, Y., Vogl, J.,
# Walczyk, T., Wang, J., Wieser, M., Yoneda, S., Zhu, X. and Meija, J. (2022) Standard atomic
# weights of the elements 2021 (IUPAC Technical Report). Pure and Applied Chemistry, Vol. 94
# (Issue 5), pp. 573-600. https://doi.org/10.1515/pac-2019-0603

# For synthetic elements (marked by "#NIST" followed by the selected isotope number if there are
# several options). The reported atomic mass corresponds to the longest-lived isotope. In these
# cases the data was taken from the NIST database:
# https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl

num2mass: Dict[int, float] = {
    1: 1.007975,
    2: 4.002602,
    3: 6.9675,
    4: 9.0121831,
    5: 10.8135,
    6: 12.0106,
    7: 14.006855,
    8: 15.9994,
    9: 18.998403162,
    10: 20.1797,
    11: 22.98976928,
    12: 24.3055,
    13: 26.9815384,
    14: 28.085,
    15: 30.973761998,
    16: 32.0675,
    17: 35.4515,
    18: 39.8775,
    19: 39.0983,
    20: 40.078,
    21: 44.955907,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938043,
    26: 55.845,
    27: 58.933194,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.921595,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.905838,
    40: 91.224,
    41: 92.90637,
    42: 95.95,
    43: 97.9072124,  # NIST [98]
    44: 101.07,
    45: 102.90549,
    46: 106.42,
    47: 107.8682,
    48: 112.414,
    49: 114.818,
    50: 118.71,
    51: 121.76,
    52: 127.6,
    53: 126.90447,
    54: 131.293,
    55: 132.90545196,
    56: 137.327,
    57: 138.90547,
    58: 140.116,
    59: 140.90766,
    60: 144.242,
    61: 144.9127559,  # NIST [145]
    62: 150.36,
    63: 151.964,
    64: 157.25,
    65: 158.925354,
    66: 162.5,
    67: 164.930329,
    68: 167.259,
    69: 168.934219,
    70: 173.045,
    71: 174.9668,
    72: 178.486,
    73: 180.94788,
    74: 183.84,
    75: 186.207,
    76: 190.23,
    77: 192.217,
    78: 195.084,
    79: 196.96657,
    80: 200.592,
    81: 204.3835,
    82: 207.04,
    83: 208.9804,
    84: 208.9824308,  # NIST [209]
    85: 209.9871479,  # NIST [210]
    86: 222.0175782,  # NIST [222]
    87: 223.0197360,  # NIST
    88: 226.0254103,  # NIST [226]
    89: 227.0277523,  # NIST
    90: 232.0377,
    91: 231.03588,
    92: 238.02891,
    93: 237.0481736,  # NIST [237]
    94: 244.0642053,  # NIST [244]
    95: 243.0613813,  # NIST [243]
    96: 247.0703541,  # NIST [247]
    97: 247.0703073,  # NIST [247]
    98: 251.0795886,  # NIST [251]
    99: 252.082980,  # NIST
    100: 257.0951061,  # NIST
    101: 258.0984315,  # NIST [258]
    102: 259.10103,  # NIST
    103: 262.10961,  # NIST
    104: 267.12179,  # NIST
    105: 268.12567,  # NIST
    106: 271.13393,  # NIST
    107: 272.13826,  # NIST
    108: 270.13429,  # NIST
    109: 276.15159,  # NIST
    110: 281.16451,  # NIST
    111: 280.16514,  # NIST
    112: 285.17712,  # NIST
    113: 284.17873,  # NIST
    114: 289.19042,  # NIST
    115: 288.19274,  # NIST
    116: 293.20449,  # NIST
    117: 292.20746,  # NIST
    118: 294.21392,  # NIST
}


# The original data (in angstrom) is converted to bohr using iodata conversion factor
num2vdw: Dict[int, float] = {key: value * angstrom for key, value in num2vdw.items()}
num2cov: Dict[int, float] = {key: value * angstrom for key, value in num2cov.items()}

sym2vdw: Dict[str, int] = {sym: num2vdw[Z] for sym, Z in sym2num.items() if Z <= 99}
sym2cov: Dict[str, int] = {sym: num2cov[Z] for sym, Z in sym2num.items() if Z <= 99}

# conversion from symbol to atomic mass
sym2mass: Dict[str, float] = {sym: num2mass[Z] for sym, Z in sym2num.items()}


# Functions that check when an element used is missing in the databases
def num2vdw_check(Z: int) -> None:
    if Z in _missing_vdw:
        print(
            f"Warning: Element {num2sym[Z]}(Z = {Z}) is not on the vdW radii database. Using {num2vdw[Z]} as vdW radius."
        )


def sym2vdw_check(sym: str) -> None:
    num2vdw_check(sym2num[sym])


def num2cov_check(Z: int) -> None:
    if Z in _missing_cov:
        print(
            f"Warning: Element {num2sym[Z]}(Z = {Z}) is not on the cov radii database. Using {num2cov[Z]} as cov radius."
        )


def sym2cov_check(sym: str) -> None:
    num2cov_check(sym2num[sym])
