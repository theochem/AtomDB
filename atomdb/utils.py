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

r"""Constants and utility functions."""

import csv

from os import path, environ

import numpy as np

import h5py as h5

from scipy import constants


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_DATAPATH",
    "MODULE_DATAPATH",
    "CONVERTOR_TYPES",
    "MULTIPLICITIES",
    "generate_mult_csv",
]


DEFAULT_DATASET = "slater"
r"""Default dataset to query."""


DEFAULT_DATAPATH = environ.get(
    "ATOMDB_DATAPATH",
    path.join(path.dirname(__file__), "datasets"),
)
r"""The path for AtomDB datasets."""


MODULE_DATAPATH = path.join(path.dirname(__file__), "data")
r"""The path for AtomDB data files."""


HDF5_NIST_FILE = path.join(MODULE_DATAPATH, "database_beta_1.3.0.h5")
r"""The HDF5 file containing the raw NIST data."""


MULT_TABLE_CSV = path.join(MODULE_DATAPATH, "multiplicities_table.csv")
r"""The CSV file containing the multiplicities of species."""


ANGSTROM = 100 * constants.pico * constants.m_e * constants.c * constants.alpha / constants.hbar
r"""Angstrom (:math:`\text{Å}`)."""


AMU = constants.gram / (constants.Avogadro * constants.m_e)
r"""Atomic mass unit (:math:`\text{a.m.u.}`)."""


CMINV = 2 * constants.centi * constants.Rydberg
r"""Inverse centimetre (:math:`\text{cm}^{-1}`)."""


EV = constants.eV / (2 * constants.Rydberg * constants.h * constants.c)
r"""Electron volt (:math:`\text{eV}`)"""


NEWLINE = "\n"
r"""Newline character for use in f-strings."""


CONVERTOR_TYPES = {
    "int": lambda s: int(s),
    "float": lambda s: float(s),
    "au": lambda s: float(s),  # atomic units,
    "str": lambda s: s.strip(),
    "angstrom": lambda s: float(s) * ANGSTROM,
    "2angstrom": lambda s: float(s) * ANGSTROM / 2,
    "angstrom**3": lambda s: float(s) * ANGSTROM**3,
    "amu": lambda s: float(s) * AMU,
}
rf"""
Unit conversion functions.

It has the following keys:
{NEWLINE.join(" * " + key for key in CONVERTOR_TYPES.keys())}

"""


def make_mult_dict(max_atnum=100):
    r"""
    Create dictionary from the table of multiplicities.

    The values are read from the table of multiplicities for neutral
    and charged atomic species ``multiplicities_table.csv``.

    The maximum atomic number supported is 100.

    The possible charges for a given atom range from ``-2`` to ``Z-1``.

    The multiplicities are set to zero for cases where the atomic numbers
    and charges were not present in the table. For the anions,
    the multiplicity is taken from the neutral isoelectronic species.

    Parameters
    ----------
    max_atnum : int, optional
        Highest atomic number of the elements in the multiplicities table.

    Returns
    -------
    mults_dict : dict
        Dictionary with the multiplicities for each atomic species.
        The keys are tuples of the form (atomic number, charge) and
        the values are the multiplicities.

    Examples
    --------
    To get the multiplicity of the neutral hydrogen atom (atomic number 1,
    charge 0), do:
    >>> mults_dict = _make_mults_dict()
    >>> mults_dict[(1, 0)]
    2
    """
    # Read CSV file
    with open(MULT_TABLE_CSV, "r") as file:
        reader = csv.reader(file)
        # Skip the row header
        next(reader)
        # Read column header
        col_header = next(reader)[1:]
        # Read table data
        table = list(reader)

    # Read charges from the labels
    charges = [int(charge) for charge in col_header[1:]]

    # Store multiplicities in a dictionary; keys are tuples (atnum, charge)
    mult_dict = {}
    for row in table:
        atnum, mults = int(row[0]), [int(mult) for mult in row[1:]]
        for charge, mult in zip(charges, mults):
            if mult != 0:
                mult_dict[(atnum, charge)] = mult

    return mult_dict


MULTIPLICITIES = make_mult_dict()
r"""
Dictionary of species' ground state multiplicities.

Has items ``(atnum, charge): Tuple[int, int]``.

"""


def generate_mult_csv(max_atnum=100):
    r"""
    Write a table of multiplicities to a CSV file.

    Values retrieved from ``database_beta_1.3.0.h5`` are organized into a
    table, with rows corresponding to atomic numbers and columns to charges.

    The maximum atomic number (``max_atnum``) must not exceed 100,
    which is the database's limit.

    The charge range spans from ``-2`` to ``max_atnum - 1``.

    Missing entries have multiplicities set to zero.

    Parameters
    ----------
    max_atnum : int, default=100
        Highest atomic number of the elements to be added to the table.

    Raises
    ------
    ValueError
        If the maximum allowed atomic number is greater than 100.

    """
    if max_atnum > 100:
        raise ValueError("The maximum allowed atomic number is 100.")

    # Set limits for charge
    min_charge, max_charge = -2, max_atnum

    # Create multiplicity table
    table = np.zeros((max_atnum, 1 + max_charge - min_charge), dtype=int)
    table[:, 0] = np.arange(1, max_atnum + 1)
    with h5.File(HDF5_NIST_FILE, "r") as f:
        # For each atnum
        for atnum, row in enumerate(table[:, 1:]):
            # For each charge
            for col, charge in enumerate(range(min_charge, max_charge)):
                nelec = atnum - charge
                # Skip zero or negative electron number
                if nelec <= 0:
                    break
                # Check if the data exists in the H5 file
                if charge >= 0 or nelec < 100:
                    # Read multiplicity and energy from H5 data file
                    elem = f[f"{atnum:03d}"][f"{nelec:03d}"]
                    mults = elem["Multi"][...].astype(int)
                    energies = elem["Energy"][...].astype(float)
                    # Ensure that the data is not empty
                    if len(mults) == 0 or len(energies) == 0:
                        row[col] = 0
                    else:
                        # Write multiplicity and energy of most stable species
                        row[col] = mults[np.argmin[energies]]

    # Make header lines for CSV file
    row_header = ["", "charge"] + [""] * table[:, 2:].shape[1]
    col_header = ["atnum"] + list(map(str, range(min_charge, max_charge)))

    # Write multiplicity table to CSV file
    with open(MULT_TABLE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row_header)
        writer.writerow(col_header)
        writer.writerows(table)
