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

r"""Tool functions."""

import numpy as np

import os
import h5py as h5
import csv


data_path = os.path.dirname(__file__)


__all__ = [
    "multiplicities",
]


def _get_moststable_species(atnum, nelec):
    """Get the multiplicity and energy for the most stable electronic configuration
    of a given atomic species.

    Parameters
    ----------
    atnum : int
        Atomic number of the element
    nelec : int
        Number of electrons of the species

    Returns
    -------
    mult : int
        Multiplicity of the most stable electronic configuration
    energy : float
        Energy of the most stable electronic configuration
    """
    default_mult = 0  # initialize the multiplicity to zero
    default_energy = 1e6  # initialize the energy to a large value

    # Load the contents of the database_beta_1.3.0.h5 file for a given atomic number and number of
    # electrons. Get the list of energies and multiplicities for stable electronic configurations
    # and sort them based on energy.
    z = str(atnum).zfill(3)
    ne = str(nelec).zfill(3)
    with h5.File(os.path.join(f"{data_path}/database_beta_1.3.0.h5"), "r") as f:
        mults = np.array(list(f[z][ne]["Multi"][...]), dtype=int)
        energy = f[z][ne]["Energy"][...]
    index_sorting = sorted(list(range(len(energy))), key=lambda k: energy[k])
    mults = list(mults[index_sorting])
    energy = list(energy[index_sorting])

    # Return the value of the multiplicity for the lowest energy
    # Handle missing data cases
    mult = mults[0] if len(mults) != 0 else default_mult
    energy = energy[0] if len(energy) != 0 else default_energy
    return mult, energy


def _make_mults_table(max_atnum=100):
    """Create a table of multiplicities for neutral and charged atomic species with atomic
     numbers up to `max_atnum`.

    The values are obtained from the database_beta_1.3.0.h5 file and are stored as a table with
    rows corresponding to the atomic number and columns to the charge. The maximum atomic number
    (`max_atnum`) that can be considered is 100, as the database only contains data up to Fermium (Z=100).
    The charge range goes from -2 to `max_atnum`-1. The multiplicities are initialized to zero for
    cases where the atomic numbers and charges are not present in the database.

    Parameters
    ----------
    max_atnum : int, optional
        Highest atomic number of the elements to be added to the table.

    Returns
    -------
    mult_table : np.ndarray
        Table of multiplicities

    Raises
    ------
    ValueError
        If the maximum allowed atomic number is greater than 100.
    """
    # Define the dimensions of the table by specifying the range of atomic numbers
    # and charges to consider
    # Here we only consider charges from -2 to Z-1
    neg_charge = -2
    pos_charge = max_atnum
    charge_range = range(neg_charge, pos_charge)
    num_species = len(charge_range)
    element_range = range(1, max_atnum + 1)
    mult_table = np.zeros((max_atnum, num_species), dtype=int)

    # Avoid accessing the database for atomic numbers greater than 100 since the database
    # only contains data up to Fermium (Z=100)
    if max_atnum > 100:
        raise ValueError("The maximum allowed atomic number is 100.")

    for atnum in element_range:
        for charge in charge_range:
            nelec = atnum - charge
            if nelec <= 0:  # skip if the number of electrons is negative
                break
            # Multiplicity for neutral and cations
            # when asigning the multiplicity to the table, the column/charge value is shifted by 2
            # because the charge range starts at -2
            if charge >= 0:
                mult, _ = _get_moststable_species(atnum, nelec)
                mult_table[atnum - 1, charge + 2] = mult
            else:
                # For anions, the multiplicity is taken from the neutral isoelectronic species.
                # However, database_beta_1.3.0.h5 only has data up to Fermium (Z=100), so this
                # `else`` statement only works for anions up to 100 electrons
                if nelec >= 100:
                    continue
                mult, _ = _get_moststable_species(nelec, nelec)
                mult_table[atnum - 1, charge + 2] = mult
    return mult_table


def _write_mults_table_to_csv(mults_table):
    """Write the table of multiplicities to a csv file.

    Parameters
    ----------
    mults_table : np.ndarray
        Table of multiplicities with dimension of number of elements and
        number of charged species (Z x Q).
    """
    # File with the table of multiplicities to be created in atomdb/data
    filename = f"{data_path}/multiplicities_table.csv"

    # Get the number of rows and columns of the table which correspond to the maximum atomic
    # number and the number of charged species, respectively.
    max_atnum, num_species = mults_table.shape
    element_range = range(1, max_atnum + 1)
    charge_range = range(-2, max_atnum)
    # Add a column specifiying the atomic numbers to the table
    mult_table_with_atnum = np.zeros((max_atnum, num_species + 1), dtype=int)
    mult_table_with_atnum[:, 0] = list(element_range)
    mult_table_with_atnum[:, 1:] = mults_table

    # Write the table to a csv file
    # Add labeles to the columns for atomic number and charge values
    charge_label = [f"Charge" if i == 3 else "" for i in range(num_species + 1)]
    colums_label = ["atnum"] + [f"{charge}" for charge in charge_range]
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(charge_label)
        writer.writerow(colums_label)
        writer.writerows(mult_table_with_atnum)


def _make_mults_dict(max_atnum=100):
    """Create a dictionary from a table of multiplicities for neutral and charged atomic species.

    The values were obtained from the database_beta_1.3.0.h5 file. The maximum atomic number
    that can be considered is 100, as the database only contains data up to Fermium (Z=100).
    The considered charges for a given atom range from -2 to Z-1.
    The multiplicities are taken as zero for cases where the atomic numbers and charges were
    not present in the database. For the anions, the multiplicity was taken from the neutral
    isoelectronic species.

    Parameters
    ----------
    max_atnum : int, optional
        Highest atomic number of the elements in the multiplicities table.

    Returns
    -------
    mults_dict : dict
        Dictionary with the multiplicities for each atomic species.
        The keys are tuples of the form (atomic number, charge) and the values are the multiplicities.

    Examples
    --------
    To get the multiplicity of the neutral Hidrogen atom (atomic number 1, charge 0) do:
    >>> mults_dict = _make_mults_dict()
    >>> mults_dict[(1, 0)]
    2
    """
    mults_dict = {}
    filename = f"{data_path}/multiplicities_table.csv"

    with open(filename, "r") as file:
        reader = csv.reader(file)
        # When created using the function _write_mults_table_to_csv, the table has
        # exactly two header lines that have to be skipped. The second header line
        # labels the atomic number column and the charge values.
        next(reader)
        header = next(reader)
        table = list(reader)
    # Check the table's format
    if "atnum" not in header:
        raise ValueError("The provided multiplicities table does not have the expected format")
    if len(table) != max_atnum:
        raise ValueError(
            f"Wrong multiplicities table, {max_atnum} elements expected, {len(table)} found"
        )
    # Get the charges from the header
    charges = header[1:]  # skip the first column which is for atomic numbers

    # Each row in the table corresponds to an atomic number and the multiplicities for the
    # different charged species
    for row in table:
        atnum = int(row[0])
        mults = row[1:]
        for charge, mult in zip(charges, mults):
            mults_dict[(atnum, int(charge))] = int(mult)
    return mults_dict


def _test_mults_table():
    # atnum, charge, mult
    species = [
        [1, 0, 2],  # H
        [7, 0, 4],  # N
        [7, 1, 3],  # N+
        [7, -1, 3],  # N-
        [24, 0, 7],  # Cr
        [24, 1, 6],  # Cr+
        [24, -1, 6],  # Cr-
        [30, 0, 1],  # Zn
        [30, 1, 2],  # Zn+
        [30, -1, 2],  # Zn-
    ]
    _multiplicities = _make_mults_dict()
    for atnum, charge, mult in species:
        assert _multiplicities[(atnum, charge)] == mult


multiplicities = _make_mults_dict()
