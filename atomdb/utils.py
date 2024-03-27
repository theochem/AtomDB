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

from importlib_resources import files

# get data path
TEST_DATAPATH = files("atomdb.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


__all__ = ["multiplicities", "dev_method"]


def _gs_mult_energy(atnum, nelec, datafile):
    """Retrieve multiplicity and energy for the most stable electronic configuration of a species.

    Parameters
    ----------
    atnum : int
        Atomic number of the element
    nelec : int
        Number of electrons of the species
    datafile : str
        Path to the HDF5 file containing the data

    Returns
    -------
    mult : int
        Multiplicity of the most stable electronic configuration
    energy : float
        Energy of the most stable electronic configuration
    """
    # initialize multiplicity to zero and energy to a large value
    default_mult = 0
    default_energy = 1e6

    # set keys for the atomic number and number of electrons
    z = str(atnum).zfill(3)
    ne = str(nelec).zfill(3)

    # in database_beta_1.3.0.h5
    with h5.File(datafile, "r") as f:
        # for specie with atomic number z and ne electrons retrieve all multiplicities and energies
        mults = np.array(list(f[z][ne]["Multi"][...]), dtype=int)
        energy = f[z][ne]["Energy"][...]

    # sort the multiplicities and energies in ascending of energy
    index_sorting = sorted(list(range(len(energy))), key=lambda k: energy[k])
    mults = list(mults[index_sorting])
    energy = list(energy[index_sorting])

    # return multiplicity and energy of the most stable species or default values for missing data
    mult = mults[0] if len(mults) != 0 else default_mult
    energy = energy[0] if len(energy) != 0 else default_energy
    return mult, energy


def _make_mults_table(datafile, max_atnum=100):
    """Creates Multiplicity Table for Neutral and Charged Species up to Atomic Number `max_atnum`

    Values retrieved from 'database_beta_1.3.0.h5' are organized into a table, with rows
    corresponding to atomic numbers and columns to charges. The maximum atomic number (`max_atnum`)
    must not exceed 100, the database's limit. The charge range spans from -2 to `max_atnum`-1.
    Missing combinations have multiplicities set to zero.

    Parameters
    ----------
    datafile : str
        Path to the HDF5 file containing the data
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
    # set maximum limits for charge, get possible charges and count them
    neg_charge, pos_charge = -2, max_atnum
    charge_range = range(neg_charge, pos_charge)
    num_species = len(charge_range)

    # create multiplicity table (number of elements x number of charged species) with zeros
    mult_table = np.zeros((max_atnum, num_species), dtype=int)

    # check if the maximum atomic number is within the database's limit
    if max_atnum > 100:
        raise ValueError("The maximum allowed atomic number is 100.")

    # for each atomic number between 1 and the maximum atomic number
    for atnum in range(1, max_atnum + 1):
        # for each charge in the charge range
        for charge in charge_range:
            nelec = atnum - charge
            # if number of electrons is non-positive, go to the next atomic number
            if nelec <= 0:
                break

            # case 1: neutral or cationic species
            if charge >= 0:
                mult, _ = _gs_mult_energy(atnum, nelec, datafile)
            # case 2: anionic species, read multiplicity from neutral isoelectronic species
            else:
                # check if the neutral isoelectronic species is in the database, if not, skip case
                if nelec >= 100:
                    continue
                mult, _ = _gs_mult_energy(nelec, nelec, datafile)

            # column 0 corresponds to charge -2, column 1 to charge -1, and so on
            mult_table[atnum - 1, charge + 2] = mult
    return mult_table


def _write_mults_table_to_csv(mults_table, csv_file):
    """Write the table of multiplicities to a csv file.

    Parameters
    ----------
    mults_table : np.ndarray
        Table of multiplicities with dimension of number of elements and
        number of charged species (Z x Q).
    csv_file : str
        Path to the csv file to save the table.
    """

    # get the maximum atomic number and maximum number of charged species
    max_atnum, num_species = mults_table.shape
    element_range = range(1, max_atnum + 1)
    charge_range = range(-2, max_atnum)

    # first column is for atomic numbers, the other columns are for charges
    mult_table_with_atnum = np.zeros((max_atnum, num_species + 1), dtype=int)
    mult_table_with_atnum[:, 0] = list(element_range)
    mult_table_with_atnum[:, 1:] = mults_table

    # Write the table to a csv file
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        # write the header
        charge_label = [f"Charge" if i == 3 else "" for i in range(num_species + 1)]
        writer.writerow(charge_label)

        # write column labels
        colums_label = ["atnum"] + [f"{charge}" for charge in charge_range]
        writer.writerow(colums_label)

        # write table data
        writer.writerows(mult_table_with_atnum)


def _make_mults_dict(csv_file, max_atnum=100):
    """Create dictionary from the table of multiplicities for neutral and charged atomic species.

    The values are read from the table of multiplicities for neutral and charged atomic species
    `multiplicities_table.csv`. The maximum atomic number supported is 100. The possible charges
    for a given atom range from -2 to Z-1. The multiplicities are set to zero for cases where the
    atomic numbers and charges were not present in the table. For the anions, the multiplicity is
    taken from the neutral isoelectronic species.

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

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        # skip the header
        next(reader)
        # read column labels
        labels = next(reader)
        # read table data
        table = list(reader)

    # check table columns format
    if labels[0] != "atnum" and not all(labels[1:] == [str(i) for i in range(-2, max_atnum)]):
        raise ValueError("The provided multiplicities table does not have the expected format")

    # check table rows format
    if len(table) != max_atnum:
        raise ValueError(f"Expected {max_atnum} elements in multiplicities table, got {len(table)}")

    # read charges from the column labels, skip the first column which is for atomic numbers
    charges = labels[1:]

    # store multiplicities in a dictionary, keys are tuples (atnum, charge)
    for row in table:
        atnum, mults = int(row[0]), row[1:]
        for charge, mult in zip(charges, mults):
            mults_dict[(atnum, int(charge))] = int(mult)
    return mults_dict


multiplicities = _make_mults_dict(os.path.join(TEST_DATAPATH, "multiplicities_table.csv"))


def dev_method(all_deps):
    """Mark a method as development.

    Parameters
    ----------
    all_deps : bool
        If True, all dependencies are available and the method can be run. If False, the method
        will be skipped and an informative error message will be printed.
    """

    def decorator(func):
        if all_deps:
            return func
        else:

            def wrapper(*args, **kwargs):
                raise ImportError(
                    f"Method {func.__name__} cannot be run. Development dependencies not available."
                )

            return wrapper

    return decorator
