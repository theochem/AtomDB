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

r"""NIST compile function."""

from tempfile import NamedTemporaryFile

import os

import numpy as np

import h5py as h5

import csv

import atomdb


__all__ = [
    "run",
]


def run(elem, charge, mult, nexc, basis, dataset, datapath):
    r"""Parse NIST related data and compile the AtomDB database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    natom = atomdb.element_number(elem)
    nelec = natom - charge
    nspin = mult - 1

    #
    # Element properties
    #
    cov_radii, vdw_radii = atomdb.get_element_data(elem)

    # Verify there is data available for the species
    if charge >= 0:
        # Check that the inputted multiplicity corresponds to the lowest-energy
        # electronic configuration of the species and get its energy from database_beta_1.3.0.h5
        z = str(natom).zfill(3)
        ne = str(nelec).zfill(3)
        with h5.File(os.path.join(os.path.dirname(__file__), "raw/database_beta_1.3.0.h5"), "r") as f:
            mults = np.array(list(f[z][ne]["Multi"][...]), dtype=int)
            energy = f[z][ne]["Energy"][...]
            # config = f[z][ne]["Config"][...]
            # j_vals = f[z][ne]["J"][...]
        # sort based on energy
        index_sorting = sorted(list(range(len(energy))), key=lambda k: energy[k])
        mults = list(mults[index_sorting])
        energy = list(energy[index_sorting])

        if not mult == mults[0]:
            raise ValueError(f"{elem} with {charge} and multiplicity {mult} not available.")
        energy = energy[0]
    elif -2 <= charge < 0:
        # Check that the inputted multiplicity is correct:
        # Get the lowest energy multiplicity of the corresponding
        # neutral isoelectronic species from database_beta_1.3.0.h5
        z = str(natom - charge).zfill(3)
        ne = str(nelec).zfill(3)
        with h5.File(os.path.join(os.path.dirname(__file__), "raw/database_beta_1.3.0.h5"), "r") as f:
            mults = np.array(list(f[z][ne]["Multi"][...]), dtype=int)
            energy = f[z][ne]["Energy"][...]
        # sort based on energy
        index_sorting = sorted(list(range(len(energy))), key=lambda k: energy[k])
        mults = list(mults[index_sorting])
        energy = list(energy[index_sorting])

        if not mult == mults[0]:
            raise ValueError(f"{elem} with {charge} and multiplicity {mult} not available.")
        # There is no data for anions in database_beta_1.3.0.h5, therefore:
        energy = None
    else:
        raise ValueError(f"{elem} with {charge} not available.")

    # Get conceptual-DFT related properties from c6cp04533b1.csv
    # Locate where each table starts: search for "Element" columns
    data = list(csv.reader(open(os.path.join(os.path.dirname(__file__), "raw/c6cp04533b1.csv"), "r")))
    tabid = [i for i, row in enumerate(data) if "Element" in row]
    # Assign each conceptual-DFT data table to a variable.
    # Remove empty and header rows
    table_ips = data[tabid[0]:tabid[1]]
    table_ips = [row for row in table_ips if len(row[1]) > 0]
    table_mus = data[tabid[1]:tabid[2]]
    table_mus = [row for row in table_mus if len(row[1]) > 0]
    table_etas = data[tabid[2]:]
    table_etas = [row for row in table_etas if len(row[1]) > 0]
    # Get property at table(natom, charge)
    colid = table_ips[0].index(str(charge))
    ip = float(table_ips[natom][colid]) if len(table_ips[natom][colid]) > 1 else None
    colid = table_mus[0].index(str(charge))
    mu = float(table_mus[natom][colid]) if len(table_mus[natom][colid]) > 1 else None
    colid = table_etas[0].index(str(charge))
    eta = float(table_etas[natom][colid]) if len(table_etas[natom][colid]) > 1 else None

    # Return Species instance
    return atomdb.Species(
        dataset,
        elem,
        natom,
        basis,
        nelec,
        nspin,
        nexc,
        cov_radii,
        vdw_radii,
        energy,
        ip=ip,
        mu=mu,
        eta=eta,
    )