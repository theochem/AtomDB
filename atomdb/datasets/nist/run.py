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

import os

import numpy as np

from scipy import constants

import h5py as h5

import csv

import atomdb
from atomdb.periodic import Element
from atomdb.utils import MULTIPLICITIES, CMINV, EV 
from atomdb.species import DEFAULT_DATAPATH


__all__ = [
    "run",
]


DOCSTRING = """Conceptual DFT Dataset

The following neutral and ionic species are available

`neutrals` H to Lr
`cations` H to Lr
`anions` H to Lr (up to charge -2)

The values were obtained from the paper, `Phys. Chem. Chem. Phys., 2016,18, 25721-25734 <https://doi.org/10.1039/C6CP04533B>`_.
For each element/charge pair the values correspond to the most stable electronic configuration.

"""


def load_nist_spectra_data(atnum, nelec, datafile):
    """Load data from database_beta_1.3.0.h5 file into a `SpeciesTable`.
    
    Note: function based on spectra.py module from old master
    https://github.com/theochem/AtomDB/blob/oldmaster/atomdb/io/spectra.py
    """

    # set keys for the atomic number and number of electrons
    z = str(atnum).zfill(3)
    ne = str(nelec).zfill(3)

    # in database_beta_1.3.0.h5
    with h5.File(datafile, "r") as f:
        # for specie with atomic number z and ne electrons get mults, energies, configurations & J values
        mults = np.array(list(f[z][ne]["Multi"][...]), dtype=int)
        energy = f[z][ne]["Energy"][...]
        config = f[z][ne]["Config"][...]
        j_vals = f[z][ne]["J"][...]
        assert len(mults) == len(energy) == len(config) == len(j_vals)

    # found violations in Derick"s data (they should be mult ordered!)
    if all(mults != sorted(mults)):
        print((mults, sorted(mults)))
        print((energy, config, j_vals))
        print("WARN number={0}, elec={1}, {2}, {3}".format(z, ne, mults, sorted(mults)))

    # sort based on energy
    index_sorting = sorted(list(range(len(energy))), key=lambda k: energy[k])
    # mults = list(mults[index_sorting])
    # energy = list(energy[index_sorting])

    # sort and store the mults, energies, configurations & J values in ascending order of energy
    output = {"mult": list(mults[index_sorting]),
            "energy": list(energy[index_sorting]),
            "config": list(config[index_sorting]),
            "j_vals": list(j_vals[index_sorting])}
    
    return output


def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Parse NIST related data and compile the AtomDB database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    atnum = atomdb.element_number(elem)
    nelec = atnum - charge
    nspin = mult - 1
    obasis_name = None

    # Check that the input charge is valid
    if charge < -2 or charge > atnum:
        raise ValueError(f"{elem} with {charge} not available.")
    
    # Check that the input multiplicity corresponds to this configuration.
    if not mult == MULTIPLICITIES[(atnum, charge)]:
        raise ValueError(f"{elem} with charge {charge} and multiplicity {mult} not available.")

    print(f"Generating {elem} with charge {charge} and multiplicity {mult}.")
    # #
    # # Element properties
    # #
    # atom = Element(elem)
    # atmass = atom.mass["stb"]
    # cov_radius, vdw_radius, at_radius, polarizability, dispersion_c6 = [
    #     None,
    # ] * 5
    # if charge == 0:
    #     # overwrite values for neutral atomic species
    #     cov_radius, vdw_radius, at_radius = (atom.cov_radius, atom.vdw_radius, atom.at_radius)
    #     polarizability = atom.pold
    #     dispersion_c6 = atom.c6

    #
    # Get the ground state energy from database_beta_1.3.0.h5.
    #
    # Set an energy default value since there is no data for anions in database_beta_1.3.0.h5.
    energy = None
    datapath = f"{DEFAULT_DATAPATH}/{dataset.lower()}/raw/database_beta_1.3.0.h5"
    if charge >= 0: # neutral or cationic species
        spectra_data = load_nist_spectra_data(atnum, nelec, datapath)
        energies = spectra_data["energy"]
        # Convert energy to Hartree from cm^{-1} if available
        energy = energies[0] * CMINV if len(energies) != 0 else energy

    # Get conceptual-DFT related properties from c6cp04533b1.csv
    # Locate where each table starts: search for "Element" columns
    datapath = f"{DEFAULT_DATAPATH}/{dataset.lower()}/raw/c6cp04533b1.csv"
    data = list(csv.reader(open(datapath, "r")))
    tabid = [i for i, row in enumerate(data) if "Element" in row]
    # Assign each conceptual-DFT data table to a variable.
    # Remove empty and header rows
    table_ips = data[tabid[0] : tabid[1]]
    table_ips = [row for row in table_ips if len(row[1]) > 0]
    table_mus = data[tabid[1] : tabid[2]]
    table_mus = [row for row in table_mus if len(row[1]) > 0]
    table_etas = data[tabid[2] :]
    table_etas = [row for row in table_etas if len(row[1]) > 0]
    # Get property at table(atnum, charge); convert to Hartree
    colid = table_ips[0].index(str(charge))
    ip = float(table_ips[atnum][colid]) * EV if len(table_ips[atnum][colid]) > 1 else None
    colid = table_mus[0].index(str(charge))
    mu = float(table_mus[atnum][colid]) * EV if len(table_mus[atnum][colid]) > 1 else None
    colid = table_etas[0].index(str(charge))
    eta = float(table_etas[atnum][colid]) * EV if len(table_etas[atnum][colid]) > 1 else None

    # Return Species instance
    # return atomdb.Species(
    #     dataset,
    #     elem,
    #     atnum,
    #     obasis_name,
    #     nelec,
    #     nspin,
    #     nexc,
    #     # atmass,
    #     # cov_radius,
    #     # vdw_radius,
    #     # at_radius,
    #     # polarizability,
    #     # dispersion_c6,
    #     energy,
    #     ip=ip,
    #     mu=mu,
    #     eta=eta,
    # )
    atom =  atomdb.Species(
        dataset,
        elem,
        atnum,
        obasis_name,
        nelec,
        nspin,
        nexc,
        energy=energy,
        ip=ip,
        mu=mu,
        eta=eta,
    )

    print(vars(atom))
    # print(atom._data.elem, atom.charge, atom.mult)
    # print(atom.energy, atom.ip, atom.mu, atom.nexc)
    print(atom.atmass)
    print(atom.spinpol)
    print(atom.polarizability)
    return atom
