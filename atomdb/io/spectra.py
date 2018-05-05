# -*- coding: utf-8 -*-
# AtomDB is an extended periodic table database containing experimental
# and/or computational information on stable ground state
# and/or excited states of neutral and charged atomic species.
#
# Copyright (C) 2014-2015 The AtomDB Development Team
#
# This file is part of AtomDB.
#
# AtomDB is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AtomDB is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Spectra Data from NIST."""


import h5py as h5
import numpy as np

from atomdb.base import Species, SpeciesTable


def load_nist_spectra_data():
    """Load data from database_beta_1.3.0.h5 file into a `SpeciesTable`."""

    species = []
    with h5.File("atomdb/data/database_beta_1.3.0.h5", "r") as f:
        for number in f.keys():
            electrons = f[number].keys()
            assert len(electrons) == int(number)
            for electron in electrons:
                assert int(electron) <= int(number) + 1

                # get mults, energies, configurations & J values
                mults = np.array(list(f[number][electron]["Multi"][...]), dtype=int)
                energy = f[number][electron]["Energy"][...]
                config = f[number][electron]["Config"][...]
                j_vals = f[number][electron]["J"][...]
                assert len(mults) == len(energy) == len(config) == len(j_vals)

                # found violations in Derick"s data (they should be mult ordered!)
                if all(mults != sorted(mults)):
                    print(mults, sorted(mults))
                    print(energy, config, j_vals)
                    print "WARN number={0}, elec={1}, {2}, {3}".format(number, electron, mults, sorted(mults))

                # sort based on energy
                index_sorting = sorted(range(len(energy)), key=lambda k: energy[k])

                # store spectra values
                kwargs = {"mult": list(mults[index_sorting]),
                          "energy": list(energy[index_sorting]),
                          "config": list(config[index_sorting]),
                          "j_vals": list(j_vals[index_sorting])}

                # add the new atomic species
                species.append(Species(int(number), int(electron), **kwargs))
    # checks
    assert len(species) == 5050
    return SpeciesTable(species)


table_spectra = load_nist_spectra_data()
