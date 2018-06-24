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
#
"""Numerical Hartree-Fock Data on a Grid."""


import os
import numpy as np

from atomdb.base import Species, SpeciesTable


def load_numerical_hf_data():
    """Load data from desnity.out file into a `SpeciesTable`."""

    from StringIO import StringIO

    def helper_skip():
        """Skip the header for each species."""
        for _ in xrange(4): f.readline()

    def helper_data():
        """Read the grid, density, gradient, laplacian values into arrays."""
        data = ""
        for i in xrange(number_points):
            data += f.readline()
        data = np.loadtxt(StringIO(data))
        assert data.shape == (number_points, 4)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    species = []
    with open(os.path.join(os.path.dirname(__file__), '../data/density.out'), 'r') as f:
        line = f.readline()
        while line:
            if line.startswith(" 1st line is atomic no"):
                helper_skip()
                line = f.readline()
            if line.startswith("     "):
                # dictionary to store attributes of the new atomic species
                kwargs = {}

                atomic_number, number_electrons, number_points = [int(item) for item in line.split()]
                energy = [float(item) for item in f.readline().split()]
                assert len(energy) == 5
                assert abs(sum(energy[:-1]) - energy[-1]) < 1.e-8  #1.e-12 didn"t work?!

                # store energy component values
                kwargs["energy_components"] = dict([("T", energy[0]), ("Vne", energy[1]),
                                                    ("J", energy[2]), ("Ex", energy[3]),
                                                    ("E", energy[4])])

                grid, density, gradient, laplacian = helper_data()
                assert grid.shape == density.shape == gradient.shape == laplacian.shape

                # store grid dependent values
                kwargs.update({"grid":grid, "density":density, "gradient":gradient,
                               "laplacian":laplacian})

                # add the new atomic species
                species.append(Species(atomic_number, number_electrons, **kwargs))
                line = f.readline()

    return SpeciesTable(species)


table_numeric = load_numerical_hf_data()
