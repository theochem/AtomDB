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

r"""AtomDB promolecule class."""

from .api import DEFAULT_DATAPATH, DEFAULT_DATASET
from .api import load

import numpy as np


__all__ = [
    "Promolecule",
]


class Promolecule:
    r"""
    TODO: have coefficients besides the default "1".

    """

    def __init__(self, atoms, coords, charges, mults, dataset=DEFAULT_DATASET, datapath=DEFAULT_DATAPATH):
        r"""
        TODO: have a table of default charges/mults and make them optional inputs.

        """
        self.atoms = [
            load(atom, charge, mult, dataset=dataset, datapath=datapath)
            for atom, charge, mult in zip(atoms, charges, mults)
        ]
        self.coords = [coord for coord in coords]


def extensive_property(atoms, atom_coords, point):
    r"""
    TODO: add to Promolecule class

    """
    prop = 0
    for (atom, coords) in zip(atoms, atom_coords):
        # Get radius between point of interest and atom
        r = np.sqrt(np.dot(coords - point))
        # Compute property at the proper radius
        prop += atom.extensive_property(r)
    return prop


def intensive_property(atoms, p=1):
    r"""
    TODO: add to Promolecule class

    """
    # P-mean of each atom's property value
    return (sum(atom.intensive_property ** p for atom in atoms) / len(atoms)) ** (1 / p)
