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
