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

r"""AtomDB, a database of atomic and ionic properties."""

import importlib
from importlib.metadata import PackageNotFoundError

from atomdb.periodic import Element

from atomdb.species import Species

from atomdb.promolecule import Promolecule

from atomdb.periodic import element_number, element_symbol, element_name

from atomdb.species import compile_species, load, dump, raw_datafile

from atomdb.promolecule import make_promolecule


__all__ = [
    "Element",
    "Species",
    "Promolecule",
    "element_number",
    "element_symbol",
    "element_name",
    "compile_species",
    "load",
    "dump",
    "raw_datafile",
    "make_promolecule",
]


r"""AtomDB version string."""

try:
    __version__ = importlib.metadata.version("qc-AtomDB")
except PackageNotFoundError:
    # Package is not installed
    print("Package 'qc-AtomDB' is not installed.")
