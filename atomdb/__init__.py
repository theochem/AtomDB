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

from atomdb.periodic import Element

from atomdb.species import Species

from atomdb.promolecule import Promolecule

from atomdb.periodic import element_number, element_symbol, element_name

from atomdb.species import compile, load, dump, raw_datafile

from atomdb.promolecule import make_promolecule

from atomdb.version import version


__all__ = [
    "Element",
    "Species",
    "Promolecule",
    "element_number",
    "element_symbol",
    "element_name",
    "compile",
    "load",
    "dump",
    "raw_datafile",
    "make_promolecule" "version",
]


__version__ = version
r"""AtomDB version string."""
