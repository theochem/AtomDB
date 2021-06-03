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

r"""AtomDB configuration file."""


from os import environ

from os.path import abspath, dirname, join


__all__ = [
    "DEFAULT_DATASET",
    "DATAPATH",
]


DEFAULT_DATASET = "hci_ccpwcvqz"
r"""Default dataset to query."""


DATAPATH_ENV = environ.get('ATOMDB_DATAPATH')
r"""The value of the environment variable `ATOMDB_DATAPATH`."""


DATAPATH = abspath(DATAPATH_ENV if DATAPATH_ENV else join(dirname(__file__), "datasets/"))
r"""The default path for raw and compiled AtomDB data files."""
