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

r"""Slater dataset.

Module responsible for reading and storing atomic wave-function information from '.slater' files and
    computing electron density, and kinetic energy from them.

AtomicDensity:
    Information about atoms obtained from .slater file and able to construct atomic density
        (total, core and valence) from the linear combination of Slater-type orbitals.
    Elements supported by default from "./atomdb/data/slater_atom/" range from Hydrogen to Xenon.

load_slater_wfn : Function for reading and returning information from '.slater' files that consist
    of anion, cation and neutral atomic wave-function information.

"""
