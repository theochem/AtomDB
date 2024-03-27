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

import pytest
from atomdb.utils import multiplicities

# Mark all tests in this file with the 'dev' flag
pytestmark = pytest.mark.dev


def test_mults_table():
    # atnum, charge, mult
    species = [
        [1, 0, 2],  # H
        [7, 0, 4],  # N
        [7, 1, 3],  # N+
        [7, -1, 3],  # N-
        [24, 0, 7],  # Cr
        [24, 1, 6],  # Cr+
        [24, -1, 6],  # Cr-
        [30, 0, 1],  # Zn
        [30, 1, 2],  # Zn+
        [30, -1, 2],  # Zn-
    ]
    for atnum, charge, mult in species:
        assert multiplicities[(atnum, charge)] == mult
