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


from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from atomdb.base import Species, SpeciesTable


def test_species_raises():
    # raises for negative, non-integer, or higher than 118 atomic numbers
    assert_raises(ValueError, Species, -5, 5)
    assert_raises(ValueError, Species, 120, 10)
    assert_raises(ValueError, Species, 10.1, 10)
    # raises for negative or fractional
    assert_raises(ValueError, Species, 5, -1)
    assert_raises(ValueError, Species, 6, 1.5)


def test_species():
    # C atom
    sp = Species(6, 6)
    assert_equal(sp.number, 6)
    assert_equal(sp.nelectron, 6)
    sp = Species(6, 6, **{})
    assert_equal(sp.number, 6)
    assert_equal(sp.nelectron, 6)
    # H+ cation
    sp = Species(1, 0)
    assert_equal(sp.number, 1)
    assert_equal(sp.nelectron, 0)
    # Li-1 anion
    sp = Species(3, 4)
    assert_equal(sp.number, 3)
    assert_equal(sp.nelectron, 4)
    # Ca+2 with kwargs
    sp = Species(20, 18, **{"symbol": "Ca", "name": "Calcium", "mw": 40.078})
    assert_equal(sp.number, 20)
    assert_equal(sp.nelectron, 18)
    assert_equal(sp.symbol, "Ca")
    assert_equal(sp.name, "Calcium")
    assert_almost_equal(sp.mw, 40.078, decimal=4)
