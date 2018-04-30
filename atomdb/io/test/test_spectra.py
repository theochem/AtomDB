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


from numpy.testing import assert_equal, assert_almost_equal
from atomdb.io.spectra import table_Spectra as table


def test_load_nist_spectra_data():
    # check that there is one species with Z=1
    assert_equal(len(table.available_species(1)), 1)
    # check that there are 6 species with Z=6
    assert_equal(len(table.available_species(6)), 6)
    # check that there are 50 species with Z=50
    assert_equal(len(table.available_species(50)), 50)


def test_load_nist_spectra_data_h():
    # check H atom
    sp = table[(1, 1)]
    assert_equal(sp.mult, [2])
    assert_equal(sp.config, ["1s"])
    assert_equal(sp.j_vals, ["1/2"])
    assert_almost_equal(sp.energy, [-109678.77174307], decimal=6)


def test_load_nist_spectra_data_c():
    # check C+4 cation
    sp = table[(6, 2)]
    assert_equal(sp.mult, [1, 3])
    assert_equal(sp.config, ["1s2", "1s.2s"])
    assert_equal(sp.j_vals, ["0", "1"])
    assert_almost_equal(sp.energy, [-7114484.97, -4703213.77], decimal=2)


def test_load_nist_spectra_data_sn():
    # check Sn+16 cation
    sp = table[(50, 34)]
    assert_equal(sp.mult, [3])
    assert_equal(sp.config, ["4p4"])
    assert_equal(sp.j_vals, ["2"])
    assert_almost_equal(sp.energy, [-1.33299759e9], decimal=4)
