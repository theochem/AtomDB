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

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from atomdb.api import load


TEST_DATAPATH = "atomdb/test/data/"


def test_gaussian_hf_data_be():
    # get Be atomic data
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    # check values of energy components
    answer = -14.55433897481303
    assert_almost_equal(sp.energy, answer, decimal=10)
    # check shape of arrays
    grid = sp.rs
    dens = sp.dens_tot
    orb_dens_a = sp._orb_dens_up
    energy_a = sp.ao.energy_a
    assert_equal(energy_a.shape, (12,))
    assert_equal(grid.shape, (1000,))
    assert_equal(dens.shape, grid.shape)
    assert_equal(orb_dens_a.shape, (12, 1000))
    # check array elements
    assert_equal(grid >= 0.0, [True] * 1000)
    assert_equal(dens >= 0.0, [True] * 1000)
    energy = np.array(
        [
            -4.7333071,
            -0.3114001,
            0.0354727,
            0.0354727,
            0.0387768,
            0.1265828,
            0.1265828,
            0.1267737,
            0.1892719,
            0.4415274,
            0.4415274,
            0.4425412,
        ]
    )
    assert (abs(energy_a - energy) < 1.0e-6).all()
    assert (abs(sp.ao.energy_b - energy) < 1.0e-6).all()
    # check density
    assert_almost_equal(4 * np.pi * np.trapz(grid ** 2 * dens, grid), 4, decimal=4)
    dens_a = np.sum(orb_dens_a, axis=0)
    assert_almost_equal(4 * np.pi * np.trapz(grid ** 2 * dens_a, grid), 2, decimal=4)
