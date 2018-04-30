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
from atomdb.io.numeric import table_NHFG as table


def test_numerical_hf_data():
    # check that there are two H species
    assert_equal(len(table.available_species(1)), 2)


def test_numerical_hf_data_h():
    # get H species
    sp = table[(1, 1)]
    # check values of energy components
    assert_almost_equal(sp.energy_components["E"], -0.499999999545641, decimal=10)
    assert_almost_equal(sp.energy_components["T"], 0.50000000248881, decimal=10)
    assert_almost_equal(sp.energy_components["Vne"], -0.100000000203445e1, decimal=10)
    assert_almost_equal(sp.energy_components["J"], 0.312500000575059, decimal=10)
    assert_almost_equal(sp.energy_components["Ex"], -0.312500000575059, decimal=10)
    # check shape of arrays
    assert_equal(sp.grid.shape, (122,))
    assert_equal(sp.density.shape, (122,))
    assert_equal(sp.gradient.shape, (122,))
    assert_equal(sp.laplacian.shape, (122,))
    # check array elements
    assert_equal(sp.grid >= 0.0, [True] * 122)
    assert_equal(sp.density >= 0.0, [True] * 122)
    assert_almost_equal(sp.grid[:3], [0., 0.183156388887342e-01, 0.194968961085980e-01], decimal=10)
    assert_almost_equal(sp.grid[-3:], [0.292242837812349e2, 0.311090881509677e2, 0.331154519586923e2], decimal=10)
    assert_almost_equal(sp.density[:2], [0.318309887124870, 0.306860767394852], decimal=10)
    assert_almost_equal(sp.density[4:6], [0.304551328899830, 0.303684673354233], decimal=10)
    assert_almost_equal(sp.density[-2:], [0., 0.], decimal=10)
    assert_almost_equal(sp.gradient[:2], [-0.636619761671399, -0.613581739284137], decimal=10)
    assert_almost_equal(sp.gradient[-2:], [0., 0.], decimal=10)
    assert_almost_equal(sp.laplacian[:2], [0., -0.218412158120713e-1], decimal=10)
    assert_almost_equal(sp.laplacian[-2:], [0., 0.], decimal=10)


def test_numerical_hf_data_h_anion():
    # get H- species
    sp = table[(1, 2)]
    # check values of energy components
    assert_almost_equal(sp.energy_components["E"], -0.487929734301232, decimal=10)
    assert_almost_equal(sp.energy_components["T"], 0.487929736929072, decimal=10)
    assert_almost_equal(sp.energy_components["Vne"], -0.137134431428370e1, decimal=10)
    assert_almost_equal(sp.energy_components["J"], 0.790969686106793, decimal=10)
    assert_almost_equal(sp.energy_components["Ex"], -0.395484843053397, decimal=10)
    # check shape of arrays
    assert_equal(sp.grid.shape, (139,))
    assert_equal(sp.density.shape, (139,))
    assert_equal(sp.gradient.shape, (139,))
    assert_equal(sp.laplacian.shape, (139,))
    # check array elements
    assert_equal(sp.grid >= 0., [True] * 139)
    assert_equal(sp.density >= 0., [True] * 139)
    assert_almost_equal(sp.grid[:3], [0.0, 0.183156388887342e-1, 0.194968961085980e-1], decimal=10)
    assert_almost_equal(sp.grid[10], 0.321449473268761e-1, decimal=10)
    assert_almost_equal(sp.grid[-2:], [0.900171313005218e2, 0.958227374770869e2], decimal=10)
    assert_almost_equal(sp.density[:2], [0.309193381788357, 0.298087713771564], decimal=10)
    assert_almost_equal(sp.density[7:9], [0.293177850175325, 0.292176126765437], decimal=10)
    assert_almost_equal(sp.density[-20:], [0.] * 20, decimal=10)
    assert_almost_equal(sp.gradient[:2], [-0.618386750431843, -0.594311093621533], decimal=10)
    assert_almost_equal(sp.gradient[20:22], [-0.543476018733641, -0.538979599233911], decimal=10)
    assert_almost_equal(sp.gradient[-20:], [0.] * 20, decimal=10)
    assert_almost_equal(sp.laplacian[:2], [0.0, -0.211096500535927e-1], decimal=10)
    assert_almost_equal(sp.laplacian[26:28], [-0.811997961294052e-1, -0.848374851308609e-1], decimal=10)
    assert_almost_equal(sp.laplacian[-15:], [0.] * 15, decimal=10)
