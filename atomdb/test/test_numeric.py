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


import pytest

from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
from atomdb.api import load


TEST_DATAPATH = 'atomdb/test/data/'


def test_numerical_hf_data_h():
    # get H species
    sp = load('H', 0, 2, dataset='numeric', datapath=TEST_DATAPATH)
    # check shape of arrays
    assert_equal(sp.rs.shape, (122,))
    assert_equal(sp.dens_tot.shape, (122,))
    # check array elements
    assert_equal(sp.rs >= 0.0, [True] * 122)
    assert_equal(sp.dens_tot >= 0.0, [True] * 122)
    assert_almost_equal(sp.rs[:3], [0., 0.183156388887342e-01, 0.194968961085980e-01], decimal=10)
    assert_almost_equal(sp.rs[-3:], [0.292242837812349e2, 0.311090881509677e2, 0.331154519586923e2], decimal=10)
    assert_almost_equal(sp.dens_tot[:2], [0.318309887124870, 0.306860767394852], decimal=10)
    assert_almost_equal(sp.dens_tot[4:6], [0.304551328899830, 0.303684673354233], decimal=10)
    assert_almost_equal(sp.dens_tot[-2:], [0., 0.], decimal=10)    
    # get density derivative evaluated on a radial grid
    dens = sp.interpolate_dens(spin='ab', log=False)
    gradient = dens(sp.rs, deriv=1)
    # get reference values from numerical HF raw files
    fname = f'001_q000_m02_numeric_gradient.npy'
    answer = np.load(f'{TEST_DATAPATH}/numeric/db/{fname}')
    assert_almost_equal(answer[:2], [-0.636619761671399, -0.613581739284137], decimal=10)
    assert_almost_equal(answer[-2:], [0., 0.], decimal=10)
    # check array elements
    assert_almost_equal(gradient[:2], answer[:2], decimal=4)
    assert_almost_equal(gradient[-2:], answer[-2:], decimal=10)
    # assert_almost_equal(sp.lapl_tot[:2], [0., -0.218412158120713e-1], decimal=10)
    # assert_almost_equal(sp.lapl_tot[-2:], [0., 0.], decimal=10)


def test_numerical_hf_data_h_anion():
    # get H- species
    sp = load('H', -1, 1, dataset='numeric', datapath=TEST_DATAPATH)
    # check values of energy components
    answer = -0.487929734301232
    assert_almost_equal(sp.energy, answer, decimal=10)
    # assert_almost_equal(sp.energy_components["T"], 0.487929736929072, decimal=10)
    # assert_almost_equal(sp.energy_components["Vne"], -0.137134431428370e1, decimal=10)
    # assert_almost_equal(sp.energy_components["J"], 0.790969686106793, decimal=10)
    # assert_almost_equal(sp.energy_components["Ex"], -0.395484843053397, decimal=10)
    # check shape of arrays
    assert_equal(sp.rs.shape, (139,))
    assert_equal(sp.dens_tot.shape, (139,))
    # assert_equal(sp.d_dens_tot.shape, (139,))
    # assert_equal(sp.lapl_tot.shape, (139,))
    # check array elements
    assert_equal(sp.rs >= 0., [True] * 139)
    assert_equal(sp.dens_tot >= 0., [True] * 139)
    assert_almost_equal(sp.rs[:3], [0.0, 0.183156388887342e-1, 0.194968961085980e-1], decimal=10)
    assert_almost_equal(sp.rs[10], 0.321449473268761e-1, decimal=10)
    assert_almost_equal(sp.rs[-2:], [0.900171313005218e2, 0.958227374770869e2], decimal=10)
    assert_almost_equal(sp.dens_tot[:2], [0.309193381788357, 0.298087713771564], decimal=10)
    assert_almost_equal(sp.dens_tot[7:9], [0.293177850175325, 0.292176126765437], decimal=10)
    assert_almost_equal(sp.dens_tot[-20:], [0.] * 20, decimal=10)
    # get density derivative evaluated on a radial grid
    dens = sp.interpolate_dens(spin='ab', log=False)
    gradient = dens(sp.rs, deriv=1)
    # get reference values from numerical HF raw files
    fname = f'001_q-01_m01_numeric_gradient.npy'
    answer = np.load(f'{TEST_DATAPATH}/numeric/db/{fname}')
    assert_almost_equal(answer[:2], [-0.618386750431843, -0.594311093621533], decimal=10)
    assert_almost_equal(answer[20:22], [-0.543476018733641, -0.538979599233911], decimal=10)
    assert_almost_equal(answer[-20:], [0.] * 20, decimal=10)
    # check array elements
    assert_almost_equal(gradient[:2], answer[:2], decimal=3)
    assert_almost_equal(gradient[-2:], answer[-2:], decimal=10)
    # assert_almost_equal(sp.lapl_tot[:2], [0.0, -0.211096500535927e-1], decimal=10)
    # assert_almost_equal(sp.lapl_tot[26:28], [-0.811997961294052e-1, -0.848374851308609e-1], decimal=10)
    # assert_almost_equal(sp.lapl_tot[-15:], [0.] * 15, decimal=10)


@pytest.mark.parametrize(
    "atom, mult, answer",
    [
        ("H", 2, -0.499999999545641),
        ("Be", 1, -14.5730231594067),
        ("Cl", 2, -459.482072229365),
        ("Ne", 1, -128.547098072128),
    ],
)
def test_numerical_hf_energy_especies(atom, mult, answer):
    # load species and check energy
    sp = load(atom, 0, mult, dataset="numeric", datapath=TEST_DATAPATH)
    assert_almost_equal(sp.energy, answer, decimal=6)


@pytest.mark.parametrize(
    "atom, mult, npoints, answer", [("Be", 1, 146, 4.0), ("Cl", 2, 164, 17.0), ("Ne", 1, 151, 10.0)]
)
def test_numerical_hf_atomic_density(atom, mult, npoints, answer):
    # load atomic and density data
    sp = load(atom, 0, mult, dataset="numeric", datapath=TEST_DATAPATH)
    # get density evaluated on a radial grid
    grid = sp.rs
    dens = sp.dens_tot
    # check shape of arrays
    assert_equal(sp.rs.shape, (npoints,))
    assert_equal(dens.shape, grid.shape)
    # check array elements
    assert_equal(sp.rs >= 0.0, [True] * npoints)
    assert_equal(sp.dens_tot >= 0.0, [True] * npoints)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid ** 2 * dens, grid), answer, decimal=2)
    # check interpolated densities
    spline = sp.interpolate_dens(spin='ab', log=False)
    assert_almost_equal(spline(grid), dens, decimal=6)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "atom, charge, mult", [("H", 0, 2), ("Be", 0, 1), ("Cl", 0, 2), ("Ne", 0, 1)]
)
def test_numerical_hf_density_gradient(atom, charge, mult):
    # load atomic and density data
    sp = load(atom, charge, mult, dataset="numeric", datapath=TEST_DATAPATH)
    # get density derivative evaluated on a radial grid
    grid = sp.rs
    spline = sp.interpolate_dens(spin='ab', log=False)
    gradient = spline(grid, deriv=1)
    # check shape of arrays
    assert_equal(gradient.shape, grid.shape)
    # get reference values from numerical HF raw files
    id = f"{str(sp.natom).zfill(3)}_q{str(charge).zfill(3)}_m{mult:02d}"
    fname = f'{id}_numeric_gradient.npy'
    answer = np.load(f'{TEST_DATAPATH}/numeric/db/{fname}')
    # check array elements
    assert_almost_equal(gradient[:5], answer[:5], decimal=3)


@pytest.mark.xfail(reason = "Bug in spline derivative order 2")
@pytest.mark.parametrize(
    "atom, charge, mult", [("H", 0, 2), ("H", -1, 1), ("Be", 0, 1), ("Cl", 0, 3), ("Ne", 0, 1)]
)
def test_numerical_hf_density_laplacian(atom, charge, mult):
    # load atomic and density data
    sp = load(atom, charge, mult, dataset="numeric", datapath=TEST_DATAPATH)
    # get density derivative evaluated on a radial grid
    grid = sp.rs
    spline = sp.interpolate_dens(spin='ab', log=False)
    lapl = spline(grid, deriv=2)
    # check shape of arrays
    assert_equal(lapl.shape, grid.shape)
    # get reference values from numerical HF raw files
    id = f"{str(sp.natom).zfill(3)}_q{str(charge).zfill(3)}_m{mult:02d}"
    fname = f'{id}_numeric_laplacian.npy'
    answer = np.load(f'{TEST_DATAPATH}/numeric/db/{fname}')
    # check array elements
    assert_almost_equal(lapl, answer, decimal=6)
