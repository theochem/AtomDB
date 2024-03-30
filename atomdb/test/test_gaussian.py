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

from atomdb.api import load
from importlib_resources import files
import numpy as np
from numpy.testing import assert_almost_equal
import os

# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


def test_compiled_gaussian_hf_data():
    ### Use Be atomic data as a test case
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)

    # check values of energy components
    answer = -14.55433897481303
    assert_almost_equal(sp.energy, answer, decimal=10)

    # load radial grid, total density, alpha and beta (radial) orbital densities, orbital energies
    grid = sp.rs
    dens = sp.dens_tot
    orb_dens_a = sp.mo_dens_a
    energy_a = sp.ao.energy_a

    # check array shapes
    assert energy_a.shape == (12,)
    assert grid.shape == (1000,)
    assert dens.shape == grid.shape
    assert orb_dens_a.shape == (12, 1000)

    # check array elements
    # all R and density values are positive
    assert all(grid >= 0.0)
    assert all(dens >= 0.0)

    # check alpha and beta energies are the same (closed shell) and have correct values
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
    assert np.allclose(energy_a, energy, atol=1.0e-6)
    assert np.allclose(sp.ao.energy_b, energy, atol=1.0e-6)


@pytest.mark.parametrize("atom, mult, nelec, nalpha", [("Be", 1, 4, 2), ("B", 2, 5, 3)])
def test_gaussian_hf_density_be(atom, mult, nelec, nalpha):
    # load Be atomic data and make density spline
    sp = load(atom, 0, mult, dataset="gaussian", datapath=TEST_DATAPATH)
    grid = sp.rs
    orb_dens_a = sp.mo_dens_a
    spline_dens = sp.density_func(spin="ab", log=True)
    dens_a = np.sum(orb_dens_a, axis=0)

    # check density values
    # check the total density integrates to the number of electrons
    assert np.allclose(4 * np.pi * np.trapz(grid**2 * sp.dens_tot, grid), nelec, rtol=1e-3)

    # check the alpha density integrates to the number of alpha electrons
    assert np.allclose(4 * np.pi * np.trapz(grid**2 * dens_a, grid), nalpha, rtol=1e-3)

    # check interpolated densities
    assert np.allclose(spline_dens(grid), sp.dens_tot, atol=1e-6)


@pytest.mark.xfail
@pytest.mark.parametrize("atom, mult", [("Be", 1), ("B", 2)])
def test_gaussian_hf_gradient_be(atom, mult):
    # load Be atomic data, make density spline and evalaute 1st derivative of density
    sp = load(atom, 0, mult, dataset="gaussian", datapath=TEST_DATAPATH)
    grid = sp.rs
    spline_dens = sp.density_func(spin="ab", log=True)
    gradient = spline_dens(grid, deriv=1)
    np_gradient = np.gradient(sp.dens_tot, grid)

    # check spline gradient against numpy gradient
    assert np.allclose(gradient, np_gradient, rtol=1e-3)


@pytest.mark.xfail
@pytest.mark.parametrize("atom, mult", [("Be", 1), ("B", 2)])
def test_gaussian_hf_laplacian_be(atom, mult):
    # load the atomic data, make a spline of the density and evaluate the second derivative
    # of the density from it. Compare with gradient from numpy.
    sp = load(atom, 0, mult, dataset="gaussian", datapath=TEST_DATAPATH)
    grid = sp.rs
    spline_dens = sp.density_func(spin="ab", log=True)
    d2dens = spline_dens(grid, deriv=2)

    # load reference values
    np_gradient = np.gradient(sp.dens_tot, grid)
    np_d2dens = np.gradient(np_gradient, grid)

    # check interpolated laplacian values against reference
    assert np.allclose(d2dens, np_d2dens, rtol=1e-3)


@pytest.mark.parametrize("atom, mult", [("Be", 1), ("B", 2)])
def test_gaussian_hf_ked_be(atom, mult):
    # load the atomic data and make a spline of the kinetic energy density.
    sp = load(atom, 0, mult, dataset="gaussian", datapath=TEST_DATAPATH)
    grid = sp.rs
    spline_kdens = sp.ked_func(spin="ab", log=True)

    # check interpolated densities
    assert np.allclose(spline_kdens(grid), sp.ked_tot, atol=1e-6)
