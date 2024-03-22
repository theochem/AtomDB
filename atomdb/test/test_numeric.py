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


TEST_DATAPATH = "atomdb/test/data/"


def test_numerical_hf_data_h():
    # load H species
    sp = load("H", 0, 2, dataset="numeric", datapath=TEST_DATAPATH)

    # check shape radial grid and total density arrays
    assert sp.rs.shape == (122,)
    assert sp.dens_tot.shape == sp.rs.shape

    # check radial grid and total density arrays values
    assert all(sp.rs >= 0.0)
    assert all(sp.dens_tot >= 0.0)
    assert np.allclose(sp.rs[:3], [0.0, 0.183156388887342e-01, 0.194968961085980e-01], atol=1e-10)
    assert np.allclose(
        sp.rs[-3:], [0.292242837812349e2, 0.311090881509677e2, 0.331154519586923e2], atol=1e-10
    )
    assert np.allclose(sp.dens_tot[:2], [0.318309887124870, 0.306860767394852], atol=1e-10)
    assert np.allclose(sp.dens_tot[4:6], [0.304551328899830, 0.303684673354233], atol=1e-10)
    assert np.allclose(sp.dens_tot[-2:], [0.0, 0.0], atol=1e-10)

    # evaluate radial density gradient (first derivative of density spline)
    dens = sp.density_func(spin="ab", log=True)
    gradient = dens(sp.rs, deriv=1)

    # load gradient reference values from numerical HF raw files
    fname = "001_q000_m02_numeric_gradient.npy"
    ref_grad = np.load(f"{TEST_DATAPATH}/numeric/db/{fname}")

    # check interpolated gradient values against reference
    # close to the nuclei the spline derivative does not describe the gradient well
    assert np.allclose(gradient[:2], ref_grad[:2], atol=1e-3)
    # away from the nuclei the spline derivative describes the gradient well
    assert np.allclose(gradient[-2:], ref_grad[-2:], atol=1e-10)


def test_numerical_hf_data_h_anion():
    # load H- species
    sp = load("H", -1, 1, dataset="numeric", datapath=TEST_DATAPATH)

    # check energy
    assert_almost_equal(sp.energy, -0.487929734301232, decimal=10)

    # check shape radial grid and total density arrays
    assert sp.rs.shape == (139,)
    assert sp.dens_tot.shape == sp.rs.shape

    # reference radial values sample and corresponding indices
    ref_rs = np.array(
        [
            0.0,
            0.183156388887342e-1,
            0.194968961085980e-1,
            0.321449473268761e-1,
            0.900171313005218e2,
            0.958227374770869e2,
        ]
    )
    ref_rs_idx = np.array([0, 1, 2, 10, -2, -1])

    # check radial grid array values using reference sample
    assert all(sp.rs >= 0.0)
    assert np.allclose(sp.rs[ref_rs_idx], ref_rs, atol=1e-10)

    # reference density values sample and corresponding indices
    ref_dens_tot = [0.309193381788357, 0.298087713771564, 0.293177850175325, 0.292176126765437]
    ref_dens_tot_idx = np.array([0, 1, 7, 8])

    # check total density array values using reference sample
    assert all(sp.dens_tot >= 0.0)
    assert np.allclose(sp.dens_tot[ref_dens_tot_idx], ref_dens_tot, atol=1e-10)
    assert np.allclose(sp.dens_tot[-20:], 0.0, atol=1e-10)

    # evaluate radial density gradient (first derivative of density spline)
    dens = sp.density_func(spin="ab", log=True)
    gradient = dens(sp.rs, deriv=1)

    # load gradient reference values from numerical HF raw files
    fname = "001_q-01_m01_numeric_gradient.npy"
    ref_grad = np.load(f"{TEST_DATAPATH}/numeric/db/{fname}")

    # check interpolated gradient values against reference
    # close to the nuclei the spline derivative does not describe the gradient well
    assert np.allclose(gradient[:2], ref_grad[:2], atol=1e-3)
    # away from the nuclei the spline derivative describes the gradient well
    assert np.allclose(gradient[-2:], ref_grad[-2:], atol=1e-10)


@pytest.mark.parametrize(
    "atom, mult, energy",
    [
        ("H", 2, -0.499999999545641),
        ("Be", 1, -14.5730231594067),
        ("Cl", 2, -459.482072229365),
        ("Ne", 1, -128.547098072128),
    ],
)
def test_numerical_hf_energy_especies(atom, mult, energy):
    # load species and check energy
    sp = load(atom, 0, mult, dataset="numeric", datapath=TEST_DATAPATH)
    assert_almost_equal(sp.energy, energy, decimal=6)


@pytest.mark.parametrize(
    "atom, mult, npoints, nelec", [("Be", 1, 146, 4.0), ("Cl", 2, 164, 17.0), ("Ne", 1, 151, 10.0)]
)
def test_numerical_hf_atomic_density(atom, mult, npoints, nelec):
    # load atomic and density data
    sp = load(atom, 0, mult, dataset="numeric", datapath=TEST_DATAPATH)
    # load radial grid and total density arrays
    grid, dens = sp.rs, sp.dens_tot

    # check shape of arrays
    assert sp.rs.shape == (npoints,)
    assert dens.shape == grid.shape

    # check radial grid and total density arrays values
    assert all(sp.rs >= 0.0)
    assert all(sp.dens_tot >= 0.0)

    # check the density integrates to the correct number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), nelec, decimal=2)

    # get density spline and check its values
    spline = sp.density_func(spin="ab", log=True)
    assert np.allclose(spline(grid), dens, atol=1e-6)


@pytest.mark.parametrize(
    "atom, charge, mult", [("H", 0, 2), ("Be", 0, 1), ("Cl", 0, 2), ("Ne", 0, 1)]
)
def test_numerical_hf_density_gradient(atom, charge, mult):
    # load density and radial gradient (spline derivative) evaluated the radial grid
    sp = load(atom, charge, mult, dataset="numeric", datapath=TEST_DATAPATH)
    grid, spline = sp.rs, sp.density_func(spin="ab", log=True)
    gradient = spline(grid, deriv=1)

    # check shape of arrays
    assert gradient.shape == grid.shape

    # load reference values from numerical HF raw file
    id = f"{str(sp.natom).zfill(3)}_q{str(charge).zfill(3)}_m{mult:02d}"
    fname = f"{id}_numeric_gradient.npy"
    ref_grad = np.load(f"{TEST_DATAPATH}/numeric/db/{fname}")

    # the change in the density is very steep close to the nuclei and the derivative of the
    # density is not well described by the spline interpolation. Therefore, here we only compare
    # the gradient at distances larger than half the covalent radius.
    radii_cutoff = sp.cov_radii["cordero"] / 2
    indx_radii = np.where(grid > radii_cutoff)
    assert np.allclose(gradient[indx_radii], ref_grad[indx_radii], atol=1e-3)


@pytest.mark.xfail(reason="High errors in spline derivative of order 2")
@pytest.mark.parametrize(
    "atom, charge, mult", [("H", 0, 2), ("H", -1, 1), ("Be", 0, 1), ("Cl", 0, 3), ("Ne", 0, 1)]
)
def test_numerical_hf_density_laplacian(atom, charge, mult):
    # load atomic and density data
    sp = load(atom, charge, mult, dataset="numeric", datapath=TEST_DATAPATH)

    # evaluate density and laplacian (second derivative of density spline) on the radial grid
    grid = sp.rs
    spline = sp.density_func(spin="ab", log=False)
    lapl = spline(grid, deriv=2)

    # check shape of arrays
    assert lapl.shape == grid.shape

    # load reference values from numerical HF raw files
    id = f"{str(sp.natom).zfill(3)}_q{str(charge).zfill(3)}_m{mult:02d}"
    fname = f"{id}_numeric_laplacian.npy"
    ref_lapl = np.load(f"{TEST_DATAPATH}/numeric/db/{fname}")

    # interpolated laplacian values against reference
    assert np.allclose(lapl, ref_lapl, atol=1e-6)
