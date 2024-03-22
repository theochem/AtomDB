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
from numpy.testing import assert_equal, assert_almost_equal
import os

# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


@pytest.mark.dev
@pytest.mark.parametrize(
    "atom", ["atom_018_Ar_N18_M1_uhf_def2svpd_g09.fchk", "atom_001_H_N02_M1_uhf_def2svpd_g09.fchk"]
)
def test_eval_radial_d_density(atom):
    # conditional import of dev modules needed for this test
    try:
        from atomdb.datasets.gaussian import eval_radial_d_density
        from gbasis.wrappers import from_iodata
        from gbasis.evals.density import evaluate_density_gradient
        from grid import UniformInteger, LinearInfiniteRTransform, AtomGrid
        from iodata import load_one

        assert True
    except ImportError:
        pytest.skip("Gbasis or IOData modules not available, skipping test")

    # create atomic grid from 0 to 20 bohr
    oned = UniformInteger(npoints=100)
    rgrid = LinearInfiniteRTransform(1e-4, 20).transform_1d_grid(oned)
    atgrid = AtomGrid(rgrid, degrees=[10])

    # load the fchk file
    mol_data = load_one(os.path.join(TEST_DATAPATH, "gaussian", atom))
    ao_basis = from_iodata(mol_data)

    # one electron RDM from fchk file
    rdm = mol_data.one_rdms["scf"]

    # evaluate gradient of the electron density on the grid
    rho_grad = evaluate_density_gradient(rdm, ao_basis, atgrid.points)
    # evaluate derivative of rho vs r on the grid
    radial_d_rho = eval_radial_d_density(rdm, ao_basis, atgrid.points)
    # compute unitary projection of r in x,y,z
    unitvects = atgrid.points / np.linalg.norm(atgrid.points, axis=1)[:, None]

    # recover cartesian gradient from radial derivative (for spheric atoms) and compare
    rho_grad_cart = radial_d_rho[:, None] * unitvects

    assert np.allclose(rho_grad, rho_grad_cart, rtol=1e-6)


@pytest.mark.dev
@pytest.mark.parametrize(
    "atom", ["atom_018_Ar_N18_M1_uhf_def2svpd_g09.fchk", "atom_001_H_N02_M1_uhf_def2svpd_g09.fchk"]
)
def test_eval_radial_dd_density(atom):
    # conditional import of dev modules needed for this test
    try:
        from atomdb.datasets.gaussian import eval_radial_d_density, eval_radial_dd_density
        from gbasis.wrappers import from_iodata
        from gbasis.evals.density import evaluate_density_laplacian
        from grid import UniformInteger, LinearInfiniteRTransform, AtomGrid
        from iodata import load_one

        assert True
    except ImportError:
        pytest.skip("Gbasis or IOData modules not available, skipping test")

    # create atomic grid from 0 to 20 bohr
    oned = UniformInteger(npoints=100)
    rgrid = LinearInfiniteRTransform(1e-4, 20).transform_1d_grid(oned)
    atgrid = AtomGrid(rgrid, degrees=[10])

    # load the fchk file
    mol_data = load_one(os.path.join(TEST_DATAPATH, "gaussian", atom))
    ao_basis = from_iodata(mol_data)

    # one electron RDM from fchk file
    rdm = mol_data.one_rdms["scf"]

    # evaluate fisrt and second derivatives of rho vs r on the grid
    radial_d_rho = eval_radial_d_density(rdm, ao_basis, atgrid.points)
    radial_dd_rho = eval_radial_dd_density(rdm, ao_basis, atgrid.points)

    # evaluate gradient, hessian and laplacian of the electron density on the grid
    rho_lapl = evaluate_density_laplacian(rdm, ao_basis, atgrid.points)

    # compute laplacian from first and second radial derivatives (for spheric atoms) and compare
    rho_lapl_rec = radial_dd_rho + 2 * radial_d_rho / np.linalg.norm(atgrid.points, axis=1)
    assert np.allclose(rho_lapl, rho_lapl_rec, rtol=1e-6)


@pytest.mark.dev
@pytest.mark.parametrize(
    "atom", ["atom_018_Ar_N18_M1_uhf_def2svpd_g09.fchk", "atom_001_H_N02_M1_uhf_def2svpd_g09.fchk"]
)
def test_eval_orbs_radial_d_density(atom):
    # conditional import of dev modules needed for this test
    try:

        from atomdb.datasets.gaussian import eval_radial_d_density, eval_orbs_radial_d_density
        from gbasis.wrappers import from_iodata
        from grid import UniformInteger, LinearInfiniteRTransform, AtomGrid
        from iodata import load_one

        assert True
    except ImportError:
        pytest.skip("Gbasis or IOData modules not available, skipping test")

    # create atomic grid from 0 to 20 bohr
    oned = UniformInteger(npoints=100)
    rgrid = LinearInfiniteRTransform(1e-4, 20).transform_1d_grid(oned)
    atgrid = AtomGrid(rgrid, degrees=[10])

    # load the fchk file
    mol_data = load_one(os.path.join(TEST_DATAPATH, "gaussian", atom))
    ao_basis = from_iodata(mol_data)

    # one electron RDM and MO coefficients from fchk file
    rdm = mol_data.one_rdms["scf"]

    # evaluate derivative of rho vs r on the grid
    radial_d_rho = eval_radial_d_density(rdm, ao_basis, atgrid.points)

    # evaluate radial derivative of the orbital densities on the grid
    radial_d_rho_orbs = eval_orbs_radial_d_density(rdm, ao_basis, atgrid.points)
    # compute total density from orbital densities
    radial_d_rho_from_orbs = np.einsum("ij->j", radial_d_rho_orbs)

    assert np.allclose(radial_d_rho_from_orbs, radial_d_rho, rtol=1e-6)


@pytest.mark.dev
@pytest.mark.parametrize(
    "atom", ["atom_018_Ar_N18_M1_uhf_def2svpd_g09.fchk", "atom_001_H_N02_M1_uhf_def2svpd_g09.fchk"]
)
def test_eval_orbs_radial_dd_density(atom):
    # conditional import of dev modules needed for this test
    try:
        from atomdb.datasets.gaussian import eval_radial_dd_density, eval_orbs_radial_dd_density
        from gbasis.wrappers import from_iodata
        from grid import UniformInteger, LinearInfiniteRTransform, AtomGrid
        from iodata import load_one

        assert True
    except ImportError:
        pytest.skip("Gbasis or IOData modules not available, skipping test")

    # create atomic grid from 0 to 20 bohr
    oned = UniformInteger(npoints=100)
    rgrid = LinearInfiniteRTransform(1e-4, 20).transform_1d_grid(oned)
    atgrid = AtomGrid(rgrid, degrees=[10])

    # load the fchk file
    mol_data = load_one(os.path.join(TEST_DATAPATH, "gaussian", atom))
    ao_basis = from_iodata(mol_data)

    # one electron RDM from fchk file
    rdm = mol_data.one_rdms["scf"]

    # evaluate second derivatives of rho vs r on the grid
    radial_dd_rho = eval_radial_dd_density(rdm, ao_basis, atgrid.points)
    # evaluate second derivatives of the orbital densities on the grid
    radial_dd_rho_orbs = eval_orbs_radial_dd_density(rdm, ao_basis, atgrid.points)
    # compute total second derivative of the density
    recov_radial_dd_rho = np.sum(radial_dd_rho_orbs, axis=0)
    # compare with the total second derivative of the density
    assert np.allclose(recov_radial_dd_rho, radial_dd_rho, rtol=1e-6)


def test_compiled_gaussian_hf_data():
    ### Use Be atomic data as a test case
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)

    # check values of energy components
    answer = -14.55433897481303
    assert_almost_equal(sp.energy, answer, decimal=10)

    # load radial grid, total density, alpha and beta (radial) orbital densities, orbital energies
    grid = sp.rs
    dens = sp.dens_tot
    orb_dens_a = sp._orb_dens_up
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
    orb_dens_a = sp._orb_dens_up
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
