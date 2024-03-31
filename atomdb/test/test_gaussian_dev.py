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
import numpy as np
import os
from importlib_resources import files

# avoid import error on 'not dev' tests, these statements are run even when the tests are skipped
try:
    from atomdb.datasets.gaussian import eval_radial_d_density, eval_radial_dd_density
    from atomdb.datasets.gaussian import eval_orbs_radial_d_density, eval_orbs_radial_dd_density
    from gbasis.wrappers import from_iodata
    from gbasis.evals.density import evaluate_density_gradient, evaluate_density_laplacian
    from grid import UniformInteger, LinearInfiniteRTransform, AtomGrid
    from iodata import load_one
except ImportError:
    pass

# mark all tests in this file as development tests
pytestmark = pytest.mark.dev

# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


@pytest.mark.parametrize(
    "atom", ["atom_018_Ar_N18_M1_uhf_def2svpd_g09.fchk", "atom_001_H_N02_M1_uhf_def2svpd_g09.fchk"]
)
def test_eval_radial_d_density(atom):
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


@pytest.mark.parametrize(
    "atom", ["atom_018_Ar_N18_M1_uhf_def2svpd_g09.fchk", "atom_001_H_N02_M1_uhf_def2svpd_g09.fchk"]
)
def test_eval_radial_dd_density(atom):
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
