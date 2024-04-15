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

from importlib_resources import files

import os

import pytest

import numpy as np

from scipy.interpolate import CubicSpline

from atomdb import load


# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


@pytest.mark.parametrize(
    "dataset, spin, index, log, error",
    [
        ("gaussian", "alpha", None, False, ValueError),  # wrong spin value
        ("gaussian", "ab", None, True, ValueError),  # no log of gradient
        ("numeric", "a", None, False, ValueError),  # no property per alpha orbital
        ("numeric", "ab", [0, 1], False, ValueError),  # no property per orbital
    ],
)
def test_d_dens_func_raised_errors(dataset, spin, index, log, error):
    # load Be atomic data and try to calculate the gradient of the density
    sp = load("Be", 0, 1, dataset=dataset, datapath=TEST_DATAPATH)

    with pytest.raises(error):
        sp.d_dens_func(spin=spin, index=index, log=log)


def test_d_dens_func():
    # Make a spline of the density derivative and evaluate it at the grid points.
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    points = np.linspace(0, 5, 20)

    # A) Check that the sum of the interpolated density derivatives for the alpha and
    # beta spin orbitals is consistent with the total density derivative.
    spline_d_dens_ab = sp.d_dens_func(spin="t")
    spline_d_dens_up = sp.d_dens_func(spin="a")
    spline_d_dens_dn = sp.d_dens_func(spin="b")

    expected_d_dens = spline_d_dens_ab(points)
    test_d_dens = spline_d_dens_up(points) + spline_d_dens_dn(points)
    assert np.allclose(test_d_dens, expected_d_dens, rtol=1e-6)

    # B) Check the interpolated derivative of the spin density from the 2nd and 3rd
    # molecular orbitals.
    # 1. get the density derivative values in the right shape (nbasis, ngrid)
    mo_d_dens_data_a = sp._data.mo_d_dens_a.reshape(sp.ao.nbasis, -1)
    mo_d_dens_data_b = sp._data.mo_d_dens_b.reshape(sp.ao.nbasis, -1)
    # 2. get the density derivative values for the 2nd and 3rd molecular orbitals and sum them
    mo_d_dens_a_12 = mo_d_dens_data_a[[2, 3]].sum(axis=0)
    mo_d_dens_b_12 = mo_d_dens_data_b[[2, 3]].sum(axis=0)
    # 3. find difference in density derivative values between alpha and beta spin orbitals
    d_dens_m_12 = mo_d_dens_a_12 - mo_d_dens_b_12
    # 4. create a spline of the density derivative values using CubicSpline and d_dens_func
    spline_d_dens_m_ref = CubicSpline(sp._data.rs, d_dens_m_12)
    spline_d_dens_m = sp.d_dens_func(spin="m", index=[1, 2])
    # reference spline at the grid points and compare with d_dens_func values at the grid points
    expected_d_dens = spline_d_dens_m_ref(points)
    test_d_dens = spline_d_dens_m(points)
    assert np.allclose(test_d_dens, expected_d_dens, rtol=1e-6)


def test_dd_dens_func():
    # Make a spline of the second derivative of density and evaluate it at the grid points.
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    points = np.linspace(0, 5, 20)

    # A) Check that the sum of the interpolated order 2 density derivatives for the alpha and
    # beta spin orbitals is consistent with the total second derivative of density.
    spline_dd_dens_ab = sp.dd_dens_func(spin="t")
    spline_dd_dens_up = sp.dd_dens_func(spin="a")
    spline_dd_dens_dn = sp.dd_dens_func(spin="b")

    expected_d2dens = spline_dd_dens_ab(points)
    test_d2dens = spline_dd_dens_up(points) + spline_dd_dens_dn(points)
    assert np.allclose(test_d2dens, expected_d2dens, rtol=1e-6)

    # B) Check the interpolated order 2 derivative of the spin density from the 2nd and 3rd
    # molecular orbitals.
    # 1. get the density 2 derivative values in the right shape (nbasis, ngrid)
    mo_dd_dens_data_a = sp._data.mo_dd_dens_a.reshape(sp.ao.nbasis, -1)
    mo_dd_dens_data_b = sp._data.mo_dd_dens_b.reshape(sp.ao.nbasis, -1)
    # 2. get the density 2 derivative values for the 2nd and 3rd molecular orbitals and sum them
    mo_dd_dens_a_12 = mo_dd_dens_data_a[[2, 3]].sum(axis=0)
    mo_dd_dens_b_12 = mo_dd_dens_data_b[[2, 3]].sum(axis=0)
    # 3. find difference in density 2 derivative values between alpha and beta spin orbitals
    dd_dens_m_12 = mo_dd_dens_a_12 - mo_dd_dens_b_12
    # 4. create a spline of the density derivative values using CubicSpline and d_dens_func
    spline_dd_dens_m_ref = CubicSpline(sp._data.rs, dd_dens_m_12)
    spline_dd_dens_m = sp.d_dens_func(spin="m", index=[1, 2])
    # reference spline at the grid points and compare with d_dens_func values at the grid points
    expected_dd_dens = spline_dd_dens_m_ref(points)
    test_dd_dens = spline_dd_dens_m(points)

    assert np.allclose(test_dd_dens, expected_dd_dens, rtol=1e-6)
