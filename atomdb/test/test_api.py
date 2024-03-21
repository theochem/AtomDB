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

from scipy.interpolate import CubicSpline

import numpy as np
from atomdb.api import load


TEST_DATAPATH = "atomdb/test/data/"


@pytest.mark.parametrize(
    "dataset, spin, index, log, error",
    [
        ("gaussian", "alpha", None, False, ValueError),  # wrong spin value
        ("gaussian", "ab", None, True, ValueError),  # no log of gradient
        ("numeric", "a", None, False, ValueError),  # no property per alpha orbital
        ("numeric", "ab", [0, 1], False, ValueError),  # no property per orbital
    ],
)
def test_ddens_func_raised_errors(dataset, spin, index, log, error):
    # load Be atomic data and try to calculate the gradient of the density
    sp = load("Be", 0, 1, dataset=dataset, datapath=TEST_DATAPATH)

    with pytest.raises(error):
        sp.ddens_func(spin=spin, index=index, log=log)


def test_ddens_func():
    # Make a spline of the density derivative and evaluate it at the grid points.
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    points = np.linspace(0, 5, 20)

    # A) Check that the sum of the interpolated density derivatives for the alpha and
    # beta spin orbitals is consistent with the total density derivative.
    spline_d_dens_ab = sp.ddens_func(spin="ab")
    spline_d_dens_up = sp.ddens_func(spin="a")
    spline_d_dens_dn = sp.ddens_func(spin="b")

    expected_ddens = spline_d_dens_ab(points)
    test_ddens = spline_d_dens_up(points) + spline_d_dens_dn(points)
    assert np.allclose(test_ddens, expected_ddens, rtol=1e-6)

    # B) Check the interpolated derivative of the spin density from the 2nd and 3rd
    # molecular orbitals.
    d_dens_m_12 = np.sum(sp._orb_d_dens_up[[1, 2]], axis=0) - np.sum(
        sp._orb_d_dens_dn[[1, 2]], axis=0
    )
    spline_ddens_m = CubicSpline(sp.rs, d_dens_m_12)
    spline_d_dens_m = sp.ddens_func(spin="m", index=[1, 2])

    expected_ddens = spline_ddens_m(points)
    test_ddens = spline_d_dens_m(points)
    assert np.allclose(test_ddens, expected_ddens, rtol=1e-6)


def test_d2dens_func():
    # Make a spline of the second derivative of density and evaluate it at the grid points.
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    points = np.linspace(0, 5, 20)

    # A) Check that the sum of the interpolated order 2 density derivatives for the alpha and
    # beta spin orbitals is consistent with the total second derivative of density.
    spline_dd_dens_ab = sp.d2dens_func(spin="ab")
    spline_dd_dens_up = sp.d2dens_func(spin="a")
    spline_dd_dens_dn = sp.d2dens_func(spin="b")

    expected_d2dens = spline_dd_dens_ab(points)
    test_d2dens = spline_dd_dens_up(points) + spline_dd_dens_dn(points)
    assert np.allclose(test_d2dens, expected_d2dens, rtol=1e-6)

    # B) Check the interpolated order 2 derivative of the spin density from the 2nd and 3rd
    # molecular orbitals.
    d2dens_m_12 = np.sum(sp._orb_d_dens_up[[1, 2]], axis=0) - np.sum(
        sp._orb_d_dens_dn[[1, 2]], axis=0
    )
    spline_d2dens_m = CubicSpline(sp.rs, d2dens_m_12)
    spline_dd_dens_m = sp.d2dens_func(spin="m", index=[1, 2])

    expected_d2dens = spline_d2dens_m(points)
    test_d2dens = spline_dd_dens_m(points)
    assert np.allclose(test_d2dens, expected_d2dens, rtol=1e-6)
