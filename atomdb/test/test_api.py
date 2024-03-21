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
from atomdb.api import load


TEST_DATAPATH = "atomdb/test/data/"


@pytest.mark.parametrize(
    "dataset, spin, index, log, error",
    [
        ("gaussian", "alpha", None, False, ValueError),  # wrong spin value
        ("gaussian", "ab", None, True, ValueError),  # no log of gradient
        ("gaussian", "ab", 1, True, TypeError),  # wrong index value
        ("numeric", "a", None, False, ValueError),  # no property per alpha orbital
        ("numeric", "ab", [0, 1], False, ValueError),  # no property per orbital
    ],
)
def test_gradient_func_raised_errors(dataset, spin, index, log, error):
    # load Be atomic data and try to calculate the gradient of the density
    sp = load("Be", 0, 1, dataset=dataset, datapath=TEST_DATAPATH)

    with pytest.raises(error):
        sp.gradient_func(spin=spin, index=index, log=log)


def ddens_cases():
    # load the atomic data and return the density derivatives from all,
    # all alpha, 2nd and 3rd alpha, and 2nd and 3rd  alpha + beta spin orbitals.
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    ddens_a_12 = np.sum(sp._orb_d_dens_up[[1, 2]], axis=0)
    ddens_ab_12 = np.sum(sp._orb_d_dens_up[[1, 2]], axis=0) + np.sum(
        sp._orb_d_dens_dn[[1, 2]], axis=0
    )

    for case in [
        ("ab", None, sp.d_dens_tot),
        ("a", None, np.sum(sp._orb_d_dens_up, axis=0)),
        ("a", [1, 2], ddens_a_12),
        ("ab", [1, 2], ddens_ab_12),
    ]:
        yield case


@pytest.mark.parametrize(
    "spin, index, expected_ddens", ddens_cases(),
)
def test_gradient_func(spin, index, expected_ddens):
    # Make a spline of the density derivative and evaluate it at the grid points.
    # Compare with the stored data for the gradient.
    sp = load("Be", 0, 1, dataset="gaussian", datapath=TEST_DATAPATH)
    spline_d_dens = sp.gradient_func(spin=spin, index=index)
    d_dens = spline_d_dens(sp.rs)
    assert np.allclose(d_dens, expected_ddens, rtol=1e-6)
