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
from atomdb.units import angstrom, amu
from atomdb.io.periodic import table_periodic as table


def test_periodic_basic():
    # check H
    assert_equal(table[(1, 1)].number, 1)
    assert_equal(table[(1, 1)].group, 1)
    assert_equal(table[(1, 1)].period, 1)
    assert_equal(table[(1, 1)].symbol, "H")
    assert_equal(table[(1, 1)].name, "Hydrogen")
    # check Si
    assert_equal(table[(14, 14)].number, 14)
    assert_equal(table[(14, 14)].group, 14)
    assert_equal(table[(14, 14)].period, 3)
    assert_equal(table[(14, 14)].symbol, "Si")
    assert_equal(table[(14, 14)].name, "Silicon")
    # check atomic masses
    assert_almost_equal(table[(7, 7)].mass, 14.006855*amu, decimal=6)
    assert_almost_equal(table[(41, 41)].mass, 92.90637*amu, decimal=6)
    # check pauling electronegativity
    assert_almost_equal(table[(1, 1)].eneg_pauling, 2.2, decimal=6)
    assert_almost_equal(table[(9, 9)].eneg_pauling, 3.98, decimal=6)
    # check dipole polarizability
    assert_almost_equal(table[(2, 2)].pold_crc, 0.204956*angstrom**3, decimal=6)
    assert_almost_equal(table[(2, 2)].pold_chu, 1.38, decimal=6)
    assert_almost_equal(table[(9, 9)].pold_crc, 0.557*angstrom**3, decimal=6)
    assert_almost_equal(table[(9, 9)].pold_chu, 3.8, decimal=6)
    # check c6-coefficients
    assert_almost_equal(table[(3, 3)].c6_chu, 1392.0, decimal=6)
    assert_almost_equal(table[(15, 15)].c6_chu, 185.0, decimal=6)


def test_periodic_radii():
    # check non-available radii
    assert_equal(table[(1, 1)].vdw_radius_truhlar, None)
    assert_equal(table[(2, 2)].cov_radius_bragg, None)
    assert_equal(table[(2, 2)].cov_radius_slater, None)
    assert_equal(table[(13, 13)].vdw_radius_bondi, None)
    assert_equal(table[(18, 18)].vdw_radius_dreiding, None)
    assert_equal(table[(36, 36)].vdw_radius_batsanov, None)
    assert_equal(table[(87, 87)].cr_radius, None)
    assert_equal(table[(100, 100)].vdw_radius_rt, None)
    assert_equal(table[(100, 100)].vdw_radius_mm3, None)
    assert_equal(table[(105, 105)].wc_radius, None)
    assert_equal(table[(118, 118)].vdw_radius_uff, None)
    # check covalent radii
    assert_almost_equal(table[(4, 4)].cov_radius_bragg, 1.15*angstrom, decimal=6)
    assert_almost_equal(table[(4, 4)].cov_radius_slater, 1.05*angstrom, decimal=6)
    assert_almost_equal(table[(29, 29)].cov_radius_cordero, 1.32*angstrom, decimal=6)
    # check van der waals radii
    assert_almost_equal(table[(6, 6)].vdw_radius_bondi, 1.7*angstrom, decimal=6)
    assert_almost_equal(table[(12, 12)].vdw_radius_uff, 3.021*angstrom/2, decimal=6)
    assert_almost_equal(table[(13, 13)].vdw_radius_truhlar, 1.84*angstrom, decimal=6)
    assert_almost_equal(table[(13, 13)].vdw_radius_rt, 2.03*angstrom, decimal=6)
    assert_almost_equal(table[(17, 17)].vdw_radius_dreiding, 3.9503*angstrom/2, decimal=6)
    assert_almost_equal(table[(41, 41)].vdw_radius_batsanov, 2.15*angstrom, decimal=6)
    assert_almost_equal(table[(90, 90)].vdw_radius_mm3, 2.74*angstrom, decimal=6)
    # check other radii
    assert_almost_equal(table[(2, 2)].wc_radius, 0.291*angstrom, decimal=6)
    assert_almost_equal(table[(19, 19)].cr_radius, 2.43*angstrom, decimal=6)
