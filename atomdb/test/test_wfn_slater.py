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
#

import pytest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from atomdb.api import load


TEST_DATAPATH = "atomdb/test/data/"


# FIXME:sign error in parsed energy value (looks like T value instead of E was parsed from raw file).
# This is a temporal fix until the parsing code for Slater's data gets updated from BFit.
fix_sign = lambda value: -value


@pytest.mark.parametrize(
    "atom, mult, answer",
    [
        ("He", 1, -2.861679996),
        ("Be", 1, -14.573023167),
        ("Ag", 2, -5197.698467674),
        ("Ne", 1, -128.547098079),
    ],
)
def test_slater_energy_especies(atom, mult, answer):
    # load species and check energy
    sp = load(atom, 0, mult, dataset="slater", datapath=TEST_DATAPATH)
    assert_almost_equal(fix_sign(sp.energy), answer, decimal=6)


@pytest.mark.parametrize(
    "atom, charge, mult, tol",
    [
        ("He", 0, 1, 4),
        ("Be", 0, 1, 5),
        ("B", 0, 2, 3),
        ("Cl", 0, 2, 3),
        ("Ag", 0, 2, 2),
        ("C", 1, 2, 6),
        ("C", -1, 4, 5),
    ],
)
def test_slater_positive_definite_kinetic_energy(atom, charge, mult, tol):
    # load atomic and density data
    sp = load(atom, charge, mult, dataset="slater", datapath=TEST_DATAPATH)
    # get KED computed on an equally distant radial grid
    grid = sp.rs
    energ = sp.ked_tot
    integral = np.trapz(energ, grid)
    answer = -fix_sign(sp.energy)
    # assert np.all(np.abs(integral - answer) < tol)
    assert_almost_equal(integral, answer, decimal=tol)
    # check interpolated density
    spline = sp.ked_func(spin="ab", log=False)
    assert np.allclose(spline(grid), energ, atol=1e-6)


@pytest.mark.parametrize(
    "atom, mult, num_elect", [("H", 2, 1.0), ("Be", 1, 4.0), ("C", 3, 6.0), ("Ne", 1, 10.0)]
)
def test_slater_atomic_density(atom, mult, num_elect):
    # load Be atomic and density data
    sp = load(atom, 0, mult, dataset="slater", datapath=TEST_DATAPATH)

    # get radial grid points, total density, and its spline interpolation
    grid = sp.rs
    dens = sp.dens_tot
    spline = sp.density_func(spin="ab", log=True)

    # check shape of density and radial grid
    assert dens.shape == grid.shape

    # check density integrates to the number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), num_elect, decimal=3)

    # check interpolated density values compared to reference values
    assert np.allclose(spline(grid), dens, atol=1e-6)


@pytest.mark.parametrize(
    "atom, charge, mult, num_elect", [("H", -1, 1, 2.0), ("C", -1, 4, 7.0), ("C", 1, 2, 5.0)]
)
def test_slater_atomic_density_ions(atom, charge, mult, num_elect):
    # load atomic and density data
    sp = load(atom, charge, mult, dataset="slater", datapath=TEST_DATAPATH)

    # get radial grid points, total density, and its spline interpolation
    grid = sp.rs
    dens = sp.dens_tot
    spline = sp.density_func(spin="ab", log=True)

    # check shape of density and radial grid
    assert dens.shape == grid.shape

    # check density integrates to the number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), num_elect, decimal=3)

    # check interpolated density values compared to reference values
    assert np.allclose(spline(grid), dens, atol=1e-6)


@pytest.mark.parametrize(
    "atom, charge, mult", [("H", 0, 2), ("Be", 0, 1), ("Cl", 0, 2), ("Ne", 0, 1)]
)
def test_slater_atomic_density_gradient(atom, charge, mult):
    # load atomic and density data and get density derivative evaluated on a radial grid
    sp = load(atom, charge, mult, dataset="slater", datapath=TEST_DATAPATH)
    grid = sp.rs
    spline = sp.density_func(spin="ab", log=True)
    gradient = spline(grid, deriv=1)

    # get reference values from Slater wfn raw files
    id = f"{str(sp.natom).zfill(3)}_q{str(charge).zfill(3)}_m{mult:02d}"
    fname = f"{id}_slater_gradient.npy"
    answer = np.load(f"{TEST_DATAPATH}/slater/db/{fname}")

    # check shape of arrays and array elements
    assert gradient.shape == answer.shape

    # check interpolated density gradient values compared to reference values
    assert np.allclose(gradient, answer, rtol=1e-3)


def test_slater_h_anion_density_splines():
    # load H^- atomic and density data and  get density evaluated on an equally
    # distant radial grid: np.arange(0.0, 15.0, 0.00001)
    charge = -1
    mult = 1
    sp = load("H", charge, mult, dataset="slater", datapath=TEST_DATAPATH)
    grid = sp.rs
    dens = sp.dens_tot
    # check interpolated densities
    spline_dens = sp.density_func(spin="ab", log=True)
    assert np.allclose(spline_dens(grid), dens, atol=1e-6)
    # check interpolated kinetic energy density
    spline = sp.ked_func(spin="ab", log=False)
    assert np.allclose(spline(grid), sp.ked_tot, atol=1e-6)

    # load reference values for gradient
    id = f"{str(sp.natom).zfill(3)}_q{str(charge).zfill(3)}_m{mult:02d}"
    fname = f"{id}_slater_gradient.npy"
    gradient = np.load(f"{TEST_DATAPATH}/slater/db/{fname}")

    # check density gradient (spline derivative) vs reference values
    assert np.allclose(spline_dens(grid, deriv=1), gradient, atol=1e-6)

    # check density spline second derivative vs derivative of reference gradient values
    # FIXME: second derivative has high error
    d2dens = np.gradient(gradient, sp.rs)
    assert np.allclose(spline_dens(grid, deriv=2), d2dens, atol=1e-2)


def test_slater_missing_attributes():
    # load He data
    sp = load("He", 0, 1, dataset="slater", datapath=TEST_DATAPATH)
    # check missing attributes default to None
    assert sp.ip is None
    assert sp.mu is None
    assert sp.eta is None
    assert sp._orb_dens_up is None
    assert sp._orb_dens_dn is None


def test_slater_orbitals_be():
    # Load Be data
    sp = load("Be", 0, 1, dataset="slater", datapath=TEST_DATAPATH)

    # check mo energy and occupation arrays
    # check array shapes
    assert len(sp._mo_energy_a) == 2
    assert len(sp._mo_energy_a) == len(sp._mo_energy_b)
    assert len(sp._mo_occs_a) == 2
    assert len(sp._mo_occs_a) == len(sp._mo_occs_b)
    # check array elements
    assert np.allclose(sp._mo_energy_a, np.array([-4.7326699, -0.3092695]), atol=1e-6)
    assert np.allclose(sp._mo_energy_b, np.array([-4.7326699, -0.3092695]), atol=1e-6)
    assert np.allclose(sp._mo_occs_a, np.array([1.0, 1.0]), atol=1e-6)
    assert np.allclose(sp._mo_occs_b, np.array([1.0, 1.0]), atol=1e-6)


def test_slater_orbitals_ne():
    # Load Ne data
    sp = load("Ne", 0, 1, dataset="slater", datapath=TEST_DATAPATH)

    # check mo energy and occupation arrays
    # check array shapes
    assert len(sp._mo_energy_a) == 3
    assert len(sp._mo_energy_a) == len(sp._mo_energy_b)
    assert len(sp._mo_occs_a) == 3
    assert len(sp._mo_occs_a) == len(sp._mo_occs_b)
    # check array elements
    assert np.allclose(sp._mo_energy_a, np.array([-32.7724425, -1.9303907, -0.8504095]), atol=1e-6)
    assert np.allclose(sp._mo_energy_b, np.array([-32.7724425, -1.9303907, -0.8504095]), atol=1e-6)
    assert np.allclose(sp._mo_occs_a, np.array([1.0, 1.0, 3.0]), atol=1e-6)
    assert np.allclose(sp._mo_occs_b, np.array([1.0, 1.0, 3.0]), atol=1e-6)


def test_slater_orbitals_h():
    # Load H data
    sp = load("H", 0, 2, dataset="slater", datapath=TEST_DATAPATH)

    # check mo energy and occupation arrays
    # check array shapes
    assert len(sp._mo_energy_a) == 1
    assert len(sp._mo_energy_a) == len(sp._mo_energy_b)
    assert len(sp._mo_occs_a) == 1
    assert len(sp._mo_occs_a) == len(sp._mo_occs_b)
    # check array elements
    assert np.allclose(sp._mo_energy_a, np.array([-0.50]), atol=1e-6)
    assert np.allclose(sp._mo_energy_b, np.array([-0.50]), atol=1e-6)
    assert np.allclose(sp._mo_occs_a, np.array([1.0]), atol=1e-6)
    assert np.allclose(sp._mo_occs_b, np.array([0.0]), atol=1e-6)


def test_slater_orbitals_h_anion():
    # Load H data
    sp = load("H", -1, 1, dataset="slater", datapath=TEST_DATAPATH)

    # check mo energy and occupation arrays
    # check array shapes
    assert len(sp._mo_energy_a) == 1
    assert len(sp._mo_energy_a) == len(sp._mo_energy_b)
    assert len(sp._mo_occs_a) == 1
    assert len(sp._mo_occs_a) == len(sp._mo_occs_b)
    # check array elements
    assert np.allclose(sp._mo_energy_a, np.array([-0.0462224]), atol=1e-6)
    assert np.allclose(sp._mo_energy_b, np.array([-0.0462224]), atol=1e-6)
    assert np.allclose(sp._mo_occs_a, np.array([1.0]), atol=1e-6)
    assert np.allclose(sp._mo_occs_b, np.array([1.0]), atol=1e-6)


def test_slater_orbitals_ag():
    # Load the Ag file.
    sp = load("Ag", 0, 2, dataset="slater", datapath=TEST_DATAPATH)

    # check mo energy and occupation arrays
    # check array shapes
    assert len(sp._mo_energy_a) == 10
    assert len(sp._mo_energy_a) == len(sp._mo_energy_b)
    assert len(sp._mo_occs_a) == 10
    assert len(sp._mo_occs_a) == len(sp._mo_occs_b)
    # check array elements
    energy = np.array(
        [
            -913.8355964,
            -134.8784068,
            -25.9178242,
            -4.0014988,
            -0.2199797,
            -125.1815809,
            -21.9454343,
            -2.6768201,
            -14.6782003,
            -0.5374007,
        ]
    )
    assert np.allclose(sp._mo_energy_a, energy, atol=1e-6)
    assert np.allclose(sp._mo_energy_b, energy, atol=1e-6)
    assert np.allclose(
        sp._mo_occs_a, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 5.0, 5.0]), atol=1e-6
    )
    assert np.allclose(
        sp._mo_occs_b, np.array([1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 3.0, 3.0, 5.0, 5.0]), atol=1e-6
    )


#
# TODO: Update tests bellow
#
# def test_atomic_density_heavy_cs():
#     r"""Test integration of atomic density of carbon anion and cation."""
#     # These files don't exist.
#     assert_raises(ValueError, AtomicDensity, "cs", cation=True)
#     assert_raises(ValueError, AtomicDensity, "cs", anion=True)

#     cs = AtomicDensity("cs")
#     grid = np.arange(0.0, 40.0, 0.0001)
#     dens = cs.atomic_density(grid, mode="total")
#     assert_almost_equal(4 * np.pi * np.trapz(dens * grid ** 2., grid), 55.0, decimal=5)


# def test_atomic_density_heavy_rn():
#     r"""Test integration of atomic density of carbon anion and cation."""
#     # These files don't exist.
#     assert_raises(ValueError, AtomicDensity, "rn", cation=True)
#     assert_raises(ValueError, AtomicDensity, "rn", anion=True)

#     rn = AtomicDensity("rn")
#     grid = np.arange(0.0, 40.0, 0.0001)
#     dens = rn.atomic_density(grid, mode="total")
#     assert_almost_equal(4 * np.pi * np.trapz(dens * grid ** 2., grid), 86, decimal=5)


# def test_kinetic_energy_heavy_element_ce():
#     c = AtomicDensity("ce")
#     grid = np.arange(0.0, 25.0, 0.0001)
#     dens = c.lagrangian_kinetic_energy(grid)
#     assert_almost_equal(np.trapz(dens, grid), c.energy[0], decimal=3)


# def test_raises():
#     assert_raises(TypeError, AtomicDensity, 25)
#     assert_raises(TypeError, AtomicDensity, "be2")
#     assert_raises(ValueError, AtomicDensity.slater_orbital, np.array([[1]]), np.array([[2]]),
#                   np.array([[1.]]))
#     c = AtomicDensity("c")
#     assert_raises(ValueError, c.atomic_density, np.array([[1.]]), "not total")


# def test_parsing_slater_density_k():
#     # Load the K file
#     k = load_slater_wfn("k")

#     assert k["configuration"] == "K(2)L(8)3S(2)3P(6)4S(1)"
#     assert k["energy"] == [599.164786943]

#     assert k['orbitals'] == ["1S", "2S", "3S", "4S", "2P", "3P"]
#     assert np.all(abs(k['orbitals_cusp'] - np.array([1.0003111, 0.9994803, 1.0005849, 1.0001341,
#                                                      1.0007902, 0.9998975])[:, None]) < 1.e-6)
#     assert np.all(abs(k['orbitals_energy'] - np.array([-133.5330493, -14.4899575, -1.7487797, -0.1474751, -11.5192795,
#                                                        -0.9544227])[:, None]) < 1.e-6)
#     assert k['orbitals_basis']['P'] == ['2P', '3P', '2P', '3P', '2P', '2P', '3P', '2P', '2P', '2P']

#     basis_numbers = np.array([[2], [3], [2], [3], [2], [2], [3], [2], [2], [2]])
#     assert np.all(np.abs(k['basis_numbers']['P'] - basis_numbers) < 1e-5)

#     # Check coefficients of 3P orbital
#     coeff_3P = np.array([0.0000354, 0.0011040, -0.0153622, 0.0620133, -0.1765320, -0.3537264,
#                          -0.3401560, 1.3735350, 0.1055549, 0.0010773])
#     assert (abs(k['orbitals_coeff']['3P'] - coeff_3P.reshape(10, 1)) < 1.e-6).all()


# def test_parsing_slater_density_i():
#     i = load_slater_wfn("i")
#     assert i["configuration"] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(5)"
#     assert i["orbitals"] == ["1S", "2S", "3S", "4S", "5S", "2P", "3P", "4P", "5P", "3D", "4D"]
#     occupation = np.array([[2], [2], [2], [2], [2], [6], [6], [6], [5], [10], [10]])
#     assert (i["orbitals_occupation"] == occupation).all()

#     exponents = np.array([[58.400845, 45.117174, 24.132009, 20.588554, 12.624386, 10.217388,
#                              8.680013, 4.627159, 3.093797, 1.795536, 0.897975]])
#     assert np.all(np.abs(i["orbitals_exp"]["D"] - exponents.reshape((11, 1))) < 1e-4)


# def test_parsing_slater_density_xe():
#     # Load the Xe file
#     xe = load_slater_wfn("xe")
#     assert xe['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)"
#     assert xe['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "2P", "3P", "4P", "5P", "3D", "4D"]
#     occupation = np.array([[2], [2], [2], [2], [2], [6], [6], [6], [6], [10], [10]])
#     assert (xe["orbitals_occupation"] == occupation).all()
#     assert xe['energy'] == [7232.138367196]

#     # Check coeffs of D orbitals.
#     coeffs = np.array([-0.0006386, -0.0030974, 0.0445101, -0.1106186, -0.0924762, -0.4855794,
#                      0.1699923, 0.7240230, 0.3718553, 0.0251152, 0.0001040])
#     assert (abs(xe['orbitals_coeff']["4D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()


# def test_parsing_slater_density_xe_cation():
#     # Load the Xe file
#     np.testing.assert_raises(ValueError, load_slater_wfn, "xe", anion=True)
#     xe = load_slater_wfn("xe", cation=True)
#     assert xe['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(5)"
#     assert xe['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "2P", "3P", "4P", "5P", "3D", "4D"]
#     occupation = np.array([[2], [2], [2], [2], [2], [6], [6], [6], [5], [10], [10]])
#     assert (xe["orbitals_occupation"] == occupation).all()
#     assert xe['energy'] == [7231.708943551]

#     # Check coefficients of D orbitals.
#     coeff = np.array([-0.0004316, -0.0016577, -0.0041398, -0.2183952, 0.0051908, -0.2953384,
#                      -0.0095762, 0.6460145, 0.4573096, 0.0431928, -0.0000161])
#     assert (abs(xe['orbitals_coeff']["4D"] - coeff.reshape((len(coeff), 1))) < 1e-10).all()


# def test_parsing_slater_heavy_atom_cs():
#     np.testing.assert_raises(ValueError, load_slater_wfn, "cs", cation=True)
#     np.testing.assert_raises(ValueError, load_slater_wfn, "cs", anion=True)
#     # Load the Xe file
#     cs = load_slater_wfn("cs")
#     assert cs['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)6S(1)"
#     assert cs['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S", "2P", "3P", "4P", "5P", "3D",
#                               "4D"]
#     occupation = np.array([[2], [2], [2], [2], [2], [1], [6], [6], [6], [6], [10], [10]])
#     assert (cs["orbitals_occupation"] == occupation).all()
#     assert cs['energy'] == [7553.933539793]

#     # Check coeffs of D orbitals.
#     coeffs = np.array([-0.0025615, -0.1930536, -0.2057559, -0.1179374, 0.4341816, 0.6417053,
#                        0.1309576])
#     assert (abs(cs['orbitals_coeff']["4D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()

#     # Check Exponents of D orbitals.
#     exps = np.array([32.852137, 18.354403, 14.523221, 10.312620, 7.919345, 5.157647,
#                      3.330606])
#     assert (abs(cs['orbitals_exp']["D"] - exps.reshape((len(exps), 1))) < 1e-10).all()


# def test_parsing_slater_heavy_atom_rn():
#     # Load the Xe file
#     rn = load_slater_wfn("rn")
#     assert rn['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)4F(14)6S(2)5D(10)6P(6)"
#     print(rn["orbitals"])
#     assert rn['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S", "2P", "3P", "4P", "5P", "6P",
#                               "3D", "4D", "5D", "4F"]
#     occupation = np.array([[2], [2], [2], [2], [2], [2], [6], [6], [6], [6], [6], [10], [10],
#                            [10], [14]])
#     assert (rn["orbitals_occupation"] == occupation).all()
#     assert rn['energy'] == [21866.772036482]

#     # Check coeffs of 4F orbitals.
#     coeffs = np.array([.0196357, .2541992, .4806186, -.2542278, .5847619, .0099519])
#     assert (abs(rn['orbitals_coeff']["4F"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()


# def test_parsing_slater_heavy_atom_lr():
#     # Load the lr file
#     lr = load_slater_wfn("lr")
#     assert lr['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)4F(14)6S(2)5D(10)6P(6)" \
#                                   "7S(2)6D(1)5F(14)"
#     assert lr['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S", "7S",
#                               "2P", "3P", "4P", "5P", "6P",
#                               "3D", "4D", "5D", "6D", "4F", "5F"]
#     occupation = np.array([[2], [2], [2], [2], [2], [2], [2], [6], [6], [6], [6], [6], [10], [10],
#                            [10], [1], [14], [14]])
#     assert (lr["orbitals_occupation"] == occupation).all()
#     assert lr['energy'] == [33557.949960623]

#     # Check coeffs of 3D orbitals.
#     coeffs = np.array([0.0017317, 0.4240510, 0.4821228, 0.1753365, -0.0207393, 0.0091584,
#                        -0.0020913, 0.0005398, -0.0001604, 0.0000443, -0.0000124])
#     assert (abs(lr['orbitals_coeff']["3D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()

#     # Check coeffs of 2S orbitals.
#     coeffs = np.array([0.0386739, -0.4000069, -0.2570804, 1.2022357, 0.0787866, 0.0002957,
#                        0.0002277, 0.0002397, -0.0005650, 0.0000087, 0.0000031, -0.0000024,
#                        0.0000008, -0.0000002, 0.0000001])
#     assert (abs(lr['orbitals_coeff']["2S"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()


# def test_parsing_slater_heavy_atom_dy():
#     # Load the dy file
#     dy = load_slater_wfn("dy")
#     assert dy['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)6S(2)5D(0)4F(10)"
#     assert dy['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S",
#                               "2P", "3P", "4P", "5P",
#                               "3D", "4D", "4F"]
#     occupation = np.array([[2], [2], [2], [2], [2], [2], [6], [6], [6], [6], [10], [10], [10]])
#     assert (dy["orbitals_occupation"] == occupation).all()
#     assert dy['energy'] == [11641.452478555]

#     # Check coeffs of 4D orbitals.
#     coeffs = np.array([-0.0016462, -0.2087639, -0.2407385, -0.1008913, 0.4844709, 0.6180159,
#                        0.1070867])
#     assert (abs(dy['orbitals_coeff']["4D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()
