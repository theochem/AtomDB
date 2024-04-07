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

from atomdb.datasets.slater import get_cs_occupations
from atomdb.datasets.slater import load_slater_wfn
from atomdb.datasets.slater import AtomicDensity

from importlib_resources import files
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_almost_equal, assert_equal
import os

# mark all tests in this file as development tests
pytestmark = pytest.mark.dev

# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.path.abspath(TEST_DATAPATH._paths[0])

# raw data path
RAW_DATAPATH = files("atomdb.datasets.slater.raw")
RAW_DATAPATH = os.path.abspath(RAW_DATAPATH._paths[0])


def test_get_cs_occupations():
    configurations = ["1S(2)2S(2)2P(1)", "K(2)L(8)3S(2)3P(3)"]
    results = [
        [{"1S": 1, "2S": 1, "2P": 1}, {"1S": 1, "2S": 1, "2P": 0}],
        [
            {"1S": 1, "2S": 1, "2P": 3, "3S": 1, "3P": 3},
            {"1S": 1, "2S": 1, "2P": 3, "3S": 1, "3P": 0},
        ],
    ]
    for configuration, result in zip(configurations, results):
        alpha_occ, beta_occ, _ = get_cs_occupations(configuration)
        for key, a_occ in alpha_occ.items():
            assert a_occ == result[0][key]
            assert beta_occ[key] == result[1][key]


@pytest.mark.parametrize("atom", ["al", "c", "cu"])
def test_load_slater_wfn_orbitals_occupation_simple(atom):

    cation = load_slater_wfn(atom, cation=True)
    anion = load_slater_wfn(atom, anion=True)
    neutral = load_slater_wfn(atom)

    cation_occ = cation["orbitals_occupation"]
    anion_occ = anion["orbitals_occupation"]
    neutral_occ = neutral["orbitals_occupation"]

    # check relation in the number of electrons between cation, anion and neutral
    assert np.sum(cation_occ) + 1 == np.sum(neutral_occ)
    assert np.sum(anion_occ) - 1 == np.sum(neutral_occ)


@pytest.mark.parametrize("atom", ["al", "c", "cu"])
def test_load_slater_wfn_orbitals_occupation_complex(atom):

    cation = load_slater_wfn(atom, cation=True)
    anion = load_slater_wfn(atom, anion=True)
    neutral = load_slater_wfn(atom)

    cation_occ = cation["orbitals_occupation"]
    anion_occ = anion["orbitals_occupation"]
    neutral_occ = neutral["orbitals_occupation"]

    # check that unpaired electrons go to alpha orbitals
    for occ in [cation_occ, neutral_occ, anion_occ]:
        # split alpha and beta orbitals
        a_occ = occ[: len(occ) // 2]
        b_occ = occ[len(occ) // 2 :]
        n_a_occ = np.sum(a_occ)
        n_b_occ = np.sum(b_occ)

        min_occ = np.min([n_a_occ, n_b_occ])
        # more alpha electrons than beta electrons
        assert np.sum(n_a_occ) >= np.sum(n_b_occ)
        # occupied beta orbitals have occupied alpha counterparts
        assert_array_equal(a_occ[:min_occ].ravel(), b_occ[:min_occ].ravel())


@pytest.mark.parametrize("atom", ["al", "c", "cu"])
def test_load_slater_wfn_orbitals(atom):

    cation = load_slater_wfn(atom, cation=True)
    anion = load_slater_wfn(atom, anion=True)
    neutral = load_slater_wfn(atom)

    cation_orbs = cation["orbitals"]
    anion_orbs = anion["orbitals"]
    neutral_orbs = neutral["orbitals"]

    cat_orbs_alpha = cation_orbs[: len(cation_orbs) // 2]
    cat_orbs_beta = cation_orbs[len(cation_orbs) // 2 :]

    neu_orbs_alpha = neutral_orbs[: len(neutral_orbs) // 2]
    neu_orbs_beta = neutral_orbs[len(neutral_orbs) // 2 :]

    an_orbs_alpha = anion_orbs[: len(anion_orbs) // 2]
    an_orbs_beta = anion_orbs[len(anion_orbs) // 2 :]

    # check alpha orbitals are in the same order for cation, neutral and anion
    for c_orb, ne_orb, a_orb in zip(cat_orbs_alpha, neu_orbs_alpha, an_orbs_alpha):
        assert c_orb == ne_orb
        assert c_orb == a_orb

    # check beta orbitals are in the same order for cation, neutral and anion
    for c_orb, ne_orb, a_orb in zip(cat_orbs_beta, neu_orbs_beta, an_orbs_beta):
        assert c_orb == ne_orb
        assert c_orb == a_orb


@pytest.mark.parametrize("atom", ["al", "c", "cu"])
def test_load_slater_wfn_orbitals_energy(atom):

    cation = load_slater_wfn(atom, cation=True)
    anion = load_slater_wfn(atom, anion=True)
    neutral = load_slater_wfn(atom)

    cation_e = cation["orbitals_energy"]
    anion_e = anion["orbitals_energy"]
    neutral_e = neutral["orbitals_energy"]

    cat_orbs_alpha = cation_e[: len(cation_e) // 2]
    cat_orbs_beta = cation_e[len(cation_e) // 2 :]

    neu_orbs_alpha = neutral_e[: len(neutral_e) // 2]
    neu_orbs_beta = neutral_e[len(neutral_e) // 2 :]

    an_orbs_alpha = anion_e[: len(anion_e) // 2]
    an_orbs_beta = anion_e[len(anion_e) // 2 :]

    # check that each array is sorted in ascending order
    a_orb_sets = [cat_orbs_alpha, neu_orbs_alpha, an_orbs_alpha]
    b_orb_sets = [cat_orbs_beta, neu_orbs_beta, an_orbs_beta]
    # for each set check that alpha and beta orbitals are sorted in ascending order
    for a_orbs, b_orbs in zip(a_orb_sets, b_orb_sets):
        for orbs in [a_orbs, b_orbs]:
            assert np.all(np.diff(orbs) >= 0)


def test_load_slater_wfn_orbitals_coefficients():

    cation = load_slater_wfn("c", cation=True)
    anion = load_slater_wfn("c", anion=True)
    neutral = load_slater_wfn("c")

    ca_1s_ref = np.array(
        [
            -0.0005984,
            0.1852569,
            0.1243627,
            0.7131840,
            0.0023454,
            0.0009096,
            -0.0001368,
            0.0000628,
        ]
    )

    neu_2p_ref = np.array(
        [
            0.0000552,
            0.0137471,
            0.0162926,
            0.2011747,
            0.3651183,
            0.3811596,
            0.0951923,
        ]
    )

    an_2s_ref = np.array(
        [
            -0.0001251,
            0.0128461,
            0.0309079,
            -0.1919809,
            -0.3898360,
            0.0663604,
            0.7370742,
            0.4387447,
            0.0022083,
        ]
    )

    assert np.allclose(ca_1s_ref, cation["orbitals_coeff"]["1S"].ravel())
    assert np.allclose(neu_2p_ref, neutral["orbitals_coeff"]["2P"].ravel())
    assert np.allclose(an_2s_ref, anion["orbitals_coeff"]["2S"].ravel())


def test_load_slater_wfn_orbitals_exp():

    cation = load_slater_wfn("cu", cation=True)
    anion = load_slater_wfn("cu", anion=True)
    neutral = load_slater_wfn("cu")

    cation_exp = cation["orbitals_exp"]
    anion_exp = anion["orbitals_exp"]
    neutral_exp = neutral["orbitals_exp"]

    cation_exp_P = np.array(
        [
            54.142279,
            42.542133,
            21.883549,
            18.160695,
            14.647103,
            11.953711,
            5.504517,
            4.493556,
            2.786438,
            1.272903,
        ]
    )
    anion_exp_S = np.array(
        [
            72.014363,
            41.772406,
            30.543805,
            21.425299,
            12.757196,
            10.917885,
            9.613085,
            4.283572,
            3.026154,
            1.863101,
            0.444541,
            0.342667,
            0.205342,
        ]
    )
    neu_exp_D = np.array(
        [26.287704, 20.461389, 8.922963, 5.814383, 3.626597, 2.200138, 1.349920, 0.690642]
    )

    assert np.allclose(cation_exp["P"].ravel(), cation_exp_P)
    assert np.allclose(anion_exp["S"].ravel(), anion_exp_S)
    assert np.allclose(neutral_exp["D"].ravel(), neu_exp_D)


def test_load_slater_wfn_basis():

    cation = load_slater_wfn("al", cation=True)
    anion = load_slater_wfn("al", anion=True)
    neutral = load_slater_wfn("al")

    cation_basis = cation["orbitals_basis"]
    anion_basis = anion["orbitals_basis"]
    neutral_basis = neutral["orbitals_basis"]

    ref_cat_S_basis = ["3S", "1S", "2S", "1S", "2S", "3S", "2S", "3S", "3S", "1S"]
    ref_neu_P_basis = ["2P", "2P", "2P", "2P", "2P", "2P", "2P", "2P", "2P", "3P"]
    ref_an_S_basis = ["2S", "1S", "2S", "2S", "1S", "2S", "2S", "1S", "1S", "1S", "1S"]

    assert all([a == b for a, b in zip(cation_basis["S"], ref_cat_S_basis)])
    assert all([a == b for a, b in zip(neutral_basis["P"], ref_neu_P_basis)])
    assert all([a == b for a, b in zip(anion_basis["S"], ref_an_S_basis)])


def test_load_slater_wfn_basis_numbers():

    cation = load_slater_wfn("al", cation=True)
    anion = load_slater_wfn("al", anion=True)
    neutral = load_slater_wfn("al")

    cation_basis = cation["basis_numbers"]
    anion_basis = anion["basis_numbers"]
    neutral_basis = neutral["basis_numbers"]

    ref_cat_S_basis = ["3S", "1S", "2S", "1S", "2S", "3S", "2S", "3S", "3S", "1S"]
    ref_neu_P_basis = ["2P", "2P", "2P", "2P", "2P", "2P", "2P", "2P", "2P", "3P"]
    ref_an_S_basis = ["2S", "1S", "2S", "2S", "1S", "2S", "2S", "1S", "1S", "1S", "1S"]

    assert all([a == int(b[0]) for a, b in zip(cation_basis["S"].ravel(), ref_cat_S_basis)])
    assert all([a == int(b[0]) for a, b in zip(neutral_basis["P"].ravel(), ref_neu_P_basis)])
    assert all([a == int(b[0]) for a, b in zip(anion_basis["S"].ravel(), ref_an_S_basis)])


def test_atomic_density_heavy_cs():
    r"""Test integration of atomic density of carbon anion and cation."""
    # These files don't exist.
    with pytest.raises(ValueError):
        AtomicDensity("cs", cation=True)
    with pytest.raises(ValueError):
        AtomicDensity("cs", anion=True)

    cs = AtomicDensity("cs")
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = cs.atomic_density(grid, mode="total")
    assert_almost_equal(4 * np.pi * np.trapz(dens * grid**2.0, grid), 55.0, decimal=5)


def test_atomic_density_heavy_rn():
    r"""Test integration of atomic density of carbon anion and cation."""

    # These files don't exist.
    with pytest.raises(ValueError):
        AtomicDensity("rn", cation=True)
    with pytest.raises(ValueError):
        AtomicDensity("rn", anion=True)
    rn = AtomicDensity("rn")
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = rn.atomic_density(grid, mode="total")
    assert_almost_equal(4 * np.pi * np.trapz(dens * grid**2.0, grid), 86, decimal=5)


# def test_kinetic_energy_heavy_element_ce():
#     c = AtomicDensity("ce")
#     grid = np.arange(0.0, 25.0, 0.0001)
#     dens = c.lagrangian_kinetic_energy(grid)
#     assert_almost_equal(np.trapz(dens, grid), c.energy[0], decimal=3)


def test_raises():
    with pytest.raises(TypeError):
        AtomicDensity(25)
    with pytest.raises(TypeError):
        AtomicDensity("be2")
    with pytest.raises(ValueError):
        AtomicDensity.slater_orbital(np.array([[1]]), np.array([[2]]), np.array([[1.0]]))
    c = AtomicDensity("c")
    with pytest.raises(ValueError):
        c.atomic_density(np.array([[1.0]]), "not total")


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
