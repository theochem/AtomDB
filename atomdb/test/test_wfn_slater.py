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


import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from atomdb.io.wfn_slater import AtomicDensity, load_slater_wfn


def slater(e, n, r, derivative=False):
    """Calculates single normalized slater function at a given point."""
    norm = np.power(2. * e, n) * np.sqrt(2. * e / np.math.factorial(2. * n))
    slater = norm * np.power(r, n - 1) * np.exp(-e * r)

    if derivative:
        return (((n - 1) / r) - e) * slater
    return slater


def test_slater_type_orbital_be():
    # load Be atomic wave function
    be = AtomicDensity("Be")
    # check values of a single orbital at r=1.0
    orbital = be.slater_orbital(np.array([[12.683501]]), np.array([[1]]), np.array([1]))
    assert_almost_equal(orbital, slater(12.683501, 1, 1.0), decimal=6)
    # check values of a single orbital at r=2.0
    orbital = be.slater_orbital(np.array([[0.821620]]), np.array([[2]]), np.array([2]))
    assert_almost_equal(orbital, slater(0.821620, 2, 2.0), decimal=6)
    # check value of tow orbitals at r=1.0 & r=2.0
    exps, nums = np.array([[12.683501], [0.821620]]), np.array([[1], [2]])
    orbitals = be.slater_orbital(exps, nums, np.array([1., 2.]))
    expected = np.array([[slater(exps[0, 0], nums[0, 0], 1.), slater(exps[1, 0], nums[1, 0], 1.)],
                         [slater(exps[0, 0], nums[0, 0], 2.), slater(exps[1, 0], nums[1, 0], 2.)]])
    assert_almost_equal(orbitals, expected, decimal=6)


def test_derivative_slater_type_orbital_be():
    # load Be atomic wave function
    be = AtomicDensity("Be")
    # check values of a single orbital at r=1.0
    orbital = be.derivative_slater_type_orbital(np.array([[12.683501]]),
                                                np.array([[1]]), np.array([1]))
    assert_almost_equal(orbital, slater(12.683501, 1, 1.0, derivative=True), decimal=6)
    # check values of a single orbital at r=2.0
    orbital = be.derivative_slater_type_orbital(np.array([[0.821620]]), np.array([[2]]),
                                                np.array([2]))
    assert_almost_equal(orbital, slater(0.821620, 2, 2.0, derivative=True), decimal=6)
    # check value of tow orbitals at r=1.0 & r=2.0
    exps, nums = np.array([[12.683501], [0.821620]]), np.array([[1], [2]])
    orbitals = be.derivative_slater_type_orbital(exps, nums, np.array([1., 2.]))
    expected = np.array([[slater(exps[0, 0], nums[0, 0], 1., True),
                          slater(exps[1, 0], nums[1, 0], 1., True)],
                         [slater(exps[0, 0], nums[0, 0], 2., True),
                          slater(exps[1, 0], nums[1, 0], 2., True)]])
    assert_almost_equal(orbitals, expected, decimal=6)


def test_positive_definite_kinetic_energy_he():
    # load he atomic wave function
    he = AtomicDensity("he")
    # compute density on an equally distant grid
    grid = np.arange(0., 30.0, 0.0001)
    energ = he.lagrangian_kinetic_energy(grid)
    integral = np.trapz(energ, grid)
    assert np.all(np.abs(integral - he.energy[0]) < 1e-4)


def test_positive_definite_kinetic_energy_be():
    # load be atomic wave function
    be = AtomicDensity("be")
    # compute density on an equally distant grid
    grid = np.arange(0., 30.0, 0.0001)
    energ = be.lagrangian_kinetic_energy(grid)
    integral = np.trapz(energ, grid)
    assert np.all(np.abs(integral - be.energy[0]) < 1e-5)


def test_positive_definite_kinetic_energy_most_atoms():
    # load c atomic wave function
    for atom in ["b", "cl", "ag"]:
        adens = AtomicDensity(atom)
        # compute density on an equally distant grid
        grid = np.arange(0., 30., 0.0005)
        energ = adens.lagrangian_kinetic_energy(grid)
        integral = np.trapz(energ, grid)
        assert np.all(np.abs(integral - adens.energy[0]) < 1e-3)


def test_phi_derivative_lcao_b():
    # load Be atomic wave function
    b = AtomicDensity("b")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = b.phi_matrix(np.array([1]), deriv=True)
    def _slater_deriv(r):
        # compute expected value of 1S
        phi1S = slater(16.109305, 2, r, True) * -0.0005529 + slater(7.628062, 1, r, True) * -0.23501
        phi1S += slater(6.135799, 2, r, True) * -0.1508924 + slater(4.167618, 1, r, True) * -0.64211
        phi1S += slater(2.488602, 1, r, True) * -0.0011507 + slater(1.642523, 2, r, True) * -0.00086
        phi1S += slater(0.991698, 1, r, True) * 0.0004712 + slater(0.787218, 1, r, True) * -0.000232
        # compute expected value of 2S
        phi2S = slater(16.109305, 2, r, True) * -0.0001239 + slater(7.628062, 1, r, True) * 0.012224
        phi2S += slater(6.135799, 2, r, True) * 0.0355967 + slater(4.167618, 1, r, True) * -0.198721
        phi2S += slater(2.488602, 1, r, True) * -0.5378967 + slater(1.642523, 2, r, True) * -0.11997
        phi2S += slater(0.991698, 1, r, True) * 1.4382402 + slater(0.787218, 1, r, True) * 0.0299258
        # compute expected value of 1P
        phi3S = slater(12.135370, 3, r, True) * 0.0000599 + slater(5.508493, 2, r, True) * 0.0113751
        phi3S += slater(3.930298, 3, r, True) * 0.0095096 + slater(2.034395, 2, r, True) * 0.1647518
        phi3S += slater(1.301082, 2, r, True) * 0.3367860 + slater(0.919434, 2, r, True) * 0.4099162
        phi3S += slater(0.787218, 2, r, True) * 0.1329396
        return [phi1S, phi2S, phi3S]
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 3))
    assert_almost_equal(phi_matrix, np.array([_slater_deriv(1)]), decimal=4)
    # check the values of the phi_matrix at point 1.0, 2.0, 3.0
    phi_matrix = b.phi_matrix(np.array([1., 2., 3.]), True)
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (3, 3))
    assert_almost_equal(phi_matrix[0, :], _slater_deriv(1), decimal=4)
    assert_almost_equal(phi_matrix[1, :], _slater_deriv(2.), decimal=4)
    assert_almost_equal(phi_matrix[2, :], _slater_deriv(3.), decimal=4)


def test_coeff_matrix_be():
    # load Be atomic wave function
    be = AtomicDensity("be")
    # using one _grid point at 1.0
    coeff_1s = np.array([-0.0024917, 0.0314015, 0.0849694, 0.8685562,
                         0.0315855, -0.0035284, -0.0004149, .0012299])[:, None]
    coeff_2s = np.array([0.0004442, -0.0030990, -0.0367056, 0.0138910,
                         -0.3598016, -0.2563459, 0.2434108, 1.1150995])[:, None]
    assert_almost_equal(be.orbitals_coeff["1S"], coeff_1s, decimal=6)
    assert_almost_equal(be.orbitals_coeff["2S"], coeff_2s, decimal=6)


def test_phi_lcao_be():
    # load Be atomic wave function
    be = AtomicDensity("BE")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = be.phi_matrix(np.array([1]))
    # compute expected value of 1S
    phi1S = slater(12.683501, 1, 1) * -0.0024917 + slater(8.105927, 1, 1) * 0.0314015
    phi1S += slater(5.152556, 1, 1) * 0.0849694 + slater(3.472467, 1, 1) * 0.8685562
    phi1S += slater(2.349757, 1, 1) * 0.0315855 + slater(1.406429, 1, 1) * -0.0035284
    phi1S += slater(0.821620, 2, 1) * -0.0004149 + slater(0.786473, 1, 1) * 0.0012299
    # compute expected value of 2S
    phi2S = slater(12.683501, 1, 1) * 0.0004442 + slater(8.105927, 1, 1) * -0.0030990
    phi2S += slater(5.152556, 1, 1) * -0.0367056 + slater(3.472467, 1, 1) * 0.0138910
    phi2S += slater(2.349757, 1, 1) * -0.3598016 - slater(1.406429, 1, 1) * 0.2563459
    phi2S += slater(0.821620, 2, 1) * 0.2434108 + slater(0.786473, 1, 1) * 1.1150995
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 2))
    assert_almost_equal(phi_matrix, np.array([[phi1S, phi2S]]), decimal=6)
    # check the values of the phi_matrix at point 1.0, 2.0, 3.0
    phi_matrix = be.phi_matrix(np.array([1., 2., 3.]))
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (3, 2))
    assert_almost_equal(phi_matrix[0, :], np.array([phi1S, phi2S]), decimal=6)


def test_orbitals_function_be():
    # load Be atomic wave function
    be = AtomicDensity("bE")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = be.phi_matrix(np.array([1]))
    # compute expected value of 1S
    phi1S = slater(12.683501, 1, 1.) * -0.0024917 + slater(8.105927, 1, 1.) * 0.0314015
    phi1S += slater(5.152556, 1, 1.) * 0.0849694 + slater(3.472467, 1, 1.) * 0.8685562
    phi1S += slater(2.349757, 1, 1.) * 0.0315855 + slater(1.406429, 1, 1.) * -0.0035284
    phi1S += slater(0.821620, 2, 1.) * -0.0004149 + slater(0.786473, 1, 1.) * 0.00122991
    # compute expected value of 2S
    phi2S = slater(12.683501, 1, 1.) * 0.0004442 + slater(8.105927, 1, 1.) * -0.0030990
    phi2S += slater(5.152556, 1, 1.) * -0.0367056 + slater(3.472467, 1, 1.) * 0.0138910
    phi2S += slater(2.349757, 1, 1.) * -0.3598016 + slater(1.406429, 1, 1.) * -0.2563459
    phi2S += slater(0.821620, 2, 1.) * 0.2434108 + slater(0.786473, 1, 1.) * 1.1150995
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 2))
    assert_almost_equal(phi_matrix, np.array([[phi1S, phi2S]]), decimal=6)


def test_orbitals_norm_be():
    # load Be atomic wave function
    be = AtomicDensity("be")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = be.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 2))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)


def test_orbitals_norm_ne():
    # load Ne atomic wave function
    ne = AtomicDensity("ne")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 10.0, 0.0001)
    dens = ne.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 3))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 2], grid), 1.0, decimal=6)


def test_orbitals_norm_c():
    # load C atomic wave function
    c = AtomicDensity("c")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = c.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 3))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 2], grid), 1.0, decimal=6)


def test_atomic_density_be():
    # load Be atomic wave function
    be = AtomicDensity("be")
    # compute density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = be.atomic_density(grid, mode="total")
    core = be.atomic_density(grid, mode="core")
    valn = be.atomic_density(grid, mode="valence")
    # check shape
    assert_equal(dens.shape, grid.shape)
    assert_equal(core.shape, grid.shape)
    assert_equal(valn.shape, grid.shape)
    # check dens = core + valence
    assert_almost_equal(dens, core + valn, decimal=6)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), 4.0, decimal=6)


def test_atomic_density_ne():
    # load Ne atomic wave function
    ne = AtomicDensity("ne")
    # compute density on an equally distant grid
    grid = np.arange(0.0, 10.0, 0.0001)
    dens = ne.atomic_density(grid, mode="total")
    core = ne.atomic_density(grid, mode="core")
    valn = ne.atomic_density(grid, mode="valence")
    # check shape
    assert_equal(dens.shape, grid.shape)
    assert_equal(core.shape, grid.shape)
    assert_equal(valn.shape, grid.shape)
    # check dens = core + valence
    assert_almost_equal(dens, core + valn, decimal=6)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), 10.0, decimal=6)


def test_atomic_density_c():
    # load C atomic wave function
    c = AtomicDensity("c")
    # compute density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = c.atomic_density(grid, mode="total")
    core = c.atomic_density(grid, mode="core")
    valn = c.atomic_density(grid, mode="valence")
    # check shape
    assert_equal(dens.shape, grid.shape)
    assert_equal(core.shape, grid.shape)
    assert_equal(valn.shape, grid.shape)
    # check dens = core + valence
    assert_almost_equal(dens, core + valn, decimal=6)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), 6.0, decimal=6)


def test_atomic_density_h():
    r"""Test integration of atomic density of Hydrogen."""
    h = AtomicDensity("h")
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = h.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid**2., grid), 1.0, decimal=6)


def test_atomic_density_h_anion():
    r"""Test integration of atomic density of hydrogen anion."""
    assert_raises(ValueError, AtomicDensity, "h", cation=True)  # No cations for hydrogen.
    h = AtomicDensity("h", anion=True)
    grid = np.arange(0.0, 25.0, 0.00001)
    dens = h.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid ** 2., grid), 2.0, decimal=6)


def test_atomic_density_c_anion_cation():
    r"""Test integration of atomic density of carbon anion and cation."""
    c = AtomicDensity("c", anion=True)
    grid = np.arange(0.0, 25.0, 0.00001)
    dens = c.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid ** 2., grid), 7.0, decimal=6)

    c = AtomicDensity("c", cation=True)
    grid = np.arange(0.0, 25.0, 0.00001)
    dens = c.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid ** 2., grid), 5.0, decimal=6)


def test_atomic_density_heavy_cs():
    r"""Test integration of atomic density of carbon anion and cation."""
    # These files don't exist.
    assert_raises(ValueError, AtomicDensity, "cs", cation=True)
    assert_raises(ValueError, AtomicDensity, "cs", anion=True)

    cs = AtomicDensity("cs")
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = cs.atomic_density(grid, mode="total")
    assert_almost_equal(4 * np.pi * np.trapz(dens * grid ** 2., grid), 55.0, decimal=5)


def test_atomic_density_heavy_rn():
    r"""Test integration of atomic density of carbon anion and cation."""
    # These files don't exist.
    assert_raises(ValueError, AtomicDensity, "rn", cation=True)
    assert_raises(ValueError, AtomicDensity, "rn", anion=True)

    rn = AtomicDensity("rn")
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = rn.atomic_density(grid, mode="total")
    assert_almost_equal(4 * np.pi * np.trapz(dens * grid ** 2., grid), 86, decimal=5)


def test_kinetic_energy_cation_anion_c():
    c = AtomicDensity("c", cation=True)
    grid = np.arange(0.0, 25.0, 0.0001)
    dens = c.lagrangian_kinetic_energy(grid)
    assert_almost_equal(np.trapz(dens, grid), c.energy[0], decimal=6)

    c = AtomicDensity("c", anion=True)
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = c.lagrangian_kinetic_energy(grid)
    assert_almost_equal(np.trapz(dens, grid), c.energy[0], decimal=5)


def test_derivative_electron_density_c():
    c = AtomicDensity("c", cation=True)
    eps = 1e-10
    grid = np.array([0.1, 0.1 + eps, 0.5, 0.5 + eps])
    dens = c.atomic_density(grid)
    actual = c.derivative_density(np.array([0.1, 0.5]))
    desired_0 = (dens[1] - dens[0]) / eps
    desired_1 = (dens[3] - dens[2]) / eps
    assert_almost_equal(actual, np.array([desired_0, desired_1]), decimal=4)


def test_derivative_electron_density_cr():
    cr = AtomicDensity("cr")
    eps = 1e-10
    grid = np.array([0.1 - eps, 0.1, 0.1 + eps,
                     1. - eps, 1., 1. + eps])
    dens = cr.atomic_density(grid)
    actual = cr.derivative_density(np.array([0.1, 1.]))
    desired_0 = (dens[2] - dens[0]) / (2. * eps)
    desired_1 = (dens[5] - dens[3]) / (2. * eps)
    assert_almost_equal(actual, np.array([desired_0, desired_1]), decimal=4)


def test_kinetic_energy_heavy_element_ce():
    c = AtomicDensity("ce")
    grid = np.arange(0.0, 25.0, 0.0001)
    dens = c.lagrangian_kinetic_energy(grid)
    assert_almost_equal(np.trapz(dens, grid), c.energy[0], decimal=3)


def test_raises():
    assert_raises(TypeError, AtomicDensity, 25)
    assert_raises(TypeError, AtomicDensity, "be2")
    assert_raises(ValueError, AtomicDensity.slater_orbital, np.array([[1]]), np.array([[2]]),
                  np.array([[1.]]))
    c = AtomicDensity("c")
    assert_raises(ValueError, c.atomic_density, np.array([[1.]]), "not total")


from atomdb.io.wfn_slater import load_slater_wfn


def test_parsing_slater_density_be():
    # Load the Be file
    be = load_slater_wfn("be")

    assert be['configuration'] == '1S(2)2S(2)'
    assert be['energy'] == [14.57302313]

    # Check basis of S orbitals
    assert be['orbitals'] == ['1S', '2S']
    assert np.all(abs(be['orbitals_cusp'] - np.array([1.0001235, 0.9998774])[:, None]) < 1.e-6)
    assert np.all(abs(be['orbitals_energy'] - np.array([-4.7326699, -0.3092695])[:, None]) < 1.e-6)
    assert be['orbitals_basis']['S'] == ['1S', '1S', '1S', '1S', '1S', '1S', '2S', '1S']
    assert len(be['orbitals_occupation']) == 2
    assert (be['orbitals_occupation'] == np.array([[2], [2]])).all()
    basis_numbers = np.array([[1], [1], [1], [1], [1], [1], [2], [1]])
    assert (be['basis_numbers']['S'] == basis_numbers).all()

    # Check exponents of S orbitals
    exponents = np.array([12.683501, 8.105927, 5.152556, 3.472467, 2.349757,
                          1.406429, 0.821620, 0.786473])
    assert (abs(be['orbitals_exp']['S'] - exponents.reshape(8, 1)) < 1.e-6).all()

    # Check coefficients of S orbitals
    coeff_1s = np.array([-0.0024917, 0.0314015, 0.0849694, 0.8685562, 0.0315855,
                         -0.0035284, -0.0004149, .0012299])
    assert be['orbitals_coeff']['1S'].shape == (8, 1)
    assert (abs(be['orbitals_coeff']['1S'] - coeff_1s.reshape(8, 1)) < 1.e-6).all()
    coeff_2s = np.array([0.0004442, -0.0030990, -0.0367056, 0.0138910, -0.3598016,
                         -0.2563459, 0.2434108, 1.1150995])
    assert be['orbitals_coeff']['2S'].shape == (8, 1)
    assert (abs(be['orbitals_coeff']['2S'] - coeff_2s.reshape(8, 1)) < 1.e-6).all()


def test_parsing_slater_density_ag():
    # Load the Ag file.
    ag = load_slater_wfn("ag")

    # Check configuration and energy.
    assert ag['configuration'] == 'K(2)L(8)M(18)4S(2)4P(6)5S(1)4D(10)'
    assert ag['energy'] == [5197.698468984]

    # Check orbitals
    assert ag['orbitals'] == ['1S', '2S', '3S', '4S', '5S', '2P', '3P', '4P', '3D', '4D']

    # Check basis
    assert ag['orbitals_basis']['P'] == ['2P', '3P', '3P', '2P', '3P', '3P', '3P',
                                         '3P', '2P', '2P', '2P']
    cusp = np.array([1.0002457, 1.0000318, 1.0004188, 1.0004755, 1.0007044,
                     1.0008130, 1.0008629, 0.9998751, 0.9991182, 1.0009214])[:, None]
    energy = np.array([-913.8355964, -134.8784068, -25.9178242, -4.0014988, -0.2199797,
                       -125.1815809, -21.9454343, -2.6768201, -14.6782003, -0.5374007])[:, None]
    assert (abs(ag['orbitals_cusp'] - cusp) < 1.e-6).all()
    assert (abs(ag['orbitals_energy'] - energy) < 1.e-6).all()

    # Check exponents of D orbitals
    exp_D = np.array([53.296212, 40.214567, 21.872645, 17.024065, 10.708021, 7.859216, 5.770205,
                      3.610289, 2.243262, 1.397570, 0.663294])
    assert (abs(ag['orbitals_exp']['D'] - exp_D.reshape(11, 1)) < 1.e-6).all()

    # Check coefficients of 3D orbital
    coeff_3D = np.array([0.0006646, 0.0037211, -0.0072310, 0.1799224, 0.5205360, 0.3265622,
                         0.0373867, 0.0007434, 0.0001743, -0.0000474, 0.0000083])
    assert (abs(ag['orbitals_coeff']['3D'] - coeff_3D.reshape(11, 1)) < 1.e-6).all()

    # Check coefficients of 4D orbital
    coeff_4D = np.array([-0.0002936, -0.0016839, 0.0092799, -0.0743431, -0.1179494, -0.2809146,
                         0.1653040, 0.4851980, 0.4317110, 0.1737644, 0.0013751])
    assert (abs(ag['orbitals_coeff']['4D'] - coeff_4D.reshape(11, 1)) < 1.e-6).all()

    # Check occupation numbers
    assert len(ag['orbitals_occupation']) == 10
    assert (ag['orbitals_occupation'] == np.array([2, 2, 2, 2, 1, 6, 6, 6, 10, 10]).reshape(10, 1)).all()


def test_parsing_slater_density_ne():
    # Load the Ne file
    ne = load_slater_wfn("ne")

    assert ne['configuration'] == "1S(2)2S(2)2P(6)"
    assert ne['orbitals'] == ["1S", "2S", "2P"]
    assert (ne["orbitals_occupation"] == np.array([[2], [2], [6]])).all()
    assert ne['energy'] == [128.547098140]

    # Check orbital energy and cusp
    assert (abs(ne['orbitals_energy'] - np.array([-32.7724425, -1.9303907, -0.8504095])[:, None]) < 1.e-6).all()
    assert (abs(ne['orbitals_cusp'] - np.array([1.0000603, 0.9996584, 1.0000509])[:, None]) < 1.e-6).all()

    # Check basis
    assert ne['orbitals_basis']['P'] == ['3P', '2P', '3P', '2P', '2P', '2P', '2P']
    assert ne['orbitals'] == ['1S', '2S', '2P']

    # Check exponents of P orbitals
    exp_p = np.array([25.731219, 10.674843, 8.124569, 4.295590, 2.648660, 1.710436, 1.304155])
    assert (abs(ne['orbitals_exp']['P'] - exp_p.reshape(7, 1)) < 1.e-6).all()

    # Check coefficients of P orbitals
    coeff = np.array([0.0000409, 0.0203038, 0.0340866, 0.2801866, 0.3958489, 0.3203928, 0.0510413])
    assert (abs(ne['orbitals_coeff']['2P'] - coeff.reshape(7, 1)) < 1.e-6).all()


def test_parsing_slater_density_h():
    # Load the H file
    h = load_slater_wfn("h")

    assert h['configuration'] == "1S(1)"
    assert h['energy'] == [0.5]

    # Check orbital energy and cusp
    assert (abs(h['orbitals_energy'] - np.array([-0.50])[:, None]) < 1.e-6).all()
    assert (abs(h['orbitals_cusp'] - np.array([1.])[:, None]) < 1.e-6).all()

    # Check basis
    assert h['orbitals_basis']['S'] == ['1S']
    assert h['orbitals'] == ['1S']

    # Check exponents of S orbitals
    exp_s = np.array([1.])
    assert (abs(h['orbitals_exp']['S'] - exp_s) < 1.e-6).all()

    # Check coefficients of 1S orbitals.
    coeff = np.array([1.])
    assert (abs(h['orbitals_coeff']['1S'] - coeff) < 1.e-6).all()


def test_parsing_slater_density_k():
    # Load the K file
    k = load_slater_wfn("k")

    assert k["configuration"] == "K(2)L(8)3S(2)3P(6)4S(1)"
    assert k["energy"] == [599.164786943]

    assert k['orbitals'] == ["1S", "2S", "3S", "4S", "2P", "3P"]
    assert np.all(abs(k['orbitals_cusp'] - np.array([1.0003111, 0.9994803, 1.0005849, 1.0001341,
                                                     1.0007902, 0.9998975])[:, None]) < 1.e-6)
    assert np.all(abs(k['orbitals_energy'] - np.array([-133.5330493, -14.4899575, -1.7487797, -0.1474751, -11.5192795,
                                                       -0.9544227])[:, None]) < 1.e-6)
    assert k['orbitals_basis']['P'] == ['2P', '3P', '2P', '3P', '2P', '2P', '3P', '2P', '2P', '2P']

    basis_numbers = np.array([[2], [3], [2], [3], [2], [2], [3], [2], [2], [2]])
    assert np.all(np.abs(k['basis_numbers']['P'] - basis_numbers) < 1e-5)

    # Check coefficients of 3P orbital
    coeff_3P = np.array([0.0000354, 0.0011040, -0.0153622, 0.0620133, -0.1765320, -0.3537264,
                         -0.3401560, 1.3735350, 0.1055549, 0.0010773])
    assert (abs(k['orbitals_coeff']['3P'] - coeff_3P.reshape(10, 1)) < 1.e-6).all()


def test_parsing_slater_density_i():
    i = load_slater_wfn("i")
    assert i["configuration"] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(5)"
    assert i["orbitals"] == ["1S", "2S", "3S", "4S", "5S", "2P", "3P", "4P", "5P", "3D", "4D"]
    occupation = np.array([[2], [2], [2], [2], [2], [6], [6], [6], [5], [10], [10]])
    assert (i["orbitals_occupation"] == occupation).all()

    exponents = np.array([[58.400845, 45.117174, 24.132009, 20.588554, 12.624386, 10.217388,
                             8.680013, 4.627159, 3.093797, 1.795536, 0.897975]])
    assert np.all(np.abs(i["orbitals_exp"]["D"] - exponents.reshape((11, 1))) < 1e-4)


def test_parsing_slater_density_xe():
    # Load the Xe file
    xe = load_slater_wfn("xe")
    assert xe['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)"
    assert xe['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "2P", "3P", "4P", "5P", "3D", "4D"]
    occupation = np.array([[2], [2], [2], [2], [2], [6], [6], [6], [6], [10], [10]])
    assert (xe["orbitals_occupation"] == occupation).all()
    assert xe['energy'] == [7232.138367196]

    # Check coeffs of D orbitals.
    coeffs = np.array([-0.0006386, -0.0030974, 0.0445101, -0.1106186, -0.0924762, -0.4855794,
                     0.1699923, 0.7240230, 0.3718553, 0.0251152, 0.0001040])
    assert (abs(xe['orbitals_coeff']["4D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()


def test_parsing_slater_density_xe_cation():
    # Load the Xe file
    np.testing.assert_raises(ValueError, load_slater_wfn, "xe", anion=True)
    xe = load_slater_wfn("xe", cation=True)
    assert xe['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(5)"
    assert xe['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "2P", "3P", "4P", "5P", "3D", "4D"]
    occupation = np.array([[2], [2], [2], [2], [2], [6], [6], [6], [5], [10], [10]])
    assert (xe["orbitals_occupation"] == occupation).all()
    assert xe['energy'] == [7231.708943551]

    # Check coefficients of D orbitals.
    coeff = np.array([-0.0004316, -0.0016577, -0.0041398, -0.2183952, 0.0051908, -0.2953384,
                     -0.0095762, 0.6460145, 0.4573096, 0.0431928, -0.0000161])
    assert (abs(xe['orbitals_coeff']["4D"] - coeff.reshape((len(coeff), 1))) < 1e-10).all()


def test_parsing_slater_density_h_anion():
    # Load the Hydrogen file
    # There is no cation of hydrogen in data folder.
    np.testing.assert_raises(ValueError, load_slater_wfn, "h", cation=True)
    h = load_slater_wfn("h", anion=True)
    assert h['configuration'] == "1S(2)"
    assert h['orbitals'] == ["1S"]
    occupation = np.array([[2]])
    assert (h["orbitals_occupation"] == occupation).all()
    assert h['energy'] == [0.487929734]

    # Check coeffs of 1S orbitals.
    coeff = np.array([0.0005803, 0.0754088, 0.2438040, 0.3476471, 0.3357298, 0.0741188])
    assert (abs(h['orbitals_coeff']["1S"] - coeff.reshape((len(coeff), 1))) < 1e-10).all()

    # Check exps of 1S orbitals.
    exps = np.array([3.461036, 1.704290, 1.047762, 0.626983, 0.392736, 0.304047])
    assert (abs(h['orbitals_exp']["S"] - exps.reshape((len(exps), 1))) < 1e-10).all()


def test_parsing_slater_heavy_atom_cs():
    np.testing.assert_raises(ValueError, load_slater_wfn, "cs", cation=True)
    np.testing.assert_raises(ValueError, load_slater_wfn, "cs", anion=True)
    # Load the Xe file
    cs = load_slater_wfn("cs")
    assert cs['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)6S(1)"
    assert cs['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S", "2P", "3P", "4P", "5P", "3D",
                              "4D"]
    occupation = np.array([[2], [2], [2], [2], [2], [1], [6], [6], [6], [6], [10], [10]])
    assert (cs["orbitals_occupation"] == occupation).all()
    assert cs['energy'] == [7553.933539793]

    # Check coeffs of D orbitals.
    coeffs = np.array([-0.0025615, -0.1930536, -0.2057559, -0.1179374, 0.4341816, 0.6417053,
                       0.1309576])
    assert (abs(cs['orbitals_coeff']["4D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()

    # Check Exponents of D orbitals.
    exps = np.array([32.852137, 18.354403, 14.523221, 10.312620, 7.919345, 5.157647,
                     3.330606])
    assert (abs(cs['orbitals_exp']["D"] - exps.reshape((len(exps), 1))) < 1e-10).all()


def test_parsing_slater_heavy_atom_rn():
    # Load the Xe file
    rn = load_slater_wfn("rn")
    assert rn['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)4F(14)6S(2)5D(10)6P(6)"
    print(rn["orbitals"])
    assert rn['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S", "2P", "3P", "4P", "5P", "6P",
                              "3D", "4D", "5D", "4F"]
    occupation = np.array([[2], [2], [2], [2], [2], [2], [6], [6], [6], [6], [6], [10], [10],
                           [10], [14]])
    assert (rn["orbitals_occupation"] == occupation).all()
    assert rn['energy'] == [21866.772036482]

    # Check coeffs of 4F orbitals.
    coeffs = np.array([.0196357, .2541992, .4806186, -.2542278, .5847619, .0099519])
    assert (abs(rn['orbitals_coeff']["4F"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()


def test_parsing_slater_heavy_atom_lr():
    # Load the lr file
    lr = load_slater_wfn("lr")
    assert lr['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)4F(14)6S(2)5D(10)6P(6)" \
                                  "7S(2)6D(1)5F(14)"
    assert lr['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S", "7S",
                              "2P", "3P", "4P", "5P", "6P",
                              "3D", "4D", "5D", "6D", "4F", "5F"]
    occupation = np.array([[2], [2], [2], [2], [2], [2], [2], [6], [6], [6], [6], [6], [10], [10],
                           [10], [1], [14], [14]])
    assert (lr["orbitals_occupation"] == occupation).all()
    assert lr['energy'] == [33557.949960623]

    # Check coeffs of 3D orbitals.
    coeffs = np.array([0.0017317, 0.4240510, 0.4821228, 0.1753365, -0.0207393, 0.0091584,
                       -0.0020913, 0.0005398, -0.0001604, 0.0000443, -0.0000124])
    assert (abs(lr['orbitals_coeff']["3D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()

    # Check coeffs of 2S orbitals.
    coeffs = np.array([0.0386739, -0.4000069, -0.2570804, 1.2022357, 0.0787866, 0.0002957,
                       0.0002277, 0.0002397, -0.0005650, 0.0000087, 0.0000031, -0.0000024,
                       0.0000008, -0.0000002, 0.0000001])
    assert (abs(lr['orbitals_coeff']["2S"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()


def test_parsing_slater_heavy_atom_dy():
    # Load the dy file
    dy = load_slater_wfn("dy")
    assert dy['configuration'] == "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)6S(2)5D(0)4F(10)"
    assert dy['orbitals'] == ["1S", "2S", "3S", "4S", "5S", "6S",
                              "2P", "3P", "4P", "5P",
                              "3D", "4D", "4F"]
    occupation = np.array([[2], [2], [2], [2], [2], [2], [6], [6], [6], [6], [10], [10], [10]])
    assert (dy["orbitals_occupation"] == occupation).all()
    assert dy['energy'] == [11641.452478555]

    # Check coeffs of 4D orbitals.
    coeffs = np.array([-0.0016462, -0.2087639, -0.2407385, -0.1008913, 0.4844709, 0.6180159,
                       0.1070867])
    assert (abs(dy['orbitals_coeff']["4D"] - coeffs.reshape((len(coeffs), 1))) < 1e-10).all()
