# This file is part of AtomDB.
#
# AtomDB is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# AtomDB is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with AtomDB. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os
import re
import atomdb

from atomdb.periodic import Element
from grid.onedgrid import UniformInteger
from grid.rtransform import ExpRTransform

# from importlib_resources import files
from atomdb.utils import DEFAULT_DATAPATH
from scipy.special import factorial


__all__ = ["AtomicDensity", "load_slater_wfn", "run"]


BOUND = (1e-5, 15.0)

NPOINTS = 10000

# DATAPATH = files("atomdb.datasets.slater.raw")
# DATAPATH = os.path.abspath(DATAPATH._paths[0])


class AtomicDensity:
    r"""
    Atomic Density Class.

    Reads and Parses information from the .slater file of a atom and stores it inside this class.
    It is then able to construct the (total, core and valence) electron density based
    on linear combination of orbitals where each orbital is a linear combination of
    Slater-type orbitals.
    Elements supported by default from "./atomdb/data/examples/" range from Hydrogen to Xenon.

    Attributes
    ----------
    Attributes relating to the standard electron configuration.

    energy : list
        Energy of that atom.
    configuration : str
        Return the electron configuration of the element.
    orbitals : list, (M,)
        List of strings representing each of the orbitals in the electron configuration.
        For example, Beryllium has ["1S", "2S"] in its electron configuration.
        Ordered based on orbital energy.
    orbitals_occupation : ndarray, (M, 1)
        Returns the number of electrons in each of the orbitals in the electron configuration.
        e.g. Beryllium has two electrons in "1S" and two electrons in "2S".

    Attributes relating to representing orbitals as linear combination of Slater-type orbitals.

    orbital_energy : list, (N, 1)
        Energy of each of the N Slater-type orbital.
    orbitals_cusp : list, (N, 1)
        Cusp of each of the N Slater-type orbital. Same ordering as `orbitals`. Does not exist
        for Heavy atoms past Xenon.
    orbitals_basis : dict
        Keys are the orbitals in the electron configuration. Each orbital has N Slater-type orbital
        attached to them.
    orbitals_exp : dict (str : ndarray(N, 1))
        Key is the orbital in the electron configuration and the item of that key is the Slater
         exponents attached
         to each N Slater-type orbital.
    orbitals_coeff : dict (str : ndarray(N, 1))
        Key is the orbital in the electron configuration (e. 1S, 2S or 2P) and the item is the
        Slater coefficients attached to each of the N Slater-type orbital.
    basis_numbers : dict (str : ndarray(N, 1))
        Key is orbital in electron configuration and the item is the basis number of each of
        the N Slater-type orbital. These are the principal quantum number to each Slater-type
        orbital.

    Methods
    -------
    atomic_density(mode="total") :
        Construct the atomic density from the linear combinations of slater-type orbitals.
        Can compute the total (default), core and valence atomic density.
    lagrangian_kinetic_energy :
        Construct the Positive Definite Kinetic energy.

    Examples
    --------
    Grab information about Beryllium.
    >> be =  AtomicDensity("be")

    Some of the attributes are the following.
    >> print(be.configuration) #  Should "1S(2)2S(2)".
    >> print(be.orbitals)  # ['1S', '2S'].
    >> print(be.orbitals_occupation) # [[2], [2]] Two electrons in each orbital.
    >> print(be.orbitals_cusp)  # [1.0001235, 0.9998774].

    The Slatar coefficients and exponents of the 1S orbital can be obtained as:
    >> print(be.orbital_coeff["1S"])
    >> print(be.orbitals_exp["1S"])

    The total, core and valence electron density can be obtained as:
    >> points = np.arange(0., 25., 0.01)
    >> total_density = be.atomic_density(points, "total")
    >> core_density = be.atomic_density(points, "core")
    >> valence_density = be.atomic_density(points, "valence")

    References
    ----------
    [1] "Analytical Hartree–Fock wave functions subject to cusp and asymptotic constraints:
        He to Xe, Li+ to Cs+, H− to I−" by T. Koga, K. Kanayama, S. Watanabe and A.J. Thakkar.

    """

    def __init__(self, element, anion=False, cation=False, data_path=None):
        r"""
        Construct AtomicDensity object.

        Parameters
        ----------
        element : str
            Symbol of element.
        anion : bool
            If true, then the anion of element is used.
        cation : bool
            If true, then the cation of element is used.
        data_path : str or Path or None
            The path to the data folder if not the default.

        """
        if not isinstance(element, str) or not element.isalpha():
            raise TypeError("The element argument should be all letters string.")

        data = load_slater_wfn(element, anion, cation, data_path)
        for key, value in data.items():
            setattr(self, key, value)

    @staticmethod
    def slater_orbital(exponent, number, points):
        r"""
        Compute the Slater-type orbitals on the given points.

        A Slater-type orbital is defined as:
        .. math::
            R(r) = N r^{n-1} e^{- C r)

        where,
            :math:`n` is the principal quantum number of that orbital.
            :math:`N` is the normalizing constant.
            :math:`r` is the radial point, distance to the origin.
            :math:`C` is the zeta exponent of that orbital.

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents of Slater orbitals.
        number : ndarray, (M, 1)
            The principle quantum numbers of Slater orbitals.
        points : ndarray, (N,)
            The radial grid points.

        Returns
        -------
        slater : ndarray, (N, M)
            The Slater-type orbitals evaluated on the grid points.

        See Also
        --------
        The principal quantum number of all of the orbital are stored in `basis_numbers`.
        The zeta exponents of all of the orbitals are stored in the attribute `orbitals_exp`.

        """
        if points.ndim != 1:
            raise ValueError("The argument point should be a 1D array.")
        # compute norm & pre-factor
        norm = np.power(2.0 * exponent, number) * np.sqrt(
            (2.0 * exponent) / factorial(2.0 * number)
        )
        pref = np.power(points, number - 1).T
        # compute slater function
        slater = norm.T * pref * np.exp(-exponent * points).T
        return slater

    def phi_matrix(self, points, deriv=0):
        r"""
        Compute the linear combination of Slater-type atomic orbitals on the given points.

        Each row corresponds to a point on the grid, represented as :math:`r` and
        each column is represented as a linear combination of Slater-type atomic orbitals
        of the form:

        .. math::
            \sum c_i R(r, n_i, C_i)

        where,
            :math:`c_i` is the coefficient of the Slater-type orbital.
            :math:`C_i` is the zeta exponent attached to the Slater-type orbital.
            :math:`n_i` is the principal quantum number attached to the Slater-type orbital.
            :math:`R(r, n_i, C_i)` is the Slater-type orbital.
            i ranges from 0 to K-1 where K is the number of orbitals in electron configuration.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.
        deriv : int
            Order of the derivative. Default is 0 and can also be 1 or 2.

        Returns
        -------
        phi_matrix : ndarray(N, K)
            The linear combination of Slater-type orbitals evaluated on the grid points, where K is
            the number of orbitals. The order is S orbitals, then P then D.

        Notes
        -----
        - At r = 0, the derivative of slater-orbital is undefined and this function returns
            zero instead. See "derivative_slater_type_orbital".

        """

        # compute orbital composed of a linear combination of Slater
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            # get exponent and number of the orbital type (s, p, d, f)
            exps, numbers = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            if deriv == 0:
                slater = self.slater_orbital(exps, numbers, points)
            elif deriv == 1:
                slater = self.derivative_slater_type_orbital(exps, numbers, points)
            elif deriv == 2:
                slater = self.second_derivative_slater_type_orbital(exps, numbers, points)
            else:
                raise ValueError("Derivative order can only be 0, 1 or 2.")
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def eval_density(self, points, mode="total"):
        r"""
        Compute atomic density on the given points.

        The total density is written as a linear combination of Slater-type orbital
        whose coefficients is the orbital occupation number of the electron configuration:
        .. math::
            \sum n_i |P(r, n_i, C_i)|^2

        where,
            :math:`n_i` is the number of electrons in orbital i.
            :math:`P(r, n_i, C_i)` is a linear combination of Slater-type orbitals evaluated
                on the point :math:`r`.

        For core and valence density, please see More Info below.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.
        mode : str
            The type of atomic density, which can be "total", "valence" or "core".

        Returns
        -------
        dens : ndarray, (N,)
            The atomic density on the grid points.

        Notes
        -----
        The core density and valence density is respectively written as:
        .. math::
            \sum n_i (1 - e^{-|e_i - e_{homo}|^2}) |P(r, n_i, C_i)|
            \sum n_i e^{-|e_i - e_{homo}|^2}) |P(r, n_i. C_i)|

        where,
            :math:`e_i` is the energy of the orbital i.
            :math:`e_{homo}` is the energy of the highest occupying orbital.

        """
        if mode not in ["total", "valence", "core"]:
            raise ValueError("Argument mode not recognized!")

        # compute orbital occupation numbers
        orb_occs = self.orbitals_occupation
        if mode == "valence":
            # get index of homo (equal to sum of occupied alpha orbitals)
            orb_homo = sum(orb_occs[: len(orb_occs) // 2 - 1])
            e_orb_homo = self.orbitals_energy[orb_homo]
            orb_occs = orb_occs * np.exp(-((self.orbitals_energy - e_orb_homo) ** 2))
        elif mode == "core":
            orb_homo = self.orbitals_energy[len(self.orbitals_occupation) - 1]
            orb_occs = orb_occs * (1.0 - np.exp(-((self.orbitals_energy - orb_homo) ** 2)))
        # compute density
        dens = np.dot(self.phi_matrix(points) ** 2, orb_occs).ravel() / (4 * np.pi)
        return dens

    def eval_orbs_density(self, points):
        r"""Return each orbital density evaluated at a set of points

        rho_i(r) = n_i |P(r, n_i, C_i)|^2

        where,
        :math:`n_i` is the number of electrons in orbital i.
        :math:`P(r, n_i, C_i)` is a linear combination of Slater-type orbitals evaluated
            on the point :math:`r`.

        Parameters
        ----------
        points: np.ndarray(N)
            radial grid points

        Returns
        -------
        orb_dens : np.ndarray(K_orb, N)
            orbitals density at a set of grid points (N)
        """
        orb_occs = self.orbitals_occupation
        orb_dens = self.phi_matrix(points) ** 2 * orb_occs.ravel() / (4 * np.pi)
        return orb_dens.T

    def eval_orbs_radial_d_density(self, points):
        r"""Return each orbital density evaluated at a set of points

        math::
        \frac{d \rho_i(r)}{dr} = n_i \frac{d}{dr} |P(r, n_i, C_i)|^2

        where,
        :math:`n_i` is the number of electrons in orbital i.
        :math:`P(r, n_i, C_i)` is a linear combination of Slater-type orbitals evaluated
            on the point :math:`r`.

        Parameters
        ----------
        points: np.ndarray(N)
            radial grid points

        Returns
        -------
        orb_dens : np.ndarray(K_orb, N)
            orbitals density at a set of grid points (N)
        """

        factor = self.phi_matrix(points) * self.phi_matrix(points, deriv=1)
        orb_derivative = 2.0 * factor * self.orbitals_occupation.ravel() / (4 * np.pi)
        return orb_derivative.T

    def eval_orbs_radial_dd_density(self, points):
        r"""Return each orbital density evaluated at a set of points

        math::
        \frac{d^{2} \rho_i(r)}{dr^{2}} = n_i \frac{d^{2}}{dr^{2}} |P(r, n_i, C_i)|^2

        where,
        :math:`n_i` is the number of electrons in orbital i.
        :math:`P(r, n_i, C_i)` is a linear combination of Slater-type orbitals evaluated
            on the point :math:`r`.

        Parameters
        ----------
        points: np.ndarray(N)
            radial grid points

        Returns
        -------
        orb_dens : np.ndarray(K_orb, N)
            orbitals density at a set of grid points (N)
        """

        factor = (
            self.phi_matrix(points) * self.phi_matrix(points, deriv=2)
            + self.phi_matrix(points, deriv=1) ** 2
        )
        orb_derivative = 2.0 * factor * self.orbitals_occupation.ravel() / (4 * np.pi)
        return orb_derivative.T

    @staticmethod
    def derivative_slater_type_orbital(exponent, number, points):
        r"""
        Compute the derivative of Slater-type orbitals on the given points.

        A Slater-type orbital is defined as:
        .. math::
            \frac{d R(r)}{dr} = \bigg(\frac{n-1}{r} - C \bigg) N r^{n-1} e^{- C r),

        where,
            :math:`n` is the principal quantum number of that orbital.
            :math:`N` is the normalizing constant.
            :math:`r` is the radial point, distance to the origin.
            :math:`C` is the zeta exponent of that orbital.

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents of Slater orbitals.
        number : ndarray, (M, 1)
            The principle quantum numbers of Slater orbitals.
        points : ndarray, (N,)
            The radial grid points. If points contain zero, then it is undefined at those
            points and set to zero.

        Returns
        -------
        slater : ndarray, (N, M)
            The derivative of Slater-type orbitals evaluated on the grid points.

        Notes
        -----
        - At r = 0, the derivative is undefined and this function returns zero instead.

        References
        ----------
        See wikipedia page on "Slater-Type orbitals".

        """
        slater = AtomicDensity.slater_orbital(exponent, number, points)
        # derivative
        deriv_pref = (number.T - 1.0) / np.reshape(points, (points.shape[0], 1)) - exponent.T
        deriv = deriv_pref * slater
        return deriv

    @staticmethod
    def second_derivative_slater_type_orbital(exponent, number, points):
        r"""
        Compute the derivative of Slater-type orbitals on the given points.

        A Slater-type orbital is defined as:
        .. math::
            \frac{d R(r)}{dr} = \bigg(\frac{n-1}{r} - C \bigg) N r^{n-1} e^{- C r),

        where,
            :math:`n` is the principal quantum number of that orbital.
            :math:`N` is the normalizing constant.
            :math:`r` is the radial point, distance to the origin.
            :math:`C` is the zeta exponent of that orbital.

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents of Slater orbitals.
        number : ndarray, (M, 1)
            The principle quantum numbers of Slater orbitals.
        points : ndarray, (N,)
            The radial grid points. If points contain zero, then it is undefined at those
            points and set to zero.

        Returns
        -------
        slater : ndarray, (N, M)
            The second derivative of Slater-type orbitals evaluated on the grid points.

        Notes
        -----
        - At r = 0, the derivative is undefined and this function returns zero instead.

        References
        ----------
        See wikipedia page on "Slater-Type orbitals".

        """
        slater = AtomicDensity.slater_orbital(exponent, number, points)
        # derivative
        deriv_pref = ((number.T - 1.0) / np.reshape(points, (points.shape[0], 1)) - exponent.T) ** 2
        deriv_pref -= (number.T - 1.0) / np.reshape(points, (points.shape[0], 1)) ** 2
        deriv2 = deriv_pref * slater
        return deriv2

    def eval_orbs_ked_positive_definite(self, points):
        r"""Return each the kinetic energy density of each orbital evaluated at a set of points

        math::
            \tau_{\text{PD}}^{i} \left(\mathbf{r}\right) = \tfrac{1}{2} n_i \rvert \nabla \phi_i \left(\mathbf{r}\right) \lvert^2


        Parameters
        ----------
        points: np.ndarray(N)
            radial grid points

        Returns
        -------
        orb_ked : np.ndarray(K_orb, N)
            orbitals kinetic energy density values at a set of grid points (N).
        """
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            slater = AtomicDensity.slater_orbital(exps, number, points)
            # derivative
            deriv_pref = (number.T - 1.0) - exps.T * np.reshape(points, (points.shape[0], 1))
            deriv = deriv_pref * slater
            phi_matrix[:, index] = np.dot(deriv, self.orbitals_coeff[orbital]).ravel()

        angular = []  # Angular numbers are l(l + 1)
        for index, orbital in enumerate(self.orbitals):
            if "S" in orbital:
                angular.append(0.0)
            elif "P" in orbital:
                angular.append(2.0)
            elif "D" in orbital:
                angular.append(6.0)
            elif "F" in orbital:
                angular.append(12.0)

        orb_occs = self.orbitals_occupation
        orbs_ked = phi_matrix**2.0 * orb_occs.ravel() / 2.0
        # Add other term
        molecular = self.phi_matrix(points) ** 2.0 * np.array(angular)
        orbs_ked += molecular * orb_occs.ravel() / 2.0
        return orbs_ked.T

    def eval_ked_positive_definite(self, points):
        r"""
        Positive definite or Lagrangian kinetic energy density.

        Parameters
        ----------
        points : ndarray,(N,)
            The radial grid points.

        Returns
        -------
        energy : ndarray, (N,)
            The kinetic energy on the grid points.

        Notes
        -----
        - To integrate this to get kinetic energy in .slater files, it should not be done in
            spherically, ie the integral is exactly :math:`\int_0^{\infty} T(r) dr`.

        """

        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            slater = AtomicDensity.slater_orbital(exps, number, points)
            # derivative
            deriv_pref = (number.T - 1.0) - exps.T * np.reshape(points, (points.shape[0], 1))
            deriv = deriv_pref * slater
            phi_matrix[:, index] = np.dot(deriv, self.orbitals_coeff[orbital]).ravel()

        angular = []  # Angular numbers are l(l + 1)
        for index, orbital in enumerate(self.orbitals):
            if "S" in orbital:
                angular.append(0.0)
            elif "P" in orbital:
                angular.append(2.0)
            elif "D" in orbital:
                angular.append(6.0)
            elif "F" in orbital:
                angular.append(12.0)

        orb_occs = self.orbitals_occupation
        energy = np.dot(phi_matrix**2.0, orb_occs).ravel() / 2.0
        # Add other term
        molecular = self.phi_matrix(points) ** 2.0 * np.array(angular)
        energy += np.dot(molecular, orb_occs).ravel() / 2.0
        return energy

    def eval_radial_d_density(self, points):
        r"""
        Return the derivative of the atomic density on a set of points.

        Parameters
        ----------
        points : ndarray,(N,)
            The radial grid points.

        Returns
        -------
        deriv : ndarray, (N,)
            The derivative of atomic density on the grid points.
        """
        factor = self.phi_matrix(points) * self.phi_matrix(points, deriv=1)
        derivative = np.dot(2.0 * factor, self.orbitals_occupation).ravel() / (4 * np.pi)
        return derivative

    def eval_radial_dd_density(self, points):
        r"""
        Return the second derivative of the atomic density on a set of points.

        Parameters
        ----------
        points : ndarray,(N,)
            The radial grid points.

        Returns
        -------
        deriv : ndarray, (N,)
            The derivative of atomic density on the grid points.
        """
        factor = (
            self.phi_matrix(points) * self.phi_matrix(points, deriv=2)
            + self.phi_matrix(points, deriv=1) ** 2
        )
        dderivative = np.dot(2.0 * factor, self.orbitals_occupation).ravel() / (4 * np.pi)
        return dderivative


def get_cs_occupations(configuration):
    """
    Get the alpha and beta occupation numbers for each contracted shell.

    Parameters
    ----------
    configuration : str
        The electron configuration.

    Returns
    --------
    a_occ, b_occ, max_occ : dict, dict, dict
        Dictionaries containing the alpha, beta, and maximum occupation numbers for each
        contracted shell. Keys are the contracted shells (e.g. 1S, 2S, 2P, 3D) and values are
        the occupation numbers.

    """
    # alpha and beta occupation numbers for each contracted shell
    a_occ = {}
    b_occ = {}

    # fill in the occupation numbers for the K, L, M, N shells if they are present
    for cs in ["K", "L", "M", "N"]:
        if cs in configuration:
            if cs == "K":
                a_occ["1S"], b_occ["1S"] = 1, 1
            elif cs == "L":
                a_occ["2S"], b_occ["2S"] = 1, 1
                a_occ["2P"], b_occ["2P"] = 3, 3
            elif cs == "M":
                a_occ["3S"], b_occ["3S"] = 1, 1
                a_occ["3P"], b_occ["3P"] = 3, 3
                a_occ["3D"], b_occ["3D"] = 5, 5
            elif cs == "N":
                a_occ["4S"], b_occ["4S"] = 1, 1
                a_occ["4P"], b_occ["4P"] = 3, 3
                a_occ["4D"], b_occ["4D"] = 5, 5
                a_occ["4F"], b_occ["4F"] = 7, 7

    # create possible contracted shells and their max occupation numbers for alpha or beta
    contractions = [
        *[str(x) + "S" for x in range(1, 8)],
        *[str(x) + "P" for x in range(2, 8)],
        *[str(x) + "D" for x in range(3, 8)],
        *[str(x) + "F" for x in range(4, 8)],
    ]

    # maximum occupation numbers for each contracted shell (alpha or beta only)
    max_occ = {"S": 1, "P": 3, "D": 5, "F": 7}

    # for each possible contracted shell
    for cs in contractions:
        # check if the contracted shell is the configuration
        if cs in configuration:
            # get the total number of electrons in the contracted shell
            n_elec = re.search(cs + r"\((.*?)\)", configuration).group(1)
            n_elec = int(n_elec)

            # fill alpha and beta occupations for the contracted shell following Hund's rule
            if n_elec > max_occ[cs[-1]]:
                a_occ[cs], b_occ[cs] = max_occ[cs[-1]], n_elec - max_occ[cs[-1]]
            else:
                a_occ[cs], b_occ[cs] = n_elec, 0
    return a_occ, b_occ, max_occ


def load_slater_wfn(element, anion=False, cation=False, data_path=None):
    """
    Return the data recorded in the atomic Slater wave-function file as a dictionary.

    Parameters
    ----------
    element : str
        The element symbol.
    anion : bool
        If true, then the anion of element is used.
    cation : bool
        If true, then the cation of element is used.
    data_path : str or Path or None
        The path to the data folder.

    """
    # set the data path
    if data_path is None:
        data_path = DEFAULT_DATAPATH
    data_path = os.path.join(data_path, "slater", "raw")

    # Heavy atoms from atom cs to lr.
    heavy_atoms = [
        "cs",
        "ba",
        "la",
        "ce",
        "pr",
        "nd",
        "pm",
        "sm",
        "eu",
        "gd",
        "tb",
        "dy",
        "ho",
        "er",
        "tm",
        "yb",
        "lu",
        "hf",
        "ta",
        "w",
        "re",
        "os",
        "ir",
        "pt",
        "au",
        "hg",
        "tl",
        "pb",
        "bi",
        "po",
        "at",
        "rn",
        "fr",
        "ra",
        "ac",
        "th",
        "pa",
        "u",
        "np",
        "pu",
        "am",
        "cm",
        "bk",
        "cf",
        "es",
        "fm",
        "md",
        "no",
        "lr",
    ]

    anion_atoms = [
        "ag",
        "al",
        "as",
        "b",
        "br",
        "c",
        "cl",
        "co",
        "cr",
        "cu",
        "f",
        "fe",
        "ga",
        "ge",
        "h",
        "i",
        "in",
        "k",
        "li",
        "mn",
        "mo",
        "n",
        "na",
        "nb",
        "ni",
        "o",
        "p",
        "pd",
        "rb",
        "rh",
        "ru",
        "s",
        "sb",
        "sc",
        "se",
        "si",
        "sn",
        "tc",
        "te",
        "ti",
        "v",
        "y",
        "zr",
    ]

    cation_atoms = [
        "ag",
        "al",
        "ar",
        "as",
        "b",
        "be",
        "br",
        "c",
        "ca",
        "cd",
        "cl",
        "co",
        "cr",
        "cs",
        "cu",
        "f",
        "fe",
        "ga",
        "ge",
        "i",
        "in",
        "k",
        "kr",
        "li",
        "mg",
        "mn",
        "mo",
        "n",
        "na",
        "nb",
        "ne",
        "ni",
        "o",
        "p",
        "pd",
        "rb",
        "rh",
        "ru",
        "s",
        "sb",
        "sc",
        "se",
        "si",
        "sn",
        "sr",
        "tc",
        "te",
        "ti",
        "v",
        "xe",
        "y",
        "zn",
        "zr",
    ]

    # select and open file containing the slater data
    is_heavy_element = element.lower() in heavy_atoms
    if (anion or cation) and is_heavy_element:
        raise ValueError("Both Anion & Cation Slater File for element %s does not exist." % element)
    if anion:
        if element.lower() in anion_atoms:
            # file_path = "./atomdb/data/anion/%s.an" % element.lower()
            file_path = "anion/%s.an" % element.lower()
        else:
            raise ValueError("Anion Slater File for element %s does not exist." % element)
    elif cation:
        if element.lower() in cation_atoms:
            # file_path = "./atomdb/data/cation/%s.cat" % element.lower()
            file_path = "cation/%s.cat" % element.lower()
        else:
            raise ValueError("Cation Slater File for element %s does not exist." % element)
    else:
        # file_path = "./atomdb/data/neutral/%s.slater" % element.lower()
        file_path = "neutral/%s.slater" % element.lower()

    file_name = os.path.join(data_path, file_path)

    def get_column(t_orbital):
        """
        Correct the error in order to retrieve the correct column.

        The Columns are harder to parse as the orbitals start with one while p orbitals start at
            energy.

        Parameters
        ----------
        t_orbital : str
            orbital i.e. "1S" or "2P" or "3D"

        Returns
        -------
        int :
            Retrieve the right column index depending on whether it is "S", "P" or "D" orbital.

        """
        if t_orbital[1] == "S":
            return int(t_orbital[0]) + 1
        elif t_orbital[1] == "P":
            return int(t_orbital[0])
        elif t_orbital[1] == "D":
            return int(t_orbital[0]) - 1
        elif t_orbital[1] == "F":
            return int(t_orbital[0]) - 2
        else:
            raise ValueError("Did not recognize orbital %s " % t_orbital)

    def configuration_exact_for_heavy_elements(configuration):
        r"""later file for heavy elements does not contain the configuration in right format."""
        true_configuration = ""
        if "[XE]" in configuration:
            true_configuration += "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)"
            true_configuration += configuration.split("[XE]")[1]
        elif "[RN]" in configuration:
            # Add Xenon
            true_configuration += "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)"
            # Add Rn
            true_configuration += "4F(14)6S(2)5D(10)6P(6)"
            # Add rest
            true_configuration += configuration.split("[RN]")[1]
        else:
            raise ValueError("Heavy element is not the right format for parsing. ")
        return true_configuration

    with open(file_name, "r") as f:
        next_line = f.readline()
        configuration = next_line.split()[1].replace(",", "")
        if is_heavy_element:
            configuration = configuration_exact_for_heavy_elements(configuration)

        # in light atoms, the energy is the next line with data
        next_line = f.readline().strip()
        # some cases have empty lines before the energy (remove them)
        while not next_line:
            next_line = f.readline().strip()

        # heavy atoms have 6 lines of redundant data before the energy (skip them)
        if is_heavy_element:
            for _ in range(0, 6):
                next_line = f.readline().strip()
        # read total energy
        energy = [float(next_line.split("=")[1])]

        # declare containers for orbitals, orbital basis, cusp, energy, exponents, and coefficients
        cs = []
        cs_basis = {}
        cs_cusp = []
        cs_energy = []
        cs_exp = {}
        cs_coeff = {}

        # ignore next line, has no useful information T, V and V/T
        f.readline()
        # some files have empty lines before the first sub-shell
        while not f.readline().strip():
            pass

        while next_line:
            # If line has ___S___ or P or D where _ = " ". This is the start of a new sub-shell
            if re.search(r"  [S|P|D|F]  ", next_line):
                # read sub-shell (e.g S) and contractions (e.g 1S, 2S, 3S) for that sub-shell
                subshell = next_line.split()[0]
                list_of_orbitals = next_line.split()[1:]
                # add sub-shell contractions to the list of contractions
                cs += list_of_orbitals

                # read contractions energies from next line and store them following the
                # contractions list order
                next_line = f.readline().strip()
                cs_energy.extend([float(x) for x in next_line.split()[1:]])

                # save cusp values, for heavy atoms this is not present
                if not is_heavy_element:
                    next_line = f.readline()
                    cs_cusp.extend([float(x) for x in next_line.split()[1:]])
                # read lines until next sub-shell and get the exponents and coefficients

                for next_line in f:
                    # break if next line is not an integer followed by a sub-shell (e.g 1S, 2S)
                    if not re.match(r"\A^\d" + subshell, next_line.lstrip()):
                        break

                    list_words = next_line.split()
                    # read orbital basis from first column, save basis list for each sub-shell
                    cs_basis.setdefault(subshell, []).append(list_words[0])
                    # read basis exponents from second column and save as basis list (same order)
                    cs_exp.setdefault(subshell, []).append(float(list_words[1]))

                    # read orbital basis coefficients from the correct column.
                    # for each orbital, the coefficients are saved in same order as basis list
                    for x in list_of_orbitals:
                        coeff = float(list_words[get_column(x)])
                        cs_coeff.setdefault(x, []).append(coeff)
            else:
                next_line = f.readline()

    # create dictionaries with energy and cusps for each contracted shell
    cs_energy_dict = {i: j for i, j in zip(cs, cs_energy)}
    cs_cusp_dict = {i: j for i, j in zip(cs, cs_cusp)}

    # compute alpha, beta and max occupation numbers for each contracted shell
    a_occ_dict, b_occ_dict, max_occ_dict = get_cs_occupations(configuration)

    # list of orbital occupations
    a_occ = []
    b_occ = []
    # list of orbitals energies and cusps
    orbitals = []
    orb_energy = []
    orb_cusp = []

    # for each contracted shell, in order of increasing energy
    for _, contraction in sorted(zip(cs_energy, cs)):
        # for each orbital in the contracted shell
        for i in range(max_occ_dict[contraction[-1]]):
            # append the orbital to the list
            orbitals.append(contraction)
            # compute alpha and beta occupation numbers for the orbital
            a_occ_val = 1 if i < a_occ_dict[contraction] else 0
            b_occ_val = 1 if i < b_occ_dict[contraction] else 0
            # add orbital occupation numbers to the list
            a_occ.append(a_occ_val)
            b_occ.append(b_occ_val)
            # get orbital energy, exponent, and coefficient
            orb_energy.append(cs_energy_dict[contraction])
            # get orbital cusp
            if cs_cusp_dict:
                orb_cusp.append(cs_cusp_dict[contraction])

    # construct the total orbital list, occupation list, energy list, cusp list
    orbitals += orbitals
    orb_occ = np.array(a_occ + b_occ)
    orb_energy = np.array(orb_energy + orb_energy)
    orb_cusp = np.array(orb_cusp + orb_cusp)

    data = {
        "configuration": configuration,
        "energy": energy,
        "orbitals": orbitals,
        "orbitals_energy": orb_energy[:, None],
        "orbitals_cusp": orb_cusp[:, None],
        "orbitals_basis": cs_basis,
        "orbitals_exp": {
            key: np.asarray(value).reshape(len(value), 1)
            for key, value in cs_exp.items()
            if value != []
        },
        "orbitals_coeff": {
            key: np.asarray(value).reshape(len(value), 1)
            for key, value in cs_coeff.items()
            if value != []
        },
        "orbitals_occupation": orb_occ[:, None],
        "basis_numbers": {
            key: np.asarray([[int(x[0])] for x in value])
            for key, value in cs_basis.items()
            if len(value) != 0
        },
    }
    return data


DOCSTRING = """Slater Dataset

The following neutral and ionic (+/- 1 charge) species are available:

`neutrals` He to Xe, Cs to Lr
`cations` Li to Cs
`anions` H to I

The Slater basis set information for both anion, cation and neutral, was obtained from the paper:
`(1999), Int. J. Quantum Chem., 71: 491-497 <https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-461X(1999)71:6%3C491::AID-QUA6%3E3.0.CO;2-T>`_

The neutral heavy elements were obtained from:
`Theor Chem Acc 104, 411–413 (2000) <https://link.springer.com/article/10.1007/s002140000150>`_
"""


def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Compile the densities from Slater orbitals database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")
    if charge != 0 and abs(charge) > 1:
        raise ValueError(f"`charge` must be one of -1, 0 or 1")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    atnum = atomdb.element_number(elem)
    nelec = atnum - charge
    nspin = mult - 1

    # Retrieve Slater data
    if charge == 0:
        species = AtomicDensity(elem, anion=False, cation=False, data_path=datapath)
    elif charge > 0:
        species = AtomicDensity(elem, anion=False, cation=True, data_path=datapath)
    else:
        species = AtomicDensity(elem, anion=True, cation=False, data_path=datapath)

    # Check multiplicity value
    mo_occ = species.orbitals_occupation.ravel()
    multiplicity = int(np.sum(mo_occ[: len(mo_occ) // 2] - mo_occ[len(mo_occ) // 2 :])) + 1
    if mult != multiplicity:
        raise ValueError(f"Multiplicity {mult} is not available for {elem} with charge {charge}")

    # Get electronic structure data
    energy = species.energy[0]  # get energy from list
    norba = len(mo_occ) // 2
    # Get MO energies and occupations
    mo_e_up = species.orbitals_energy.ravel()[:norba]
    mo_e_dn = species.orbitals_energy.ravel()[norba:]
    occs_up, occs_dn = mo_occ[:norba], mo_occ[norba:]

    # Make grid
    onedg = UniformInteger(NPOINTS)  # number of uniform grid points.
    rgrid = ExpRTransform(*BOUND).transform_1d_grid(onedg)  # radial grid

    # Evaluate properties on the grid:
    # --------------------------------
    rs = rgrid.points

    # total and spin-up orbital, and spin-down orbital densities
    orb_dens_up = species.eval_orbs_density(rs)[:norba, :]
    orb_dens_dn = species.eval_orbs_density(rs)[norba:, :]
    dens_tot = species.eval_density(rs, mode="total")

    # total, spin-up orbital, and spin-down orbital first (radial) derivatives of the density
    d_dens_tot = species.eval_radial_d_density(rs)
    orb_d_dens_up = species.eval_orbs_radial_d_density(rs)[:norba, :]
    orb_d_dens_dn = species.eval_orbs_radial_d_density(rs)[norba:, :]

    # total, spin-up orbital, and spin-down orbital second (radial) derivatives of the density
    dd_dens_tot = species.eval_radial_dd_density(rs)
    orb_dd_dens_up = species.eval_orbs_radial_dd_density(rs)[:norba, :]
    orb_dd_dens_dn = species.eval_orbs_radial_dd_density(rs)[norba:, :]

    # total, spin-up orbital, and spin-down orbital kinetic energy densities
    ked_tot = species.eval_ked_positive_definite(rs)
    mo_ked_a = species.eval_orbs_ked_positive_definite(rs)[:norba, :]
    mo_ked_b = species.eval_orbs_ked_positive_definite(rs)[:norba, :]

    # Get information about the element
    atom = Element(elem)
    atmass = atom.mass["stb"]
    cov_radius, vdw_radius, at_radius, polarizability, dispersion = [
        None,
    ] * 5
    # overwrite values for neutral atomic species
    if charge == 0:
        cov_radius, vdw_radius, at_radius = (atom.cov_radius, atom.vdw_radius, atom.at_radius)
        polarizability = atom.pold
        dispersion = {"C6": atom.c6}

    # Conceptual-DFT properties (WIP)
    ip = -mo_e_up[np.sum(occs_up) - 1]  # - energy of HOMO
    # ea = -mo_e_dn[np.sum(occs_dn)] if np.sum(occs_dn) < np.sum(occs_up) else None  # - LUMO energy
    mu = None
    eta = None

    # Return Species instance
    fields = dict(
        elem=elem,
        atnum=atnum,
        obasis_name="Slater",
        nelec=nelec,
        nspin=nspin,
        nexc=nexc,
        atmass=atmass,
        cov_radius=cov_radius,
        vdw_radius=vdw_radius,
        at_radius=at_radius,
        polarizability=polarizability,
        dispersion=dispersion,
        energy=energy,
        mo_energy_a=mo_e_up,
        mo_energy_b=mo_e_dn,
        mo_occs_a=occs_up,
        mo_occs_b=occs_dn,
        ip=ip,
        mu=mu,
        eta=eta,
        rs=rs,
        # Density
        mo_dens_a=orb_dens_up.flatten(),
        mo_dens_b=orb_dens_dn.flatten(),
        dens_tot=dens_tot,
        # Density gradient
        mo_d_dens_a=orb_d_dens_up.flatten(),
        mo_d_dens_b=orb_d_dens_dn.flatten(),
        d_dens_tot=d_dens_tot,
        # Density laplacian
        mo_dd_dens_a=orb_dd_dens_up.flatten(),
        mo_dd_dens_b=orb_dd_dens_dn.flatten(),
        dd_dens_tot=dd_dens_tot,
        # KED
        mo_ked_a=mo_ked_a.flatten(),
        mo_ked_b=mo_ked_b.flatten(),
        ked_tot=ked_tot,
    )
    return atomdb.Species(dataset, fields)
