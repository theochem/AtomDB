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

r"""
Module responsible for reading and storing atomic wave-function information from '.slater' files and
    computing electron density, and kinetic energy from them.

AtomicDensity:
    Information about atoms obtained from .slater file and able to construct atomic density
        (total, core and valence) from the linear combination of Slater-type orbitals.
    Elements supported by default from "./atomdb/data/slater_atom/" range from Hydrogen to Xenon.

load_slater_wfn : Function for reading and returning information from '.slater' files that consist
    of anion, cation and neutral atomic wave-function information.

"""


import numpy as np
import os
import re
from scipy.special import factorial
import atomdb
from atomdb.periodic import Element


__all__ = ["AtomicDensity", "load_slater_wfn", "run"]


BOUND = (1e-5, 15.0)

NPOINTS = 10000


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
        Ordered based on "S", "P", "D", etc.
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

    def __init__(self, element, anion=False, cation=False):
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

        """
        if not isinstance(element, str) or not element.isalpha():
            raise TypeError("The element argument should be all letters string.")

        data = load_slater_wfn(element, anion, cation)
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

    def phi_matrix(self, points, deriv=False):
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
        deriv : bool
            If true, use the derivative of the slater-orbitals.

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
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            if deriv:
                slater = self.derivative_slater_type_orbital(exps, number, points)
            else:
                slater = self.slater_orbital(exps, number, points)
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def atomic_density(self, points, mode="total"):
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
            orb_homo = self.orbitals_energy[len(self.orbitals_occupation) - 1]
            orb_occs = orb_occs * np.exp(-((self.orbitals_energy - orb_homo) ** 2))
        elif mode == "core":
            orb_homo = self.orbitals_energy[len(self.orbitals_occupation) - 1]
            orb_occs = orb_occs * (1.0 - np.exp(-((self.orbitals_energy - orb_homo) ** 2)))
        # compute density
        dens = np.dot(self.phi_matrix(points) ** 2, orb_occs).ravel() / (4 * np.pi)
        return dens

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
            The Slater-type orbitals evaluated on the grid points.

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

    def lagrangian_kinetic_energy(self, points):
        r"""
        Positive definite or Lagrangian kinectic energy density.

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

    def derivative_density(self, points):
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
        factor = self.phi_matrix(points) * self.phi_matrix(points, deriv=True)
        derivative = np.dot(2.0 * factor, self.orbitals_occupation).ravel() / (4 * np.pi)
        return derivative


def load_slater_wfn(element, anion=False, cation=False):
    """
    Return the data recorded in the atomic Slater wave-function file as a dictionary.

    Parameters
    ----------
    file_name : str
        The path to the Slater atomic file.
    anion : bool
        If true, then the anion of element is used.
    cation : bool
        If true, then the cation of element is used.

    """
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

    is_heavy_element = element.lower() in heavy_atoms
    if (anion or cation) and is_heavy_element:
        raise ValueError("Both Anion & Cation Slater File for element %s does not exist." % element)
    if anion:
        if element.lower() in anion_atoms:
            # file_path = "./atomdb/data/anion/%s.an" % element.lower()
            file_path = "raw/anion/%s.an" % element.lower()
        else:
            raise ValueError("Anion Slater File for element %s does not exist." % element)
    elif cation:
        if element.lower() in cation_atoms:
            # file_path = "./atomdb/data/cation/%s.cat" % element.lower()
            file_path = "raw/cation/%s.cat" % element.lower()
        else:
            raise ValueError("Cation Slater File for element %s does not exist." % element)
    else:
        # file_path = "./atomdb/data/neutral/%s.slater" % element.lower()
        file_path = "raw/neutral/%s.slater" % element.lower()

    file_name = os.path.join(os.path.dirname(__file__), file_path)

    def get_number_of_electrons_per_orbital(configuration):
        """
        Get the Occupation Number for all orbitals of an _element returing an dictionary.

        Parameters
        ----------
        configuration : str
            The electron configuration.

        Returns
        --------
        dict
            a dict containing the number and orbital.

        """
        electron_config_list = configuration

        shells = ["K", "L", "M", "N"]

        out = {}
        orbitals = (
            [str(x) + "S" for x in range(1, 8)]
            + [str(x) + "P" for x in range(2, 8)]
            + [str(x) + "D" for x in range(3, 8)]
            + [str(x) + "F" for x in range(4, 8)]
        )
        for orb in orbitals:
            # Initialize all atomic orbitals to zero electrons
            out[orb] = 0

        for x in shells:
            if x in electron_config_list:
                if x == "K":
                    out["1S"] = 2
                elif x == "L":
                    out["2S"] = 2
                    out["2P"] = 6
                elif x == "M":
                    out["3S"] = 2
                    out["3P"] = 6
                    out["3D"] = 10
                elif x == "N":
                    out["4S"] = 2
                    out["4P"] = 6
                    out["4D"] = 10
                    out["4F"] = 14

        for x in orbitals:
            if x in electron_config_list:
                index = electron_config_list.index(x)
                orbital = electron_config_list[index : index + 2]

                if orbital[1] == "D" or orbital[1] == "F":
                    # num_electrons = re.sub('[(){}<>,]', "", electron_config_list.split(orbital)[1])
                    num_electrons = re.search(orbital + r"\((.*?)\)", electron_config_list).group(1)
                    out[orbital] = int(num_electrons)
                else:
                    out[orbital] = int(electron_config_list[index + 3 : index + 4])

        return {key: value for key, value in out.items() if value != 0}

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
        line = f.readline()
        configuration = line.split()[1].replace(",", "")
        if is_heavy_element:
            configuration = configuration_exact_for_heavy_elements(configuration)

        next_line = f.readline()
        # Sometimes there are blank lin es.
        while len(next_line.strip()) == 0:
            next_line = f.readline()

        if is_heavy_element:
            # Heavy element slater files has extra redundant information of 5 lines.
            for i in range(0, 6):
                f.readline()

        next_line = f.readline()
        energy = [float(next_line.split()[2])] + [
            float(x) for x in (re.findall(r"[= -]\d+.\d+", f.readline()))[:-1]
        ]

        orbitals = []
        orbitals_basis = {"S": [], "P": [], "D": [], "F": []}
        orbitals_cusp = []
        orbitals_energy = []
        orbitals_exp = {"S": [], "P": [], "D": [], "F": []}
        orbitals_coeff = {}

        line = f.readline()
        while line.strip() != "":
            # If line has ___S___ or P or D where _ = " ".
            if re.search(r"  [S|P|D|F]  ", line):
                # Get All The Orbitals
                subshell = line.split()[0]
                list_of_orbitals = line.split()[1:]
                orbitals += list_of_orbitals
                for x in list_of_orbitals:
                    orbitals_coeff[x] = []  # Initilize orbitals inside coefficient dictionary

                # Get Energy, Cusp Levels
                line = f.readline()
                orbitals_energy.extend([float(x) for x in line.split()[1:]])
                if not is_heavy_element:
                    # Heavy atoms slater files, doesn't have cusp values,.
                    line = f.readline()
                    orbitals_cusp.extend([float(x) for x in line.split()[1:]])
                line = f.readline()

                # Get Exponents, Coefficients, Orbital Basis
                while re.match(r"\A^\d" + subshell, line.lstrip()):

                    list_words = line.split()
                    orbitals_exp[subshell] += [float(list_words[1])]
                    orbitals_basis[subshell] += [list_words[0]]

                    for x in list_of_orbitals:
                        orbitals_coeff[x] += [float(list_words[get_column(x)])]
                    line = f.readline()
            else:
                line = f.readline()

    data = {
        "configuration": configuration,
        "energy": energy,
        "orbitals": orbitals,
        "orbitals_energy": np.array(orbitals_energy)[:, None],
        "orbitals_cusp": np.array(orbitals_cusp)[:, None],
        "orbitals_basis": orbitals_basis,
        "orbitals_exp": {
            key: np.asarray(value).reshape(len(value), 1)
            for key, value in orbitals_exp.items()
            if value != []
        },
        "orbitals_coeff": {
            key: np.asarray(value).reshape(len(value), 1)
            for key, value in orbitals_coeff.items()
            if value != []
        },
        "orbitals_occupation": np.array(
            [get_number_of_electrons_per_orbital(configuration)[k] for k in orbitals]
        )[:, None],
        "basis_numbers": {
            key: np.asarray([[int(x[0])] for x in value])
            for key, value in orbitals_basis.items()
            if len(value) != 0
        },
    }

    return data


def split_configuration(orbitals, occupations):
    r"""Split electronic configuration into alpha and beta components following Hund's rule

    Returns
    -------
    occs_a : list
        alpha electronic configuration for each orbital (M values)
    occs_b : list
        beta electronic configuration for each orbital (M values)
    """
    subshell_alphas = {"S": 1, "P": 3, "D": 5, "F": 7}
    occs_a = []
    occs_b = []
    for i, orb in enumerate(orbitals):
        n_el = occupations[i] - 1  # N electrons in sub shell - 1
        na_sshell = subshell_alphas[orb[-1]]  # N alpha electrons in sub shell
        row = n_el // na_sshell
        col = n_el % na_sshell
        if row == 0:
            na = (row * na_sshell + col) + 1
            nb = 0
        else:
            na = na_sshell
            nb = (row * na_sshell + col) - na + 1
        occs_a.append(na)
        occs_b.append(nb)

    return np.array(occs_a, dtype=float), np.array(occs_b, dtype=float)


def eval_multiplicity(orbitals, occupations):
    r"""Evaluate multiplicity

    Parameters
    ----------
    orbitals : list, (M,)
        List of strings representing each of the orbitals in the electron configuration.
        Ordered based on "S", "P", "D", etc. For example, Beryllium has ["1S", "2S"] in its electron
        configuration.
    occupations : ndarray, (M,)
        Number of electrons in each of the orbitals in the electron configuration.

    Returns
    -------
    Spin multiplicity : int

    """
    occs_a, occs_b = split_configuration(orbitals, occupations)
    na = sum(occs_a)
    nb = sum(occs_b)
    return int((na - nb)) + 1


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

    # Get information about the element
    atom = Element(elem)
    atmass = atom.mass["stb"]
    cov_radius, vdw_radius, at_radius, polarizability, dispersion_c6 = [
        None,
    ] * 5
    if charge == 0:
        # overwrite values for neutral atomic species
        cov_radius, vdw_radius, at_radius = (atom.cov_radius, atom.vdw_radius, atom.at_radius)
        polarizability = atom.pold
        dispersion_c6 = atom.c6

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    atnum = atomdb.element_number(elem)
    nelec = atnum - charge
    nspin = mult - 1
    obasis_name = None

    # Retrieve Slater data
    if charge == 0:
        species = AtomicDensity(elem, anion=False, cation=False)
    elif charge > 0:
        species = AtomicDensity(elem, anion=False, cation=True)
    else:
        species = AtomicDensity(elem, anion=True, cation=False)

    # Check multiplicity value
    mo_occ = species.orbitals_occupation.ravel()  # these are configurations not occupations
    multiplicity = eval_multiplicity(species.orbitals, mo_occ)
    if mult != multiplicity:
        raise ValueError(f"Multiplicity {mult} is not available for {elem} with charge {charge}")

    # Get electronic structure data
    # FIXME:sign error in parsed energy value (looks like T value instead of E was parsed from raw file).
    # This is a temporal fix until the parsing code for Slater's data gets updated from BFit.
    energy = -species.energy[0]
    mo_energies_a = species.orbitals_energy.ravel()  # assuming same alpha and beta energies
    mo_occ_a, mo_occ_b = split_configuration(species.orbitals, mo_occ)

    # Make grid
    points = np.linspace(*BOUND, NPOINTS)

    # Compute densities and derivatives
    dens_orbs = species.phi_matrix(points) ** 2 / (4 * np.pi)
    dens_tot = species.atomic_density(points, "total")
    # dens_core = species.atomic_density(points, "core")
    # dens_valence = species.atomic_density(points, "valence")
    d_dens_tot = species.derivative_density(points)

    # Compute laplacian and kinetic energy density
    # lapl_tot = None
    ked_tot = species.lagrangian_kinetic_energy(points)

    # Return Species instance
    fields = dict(
        elem=elem,
        atnum=atnum,
        obasis_name=obasis_name,
        nelec=nelec,
        nspin=nspin,
        nexc=nexc,
        # atmass=atmass,
        # cov_radius=cov_radius,
        # vdw_radius=vdw_radius,
        # at_radius=at_radius,
        # polarizability=polarizability,
        # dispersion_c6=dispersion_c6,
        energy=energy,
        mo_energy_a=mo_energies_a,
        mo_energy_b=mo_energies_a,
        mo_occs_a=mo_occ_a,
        mo_occs_b=mo_occ_b,
        rs=points,
        dens_tot=dens_tot,
        d_dens_tot=d_dens_tot,
        ked_tot=ked_tot,
    )
    return atomdb.Species(dataset, fields)
