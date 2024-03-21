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

r"""AtomDB promolecule submodule."""

from copy import deepcopy
from itertools import chain, combinations
from numbers import Integral
from operator import itemgetter
from warnings import warn

import numpy as np
from scipy.optimize import linprog

from .api import (
    DEFAULT_DATAPATH,
    DEFAULT_DATASET,
    MULTIPLICITIES,
    element_number,
    element_symbol,
    load,
    load_all,
)

__all__ = [
    "Promolecule",
    "make_promolecule",
]


class Promolecule:
    r"""
    Promolecule class.

    A promolecule is an approximation of a molecule constructed from a linear combination of atomic
    and/or ionic species. Properties of this promolecule can be computed from those of the atomic
    and/or ionic species, depending on whether the property is an extensive one or an intensive one.

    For an extensive property, the formula is a linear combination:

    .. math::

        \text{prop.}_{\text{mol;extensive}}
            = \sum_{A=1}^{N_{\text{atoms}}} c_A \text{prop.}_A

    For an intensive property, the formula is a mean:

    .. math::

        \text{prop.}_{\text{mol;intensive}}
          = {\left\langle \left\{ \text{prop.}_A \right\}_{A=1}^{N_{\text{atoms}}} \right\rangle}_p

    where the parameter ``p`` defines the type of mean used (1 = linear, 2 = geometric, etc.).

    Attributes
    ----------
    atoms: list of Species
        Species instances out of which the promolecule is composed.
    coords: np.ndarray((N, 3), dtype=float)
        Coordinates of each species component of the promolecule.
    coeffs: np.ndarray((N,), dtype=float)
        Coefficients of each species component of the promolecule.

    Methods
    -------
    density(points, spin='ab', log=False)
        Compute the electron density of the promolecule at the desired points.
    ked(points, spin='ab', log=False)
        Compute the kinetic energy density of the promolecule at the desired points.
    energy()
        Compute the energy of the promolecule.
    mass()
        Compute the mass of the promolecule.
    ip(p=1)
        Compute the ionization potential of the promolecule.
    mu(p=1)
        Compute the chemical potential of the promolecule.
    eta(p=1)
        Compute the chemical hardness of the promolecule.

    """

    def __init__(self):
        r"""
        Initialize a Promolecule instance.

        """
        self.atoms = []
        self.coords = []
        self.coeffs = []

    def _extend(self, atoms, coords, coeffs):
        r"""
        Add species to a Promolecule instance.

        Parameters
        ----------
        atoms: list of Species
            Species instances out of which the promolecule is composed.
        coords: np.ndarray((N, 3), dtype=float)
            Coordinates of each species component of the promolecule.
        coeffs: np.ndarray((N,), dtype=float)
            Coefficients of each species component of the promolecule.
        mult: (int|float)
            Multiplicity on the center.

        """
        self.atoms.extend(atoms)
        self.coords.extend(np.asarray(coord, dtype=float) for coord in coords)
        self.coeffs.extend(coeffs)

    def density(self, points, spin="ab", log=False):
        r"""
        Compute the electron density of the promolecule at the desired points.

        Parameters
        ----------
        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin,
            or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        """

        # Define the property as a function, and call `_extensive_local_property` on it
        def f(atom):
            return atom.interpolate_dens(spin=spin, log=log)

        return sum(_extensive_local_property(self.atoms, self.coords, self.coeffs, points, f))

    def ked(self, points, spin="ab", log=False):
        r"""
        Compute the kinetic energy density of the promolecule at the desired points.

        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin,
            or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        """

        def f(atom):
            return atom.interpolate_ked(spin=spin, log=log)

        return sum(_extensive_local_property(self.atoms, self.coords, self.coeffs, points, f))

    def nelec(self):
        r"""Compute the electron number of the promolecule."""

        def f(atom):
            return atom.nelec

        return _extensive_global_property(self.atoms, self.coeffs, f)

    def charge(self):
        r"""Compute the charge of the promolecule."""

        # return sum(atom.natom for atom in self.atoms) - self.nelec()
        def f(atom):
            return atom.charge

        return _extensive_global_property(self.atoms, self.coeffs, f)

    def energy(self):
        r"""Compute the energy of the promolecule."""

        def f(atom):
            return atom.energy

        return _extensive_global_property(self.atoms, self.coeffs, f)

    def mass(self):
        r"""Compute the mass of the promolecule."""

        def f(atom):
            return atom.mass

        return _extensive_global_property(self.atoms, self.coeffs, f)

    def nspin(self, p=1):
        r"""Compute the spin number of the promolecule."""

        def f(atom):
            return atom.nspin * atom.spinpol

        return _intensive_property(self.atoms, self.coeffs, f, p=1)

    def mult(self, p=1):
        r"""Compute the multiplicity of the promolecule."""
        return abs(self.nspin(p=1)) + 1

    def ip(self, p=1):
        r"""
        Compute the ionization potential of the promolecule.

        Parameters
        ----------
        p: int, default=1
            Value of ``p`` for the p-mean computation of this intensive property.

        """

        # Define the property as a function, and call `_intensive_property` on it
        def f(atom):
            return atom.ip

        return _intensive_property(self.atoms, self.coeffs, f, p=p)

    def mu(self, p=1):
        r"""
        Compute the chemical potential of the promolecule.

        Parameters
        ----------
        p: int, default=1
            Value of ``p`` for the p-mean computation of this intensive property.

        """

        # Define the property as a function, and call `_intensive_property` on it
        def f(atom):
            return atom.mu

        return _intensive_property(self.atoms, self.coeffs, f, p=p)

    def eta(self, p=1):
        r"""
        Compute the chemical hardness of the promolecule.

        Parameters
        ----------
        p: int, default=1
            Value of ``p`` for the p-mean computation of this intensive property.

        """

        # Define the property as a function, and call `_intensive_property` on it
        def f(atom):
            return atom.eta

        return _intensive_property(self.atoms, self.coeffs, f, p=p)

    def gradient(self, points, spin="ab", log=False):
        r"""
        Compute the electron density gradient of the promolecule at the desired points.

        Promolecular gradient:
        .. math::
            \nabla \rho_{\text{mol}}^{(0)} (\mathbf{R})
                = \sum_{A=1}^{N_{\text{atoms}}} c_A \nabla \rho_A^{(0)}(\mathbf{R})

        where,
            :math:`N_{\text{atoms}}` is the number of atoms in the molecule.
            :math:`c_A` are the coefficients of the specie.
            :math:`R` are points in 3D cartesian coordinates.
            :math:`\nabla \rho_A^{(0)}(\mathbf{R})` is the gradient of specie A.

        Parameters
        ----------
        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin,
            or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        Returns
        -------
        gradient: np.ndarray((N, 3), dtype=float)

        """

        # Define the property as a function, and call `_extensive_local_property` on it
        def f(atom):
            return atom.interpolate_dens(spin=spin, log=log)

        atoms_ddens = _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f, deriv=1
        )

        # The cartesian gradient (\nabla \rho_A) has to be evaluated using the chain rule:
        # \nabla \rho_A(x,y,z) = \frac{\partial \rho_A}{\partial r} *  \hat{r}
        # where `\frac{\partial \rho_A}{\partial r}` is the gradient in spherical coordinates
        # and   \hat{r} = [dr/dx, dr/dy, dr/dz] is the radial unit vector:
        # dr/dx = (x - x_A) / |r-R_A|
        #
        # Define a unit vector function
        def unit_v(vector):
            return [dr / np.linalg.norm(dr) for dr in vector]

        gradients_atoms = [
            ddens[:, None] * unit_v(points - coord)
            for (ddens, coord) in zip(atoms_ddens, self.coords)
        ]
        return sum(gradients_atoms)

    def hessian(self, points, spin="ab", log=False):
        r"""
        Compute the promolecule's electron density Hessian at the desired points.

        Parameters
        ----------
        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin,
            or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        """

        # Define the property as a function, and call `_extensive_local_property` on it
        def f(atom):
            return atom.interpolate_dens(spin=spin, log=log)

        atoms_ddens = _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f, deriv=1
        )
        atoms_d2dens = _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f, deriv=2
        )

        # Evaluate the derivatives of the radii
        atoms_router_triu = np.array(
            [_radial_vector_outer_triu(points - coord) for coord in self.coords]
        )

        # Get the unique elements of the Hessian
        # \sum_A c_A [d^2f_A/dx^2, d^2f_A/dxdy, d^2f_A/dxdz, d^2f_A/dydz, d^2f_A/dz]
        # where
        # d^2f_A/dx^2 = (d^2f_A/dr^2 - df_A/dr) (dr_A/dx)^2 + shift
        # d^2f_A/dxdy = (d^2f_A/dr^2 - df_A/dr) d^r_A/dx d^r_A/dy
        interm = 0
        for d2dens, ddens, rhess in zip(atoms_d2dens, atoms_ddens, atoms_router_triu):
            interm += (d2dens - ddens)[:, None] * rhess

        # Reconstruct the Hessian matrix
        mask = [0, 1, 2, 4, 5, 8]  # upper triangular indexes row-major order
        hess = np.zeros((len(points), 9))
        hess[:, mask] = interm
        hess = hess.reshape(len(points), 3, 3)
        # Shift diagonal terms by \sum_A c_A df_A/dr |r - R_A|^(-1)
        for i in range(3):
            hess[:, i, i] += sum(
                ddens / np.linalg.norm(points - coord)
                for (ddens, coord) in zip(atoms_ddens, self.coords)
            )
        hess += hess.transpose((0, 2, 1))
        for p in range(len(points)):
            hess[p] = hess[p] - np.diag(np.diag(hess[p]) / 2)
        return hess

    def laplacian(self, points, spin="ab", log=False):
        r"""
        Compute the promolecule's electron density Laplacian at the desired points.

        Parameters
        ----------
        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin,
            or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        """

        def f(atom):
            return atom.interpolate_dens(spin=spin, log=log)

        def shift(dens, radii):
            return 3 * dens / np.linalg.norm(radii)

        # Radial derivatives of the density
        atoms_ddens = _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f, deriv=1
        )
        atoms_d2dens = _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f, deriv=2
        )

        return sum(
            d2dens - ddens + shift(ddens, points - coord)
            for (d2dens, ddens, coord) in zip(atoms_d2dens, atoms_ddens, self.coords)
        )


def make_promolecule(
    atnums,
    coords,
    charges=None,
    mults=None,
    units=None,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""
    Construct a Promolecule instance from a set of atoms and their coordinates, charges,
    and multiplicities.

    Parameters
    ----------
    atnums: list of (str|int)
        List of element number for each atom.
    coords: list of np.ndarray((3,), dtype=float)
        List of coordinates for each atom.
    charges: list of (int | float), default=[0, ..., 0]
        List of charges.
    mults: list of (int), default=[1, ..., 1]
        List of multiplicities for each atom.
    units: ('bohr' | 'angstrom')
        Units of ``coords`` values. Default is Bohr.
    dataset: str, default=DEFAULT_DATASET
        Data set from which to load atomic species.
    datapath: str, default=DEFAULT_DATAPATH
        System path where the desired data set is located.

    """
    # Check coordinate units
    if units is None or units.lower() == "bohr":
        coords = [coord / 1 for coord in coords]
    elif units.lower() == "angstrom":
        coords = [coord / 0.52917721092 for coord in coords]
    else:
        raise ValueError("Invalid `units` parameter; must be 'bohr' or 'angstrom'")

    # Get atomic symbols/numbers from inputs
    atoms = [element_symbol(atom) for atom in atnums]
    atnums = [element_number(atom) for atom in atnums]

    # Handle default charge parameters
    if charges is None:
        charges = [0 for _ in atnums]

    # Handle default multiplicity parameters
    if mults is None:
        # Force non-int charge to be integer here; it will be overwritten below.
        mults = [
            MULTIPLICITIES[max(1, int(np.round(atnum - charge)))]
            for (atnum, charge) in zip(atnums, charges)
        ]

    # Construct linear combination of species
    promol = Promolecule()

    for atom, atnum, coord, charge, mult in zip(atoms, atnums, coords, charges, mults):

        # Integer charge and multiplicity
        #
        if isinstance(charge, Integral) and isinstance(mult, Integral):
            try:
                specie = load(atom, charge, abs(mult), dataset=dataset, datapath=datapath)
                if mult < 0:
                    specie.spinpol = -1
                promol._extend((specie,), (coord,), (1.0,))
                continue
            except FileNotFoundError:
                warn(
                    "Unable to load species corresponding to `charge, mult`; "
                    "generating species via linear combination of other species'",
                    stacklevel=1,
                )

        # Non-integer charge and multiplicity
        #
        nelec = atnum - charge
        nspin = np.sign(mult) * (abs(mult) - 1)
        # Get all candidates for linear combination
        species_list = load_all(atom, dataset=dataset, datapath=datapath)
        for specie in species_list[: len(species_list)]:
            if specie.nspin > 0:
                specie_neg_spinpol = deepcopy(specie)
                specie_neg_spinpol.spinpol = -1
                species_list.append(specie_neg_spinpol)

        trial_species = chain(combinations(species_list, 2), combinations(species_list, 3))
        good_combs = []
        for ts in trial_species:
            energies = np.asarray([t.energy for t in ts], dtype=float)
            result = linprog(
                energies,
                A_eq=np.asarray(
                    [
                        [1 for t in ts],
                        [t.nelec for t in ts],
                        [t.nspin * t.spinpol for t in ts],
                    ],
                    dtype=float,
                ),
                b_eq=np.asarray([1, nelec, nspin], dtype=float),
                bounds=(0, 1),
            )
            if result.success:
                good_combs.append((np.dot(energies, result.x), ts, [coord for t in ts], result.x))
        if len(good_combs) > 0:
            promol._extend(*(min(good_combs, key=itemgetter(0))[1:]))
        else:
            raise ValueError(
                "Unable to construct species with non-integer charge/spin from database entries"
            )

    # Return Promolecule instance
    return promol


def _extensive_global_property(atoms, coeffs, f):
    r"""Helper function for computing extensive global properties."""
    # Weighted sum of each atom's property value
    return sum(coeff * f(atom) for atom, coeff in zip(atoms, coeffs))


def _extensive_local_property(atoms, atom_coords, coeffs, points, f, deriv=0):
    r"""Helper function for computing extensive local properties."""
    # Add contribution from each atom, calculating the radius between
    # the points of interest and each atom inside the generator
    splines = [f(atom) for atom in atoms]
    return [
        coeff * spline(np.linalg.norm(points - coord, axis=1), deriv=deriv)
        for (spline, coord, coeff) in zip(splines, atom_coords, coeffs)
    ]


def _intensive_property(atoms, coeffs, f, p=1):
    r"""Helper function for computing intensive properties."""
    # P-mean of each atom's property value
    return (sum(coeff * f(atom) ** p for atom, coeff in zip(atoms, coeffs)) / sum(coeffs)) ** (
        1 / p
    )


def _radial_vector_outer_triu(radii):
    r"""Evaluate the outer products of a set of radial unit vectors."""

    # Define a unit vector function
    def unit_v(vector):
        return vector / np.linalg.norm(vector)

    # Store only upper triangular elements of the matrix.
    indices = [0, 1, 2, 4, 5, 8]  # row-major order (ij = 3 * i + j)
    radv_outer = np.empty((len(radii), len(indices)))
    # Outer product of each radial unit vector
    for col, ij in enumerate(indices):
        radv_outer[:, col] = unit_v(radii)[:, ij // 3] * unit_v(radii)[:, ij % 3]
    return radv_outer


def _cart_to_bary(x0, y0, s1, s2, s3):
    r"""Helper function for computing barycentric coordinates."""
    x1, x2, x3 = s1.nelec, s2.nelec, s3.nelec
    y1, y2, y3 = s1.nspin, s2.nspin, s3.nspin
    lambda1 = (
        (y2 - y3) * (x0 - x3)
        + (x3 - x2) * (y0 - y3) / (y2 - y3) * (x1 - x3)
        + (x3 - x2) * (y1 - y3)
    )
    lambda2 = (
        (y3 - y1) * (x0 - x3)
        + (x1 - x3) * (y0 - y3) / (y2 - y3) * (x1 - x3)
        + (x3 - x2) * (y1 - y3)
    )
    lambda3 = 1 - lambda1 - lambda2
    return (lambda1, lambda2, lambda3)
