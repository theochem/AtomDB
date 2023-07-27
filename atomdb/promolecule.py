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

from .api import DEFAULT_DATAPATH, DEFAULT_DATASET, MULTIPLICITIES
from .api import load, element_number, element_symbol

from numbers import Integral

from warnings import warn

import numpy as np


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

    def __init__(self, atoms, coords, coeffs):
        r"""
        Initialize a Promolecule instance.

        Parameters
        ----------
        atoms: list of Species
            Species instances out of which the promolecule is composed.
        coords: np.ndarray((N, 3), dtype=float)
            Coordinates of each species component of the promolecule.
        coeffs: np.ndarray((N,), dtype=float)
            Coefficients of each species component of the promolecule.

        """
        self.atoms = atoms
        self.coords = coords
        self.coeffs = coeffs

    def density(self, points, spin="ab", log=False):
        r"""
        Compute the electron density of the promolecule at the desired points.

        Parameters
        ----------
        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        """
        # Define the property as a function, and call `_extensive_local_property` on it
        f = lambda atom: atom.interpolate_dens(spin=spin, log=log)
        return sum(_extensive_local_property(self.atoms, self.coords, self.coeffs, points, f))

    def ked(self, points, spin="ab", log=False):
        r"""
        Compute the kinetic energy density of the promolecule at the desired points.

        points: np.ndarray((N, 3), dtype=float)
            Points at which to compute the density.
        spin: ('ab' | 'a' | 'b' | 'm'), default='ab'
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.

        """
        f = lambda atom: atom.interpolate_ked(spin=spin, log=log)
        return sum(_extensive_local_property(self.atoms, self.coords, self.coeffs, points, f))

    def energy(self):
        r"""Compute the energy of the promolecule."""
        f = lambda atom: atom.energy
        return _extensive_global_property(self.atoms, self.coeffs, f)

    def mass(self):
        r"""Compute the mass of the promolecule."""
        f = lambda atom: atom.mass
        return _extensive_global_property(self.atoms, self.coeffs, f)

    def ip(self, p=1):
        r"""
        Compute the ionization potential of the promolecule.

        Parameters
        ----------
        p: int, default=1
            Value of ``p`` for the p-mean computation of this intensive property.

        """
        # Define the property as a function, and call `_intensive_property` on it
        f = lambda atom: atom.ip
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
        f = lambda atom: atom.mu
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
        f = lambda atom: atom.eta
        return _intensive_property(self.atoms, self.coeffs, f, p=p)

    def gradient(self, points, spin="ab", log=False):
        r"""
        Compute the electron density gradient of the promolecule at the desired points.

        Promolecular gradient:
        .. math::
            \nabla \rho_{\text{mol}}^{(0)} (\mathbf{R}) = \sum_{A=1}^{N_{\text{atoms}}} c_A \nabla \rho_A^{(0)}(\mathbf{R})

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
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.
        
        Returns
        -------
        gradient: np.ndarray((N, 3), dtype=float)
        
        """
        # Define the property as a function, and call `_extensive_local_property` on it
        f = lambda atom: atom.interpolate_dens(spin=spin, log=log)
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
        unit_v = lambda vector: [dr/np.linalg.norm(dr ) for dr in vector]
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
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.
        
        """
        # Define the property as a function, and call `_extensive_local_property` on it
        f = lambda atom: atom.interpolate_dens(spin=spin, log=log)
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
        for (d2dens, ddens, rhess) in zip(atoms_d2dens, atoms_ddens, atoms_router_triu):
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
            Type of density to compute; either total, alpha-spin, beta-spin, or magnetization density.
        log: bool, default=False
            Whether to compute the log of the density instead of the density itself.
            May be slightly more accurate.
        
        """
        f = lambda atom: atom.interpolate_dens(spin=spin, log=log)
        shift = lambda dens, radii: 3 * dens / np.linalg.norm(radii)
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
    units="bohr",
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""
    Construct a Promolecule instance from a set of atoms and their coordinates, charges,
    and multiplicities.

    Parameters
    ----------
    atnums: list of int
        List of element number for each atom.
    coords: list of np.ndarray((3,), dtype=float)
        List of coordinates for each atom.
    charges: list of (int | float), default=[0, ..., 0]
        List of charges.
    mults: list of (int), default=[1, ..., 1]
        List of multiplicities for each atom.
    units: ('bohr' | 'angstrom')
        Units of ``coords`` values.
    dataset: str, default=DEFAULT_DATASET
        Data set from which to load atomic species.
    datapath: str, default=DEFAULT_DATAPATH
        System path where the desired data set is located.

    """
    # Get atomic symbols from inputs
    atoms = [element_symbol(atom) for atom in atnums]
    # Handle default charge and multiplicity parameters
    if charges is None:
        charges = [0 for _ in atoms]
    if mults is None:
        try:
            mults = [MULTIPLICITIES[atnum - charge] for (atnum, charge) in zip(atnums, charges)]
        except TypeError:
            # FIXME: force non-int charge to be integer here, It will be overwritten bellow.
            mults = [MULTIPLICITIES[atnum - int(charge)] for (atnum, charge) in zip(atnums, charges)]
    # Construct linear combination of species
    promol_species = []
    promol_coords = []
    promol_coeffs = []
    for (atom, atnum, coord, charge, mult) in zip(atoms, atnums, coords, charges, mults):
        if not isinstance(mult, Integral):
            raise ValueError("Non-integer multiplicity is invalid")
        if isinstance(charge, Integral):
            # Integer charge
            specie = load(atom, charge, mult, dataset=dataset, datapath=datapath)
            promol_species.append(specie)
            promol_coords.append(coord)
            promol_coeffs.append(1.0)
        else:
            # Floor charge
            try:
                charge_floor = np.floor(charge).astype(int)
                mult_floor = MULTIPLICITIES[atnum - charge_floor]
                specie = load(atom, charge_floor, mult_floor, dataset=dataset, datapath=datapath)
                promol_species.append(specie)
                promol_coords.append(coord)
                promol_coeffs.append(np.ceil(charge) - charge)
            except FileNotFoundError:
                specie = load(atom, np.ceil(charge), mult, dataset=dataset, datapath=datapath)
                promol_species.append(specie)
                promol_coords.append(coord)
                promol_coeffs.append((element_number(atom) - charge) / (element_number(atom) - np.ceil(charge)))
                warn("Coefficient of a species in the promolecule is >1, intensive properties might be incorrect")
            # Ceilling charge
            charge_ceil = np.ceil(charge).astype(int)
            mult_ceil = MULTIPLICITIES[atnum - charge_ceil]
            # FIXME: handle H^+
            if mult_ceil == 0:
                mult_ceil = 1
            specie = load(atom, charge_ceil, mult_ceil, dataset=dataset, datapath=datapath)
            promol_species.append(specie)
            promol_coords.append(coord)
            promol_coeffs.append(charge - np.floor(charge))
    # Check coordinate units, convert to array
    units = units.lower()
    promol_coords = np.asarray(promol_coords, dtype=float)
    if units == "bohr":
        promol_coords /= 1.0
    elif units == "angstrom":
        promol_coords /= 0.52917721092
    else:
        raise ValueError("Invalid `units` parameter; must be 'bohr' or 'angstrom'")
    # Convert coefficients to array
    promol_coeffs = np.asarray(promol_coeffs, dtype=float)
    # Return Promolecule instance
    return Promolecule(promol_species, promol_coords, promol_coeffs)


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
    return (
        sum(coeff * f(atom) ** p for atom, coeff in zip(atoms, coeffs)) / sum(coeffs)
    ) ** (1 / p)


def _radial_vector_outer_triu(radii):
    r""" Evaluate the outer products of a set of radial unit vectrors."""
    # Define a unit vector function
    unit_v = lambda vector: vector / np.linalg.norm(vector)
    # Store only upper triangular elements of the matrix.
    indices = [0, 1, 2, 4, 5, 8]  # row-major order (ij = 3 * i + j)
    radv_outer = np.empty((len(radii), len(indices)))
    # Outer product of each radial unit vector
    for col, ij in enumerate(indices):
        radv_outer[:, col] = unit_v(radii)[:, ij // 3] * unit_v(radii)[:, ij % 3]
    return radv_outer
