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

from .api import DEFAULT_DATAPATH, DEFAULT_DATASET
from .api import load, element_number

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
        # Define the property as a function, and call `_extensive_global_property` on it
        f = lambda atom, radii: atom.dens_spline(radii, spin=spin, log=log)
        return _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f
        )

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
        f = lambda atom, radii: atom.ked_spline(radii, spin=spin, log=log)
        return _extensive_local_property(
            self.atoms, self.coords, self.coeffs, points, f
        )

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


def make_promolecule(
    atoms,
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
    atoms: list of str
        List of element symbols for each atom.
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
    # Handle default "None" parameters
    if charges is None:
        charges = [0] * len(atoms)
    if mults is None:
        mults = [1] * len(atoms)
    # Construct linear combination of species
    promol_species = []
    promol_coords = []
    promol_coeffs = []
    for (atom, coord, charge, mult) in zip(atoms, coords, charges, mults):
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
                specie = load(
                    atom, np.floor(charge), mult, dataset=dataset, datapath=datapath
                )
                promol_species.append(specie)
                promol_coords.append(coord)
                promol_coeffs.append(np.ceil(charge) - charge)
            except FileNotFoundError:
                specie = load(
                    atom, np.ceil(charge), mult, dataset=dataset, datapath=datapath
                )
                promol_species.append(specie)
                promol_coords.append(coord)
                promol_coeffs.append(
                    (element_number(atom) - charge)
                    / (element_number(atom) - np.ceil(charge))
                )
                warn(
                    "Coefficient of a species in the promolecule is >1, intensive properties might be incorrect"
                )
            # Ceilling charge
            specie = load(
                atom, np.ceil(charge), mult, dataset=dataset, datapath=datapath
            )
            promol_species.append(specie)
            promol_coords.append(coord)
            promol_coeffs.append(charge - np.floor(charge))
    # Check coordinate units, convert to array
    units = units.lower()
    promol_coords = np.asarray(promol_coords, dtype=float)
    if units == "bohr":
        promol_coords /= 1.0
    elif units == "angstrom":
        promol_coords /= 1.8897259885789
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


def _extensive_local_property(atoms, atom_coords, coeffs, points, f):
    r"""Helper function for computing extensive local properties."""
    # Add contribution from each atom, calculating the radius between
    # the points of interest and each atom inside the generator
    return sum(
        coeff * f(atom, np.linalg.norm(points - coord, axis=1))
        for (atom, coord, coeff) in zip(atoms, atom_coords, coeffs)
    )


def _intensive_property(atoms, coeffs, f, p=1):
    r"""Helper function for computing intensive properties."""
    # P-mean of each atom's property value
    return (
        sum(coeff * f(atom) ** p for atom, coeff in zip(atoms, coeffs)) / sum(coeffs)
    ) ** (1 / p)


def _write_cube(fname, atnums, coords, charges, cb_origin, cb_shape, cb_axis, vdata):
    """_summary_

    Parameters
    ----------
    fname : srt
        File name
    atnums : list
        Atomic numbers
    coords : np.array
        Atomic coordinates
    charges : list
        Atomic charges
    cb_origin : np.array
        Box origin.
    cb_shape : list
        Box resolution on each axis
    cb_axis : np.array
        Box (X, Y, Z) axis vectors. 3D Matrix
    vdata : np.array
        Volumetri data
    """
    comments = (
        " PROMOLECULE CUBE FILE\n  'OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z'\n"
    )
    natom = len(atnums)
    nx, ny, nz = cb_shape
    _format = (
        lambda scalar, vector: f"{scalar:5d}"
        + "".join(f"{v:11.6f}" for v in vector)
        + "\n"
    )
    with open(fname, "w") as cube:
        # Header section
        cube.write(comments)
        cube.write(_format(natom, cb_origin))  # 3rd line #atoms and origin
        for i in range(3):
            cube.write(_format(cb_shape[i], cb_axis[i]))  # axis #voxels and vector
        for z, q, xyz in zip(atnums, charges, coords):
            qxyz = [q] + xyz.tolist()
            cube.write(_format(z, qxyz))  # atom#, charge and coordinates
        # Volumetric data
        vdata = vdata.reshape((nx, ny, nz))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cube.write(f" {vdata[ix, iy, iz]:12.5E}")
                    if iz % 6 == 5:
                        cube.write("\n")
                cube.write("\n")
