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

r"""AtomDB promolecule class."""

from .api import DEFAULT_DATAPATH, DEFAULT_DATASET
from .api import load

import numpy as np


__all__ = [
    "Promolecule",
]


class Promolecule:
    r"""Promolecule class."""

    def __init__(self, atoms, coords, charges=None, mults=None, coeffs=None, dataset=DEFAULT_DATASET, datapath=DEFAULT_DATAPATH):
        r"""
        """
        # Handle default charges and multiplicities
        if charges is None:
            charges = [0] * len(atoms)
        if mults is None:
            mults = [0] * len(atoms)
        # Set atoms from species files
        self.atoms = [
            load(atom, charge, mult, dataset=dataset, datapath=datapath)
            for atom, charge, mult in zip(atoms, charges, mults)
        ]
        # Set coordinates
        self.coords = np.array(coords)
        # Set coefficients
        if coeffs is None:
            self.coeffs = np.ones(len(atoms), dtype=float)
        else:
            self.coeffs = np.array(coeffs, dtype=float)

    def density(self, points, spin='ab', log=False):
        r"""
        TODO: what do we do with the "index" parameter of the atom density spline functions?

        """
        # Define the property as a function, and call `_extensive_global_property` on it
        f = lambda atom, radii: atom.dens_spline(radii, spin=spin, log=log)
        return _extensive_local_property(self.atoms, self.coords, self.coeffs, points, f)

    def ked(self, points, spin='ab', log=False):
        r"""
        TODO: what do we do with the "index" parameter of the atom kinetic energy spline functions?

        """
        f = lambda atom, radii: atom.ked_spline(radii, spin=spin, log=log)
        return _extensive_local_property(self.atoms, self.coords, self.coeffs, points, f)

    def energy(self):
        f = lambda atom: atom.energy
        return _extensive_global_property(self.atoms, self.coeffs, f)

    def mass(self):
        f = lambda atom: atom.mass
        return _extensive_global_property(self.atoms, self.coeffs, f)

    def ip(self, p=1):
        r"""
        TODO: is there even a point to using the coefficients here? I included them...

        """
        # Define the property as a function, and call `_intensive_property` on it
        f = lambda atom: atom.ip
        return _intensive_property(self.atoms, self.coeffs, f, p=p)

    def mu(self, p=1):
        r"""
        TODO: is there even a point to using the coefficients here? I included them...

        """
        # Define the property as a function, and call `_intensive_property` on it
        f = lambda atom: atom.mu
        return _intensive_property(self.atoms, self.coeffs, f, p=p)

    def eta(self, p=1):
        r"""
        TODO: is there even a point to using the coefficients here? I included them...

        """
        # Define the property as a function, and call `_intensive_property` on it
        f = lambda atom: atom.eta
        return _intensive_property(self.atoms, self.coeffs, f, p=p)


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
    return (sum(coeff * f(atom) ** p for atom, coeff in zip(atoms, coeffs)) / sum(coeffs)) ** (1 / p)


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
    comments = " PROMOLECULE CUBE FILE\n  'OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z'\n"
    natom = len(atnums)
    nx, ny, nz = cb_shape
    _format = lambda scalar, vector: f"{scalar:5d}" + "".join(f"{v:11.6f}" for v in vector) + "\n"
    with open(fname, 'w') as cube:
        # Header section
        cube.write(comments)
        cube.write(_format(natom, cb_origin)) # 3rd line #atoms and origin
        for i in range(3):
            cube.write(_format(cb_shape[i], cb_axis[i])) # axis #voxels and vector
        for z, q, xyz in zip(atnums, charges, coords):
            qxyz = [q]+xyz.tolist()
            cube.write(_format(z, qxyz)) # atom#, charge and coordinates
        # Volumetric data
        vdata = vdata.reshape((nx,ny,nz))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cube.write(f' {vdata[ix, iy, iz]:12.5E}')
                    if (iz % 6 == 5):
                        cube.write("\n")
                cube.write("\n")
