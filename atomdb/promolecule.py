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

    def __init__(self, atoms, coords, charges, mults, dataset=DEFAULT_DATASET, datapath=DEFAULT_DATAPATH):
        r"""
        TODO: have a table of default charges/mults and make them optional inputs.

        TODO: have coefficients besides the default "1".

        """
        self.atoms = [
            load(atom, charge, mult, dataset=dataset, datapath=datapath)
            for atom, charge, mult in zip(atoms, charges, mults)
        ]
        self.coords = np.array(coords)
        self.coeffs = np.ones(len(atoms))

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
    # Initialize property to zeros
    prop = np.zeros(len(points))
    # Add contribution from each atom
    for (atom, coord, coeff) in zip(atoms, atom_coords, coeffs):
        # Get radius between points of interest and atom
        radii = np.linalg.norm(points - coord, axis=1)
        # Compute property at the proper radiii
        prop += coeff * f(atom, radii)
    return prop


def _intensive_property(atoms, coeffs, f, p=1):
    r"""Helper function for computing intensive properties."""
    # P-mean of each atom's property value
    return (sum(coeff * f(atom) ** p for atom, coeff in zip(atoms, coeffs)) / len(atoms)) ** (1 / p)
