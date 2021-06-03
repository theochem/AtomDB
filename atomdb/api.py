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

r"""AtomDB API definitions."""


from dataclasses import dataclass, asdict
from importlib import import_module

import numpy as np

from os import makedirs

from atomdb.config import DEFAULT_DATASET, DATAPATH
from atomdb.utils import cubic_interp, get_element_symbol, get_element_number,
from atomdb.utils import get_data_file, get_file, pack_msg, unpack_msg, ndarray_to_bytes


__all__ = [
    "generate_species",
    "compile_species",
    "load_species",
    "dump_species",
    "build_atomdb",
]


class Species:
    r"""Properties of atomic and ionic species."""

    @dataclass
    class SpeciesData:
        r"""Contain class for species properties."""
        #
        # Species info
        #
        species: str
        natom: int
        basis: str
        nelec: int
        nspin: int
        #
        # Electronic and molecular orbital energies
        #
        energy_hf: float
        energy_ci: float
        mo_energies: np.ndarray
        mo_occs: np.ndarray
        #
        # Radial grid
        #
        rs: np.ndarray
        #
        # Density
        #
        dens_up: np.ndarray
        dens_dn: np.ndarray
        dens_tot: np.ndarray
        dens_mag: np.ndarray
        #
        # Derivative of density
        #
        d_dens_up: np.ndarray
        d_dens_dn: np.ndarray
        d_dens_tot: np.ndarray
        d_dens_mag: np.ndarray
        #
        # Laplacian
        #
        lapl_up: np.ndarray
        lapl_dn: np.ndarray
        lapl_tot: np.ndarray
        lapl_mag: np.ndarray
        #
        # Kinetic energy density
        #
        ked_up: np.ndarray
        ked_dn: np.ndarray
        ked_tot: np.ndarray
        ked_mag: np.ndarray

    def __getattr__(self, name):
        r"""Find an attribute from the SpeciesData instance `self._data`."""
        return getattr(self._data, name)

    def __init__(self, *args, **kwargs):
        r"""Initialize a Species instance."""
        self._data = Species.SpeciesData(*args, **kwargs)
        self.dens_up_spline = cubic_interp(self._data.rs, self._data.dens_up, log=False)
        self.dens_dn_spline = cubic_interp(self._data.rs, self._data.dens_dn, log=False)
        self.dens_tot_spline = cubic_interp(self._data.rs, self._data.dens_tot, log=False)
        self.dens_mag_spline = cubic_interp(self._data.rs, self._data.dens_mag, log=False)
        self.d_dens_up_spline = cubic_interp(self._data.rs, self._data.d_dens_up, log=False)
        self.d_dens_dn_spline = cubic_interp(self._data.rs, self._data.d_dens_dn, log=False)
        self.d_dens_tot_spline = cubic_interp(self._data.rs, self._data.d_dens_tot, log=False)
        self.d_dens_mag_spline = cubic_interp(self._data.rs, self._data.d_dens_mag, log=False)
        self.ked_up_spline = cubic_interp(self._data.rs, self._data.ked_up, log=True)
        self.ked_dn_spline = cubic_interp(self._data.rs, self._data.ked_dn, log=True)
        self.ked_tot_spline = cubic_interp(self._data.rs, self._data.ked_tot, log=True)
        self.ked_mag_spline = cubic_interp(self._data.rs, self._data.ked_mag, log=True)
        self.lapl_up_spline = cubic_interp(self._data.rs, self._data.lapl_up, log=False)
        self.lapl_dn_spline = cubic_interp(self._data.rs, self._data.lapl_dn, log=False)
        self.lapl_tot_spline = cubic_interp(self._data.rs, self._data.lapl_tot, log=False)
        self.lapl_mag_spline = cubic_interp(self._data.rs, self._data.lapl_mag, log=False)

    def to_dict(self):
        r"""Convert a Species instance to a dictionary."""
        return asdict(self._data)


# TODO: check argument consistency
def generate_species(dataset=DEFAULT_DATASET, **kwargs):
    r"""Generate the raw data for a species in the given dataset."""
    return import_module(f'atomdb.datasets.{dataset}.generate').generate_species(**kwargs)


# TODO: check argument consistency
def compile_species(dataset=DEFAULT_DATASET, **kwargs):
    r"""Compile a species database entry from the given dataset's raw data."""
    return import_module(f'atomdb.datasets.{dataset}.compile').compile_species(**kwargs)


# TODO: Instead of fn, arguments should be (dataset, charge, multiplicity, ...).
def load_species(dataset, elem, nelec, nspin, nexc, **kwargs):
    r"""Load an atomic or ionic species from the AtomDB database."""
    fn = get_data_file(dataset, elem, nelec, nspin, nexc, "msg")
    with open(fn, "rb") as f:
        species_dict = unpack_msg(f)

    species_dict = {
    key: np.frombuffer(val) if isinstance(val, bytes) else val
    for key, val in species_dict.items()
    }
    return Species(**species_dict)


# TODO: Instead of fn, arguments should be (dataset, charge, multiplicity, ...).
def dump_species(dataset, elem, nelec, nspin, nexc, species_dict, **kwargs):
    r""" """
    makedirs(get_file(dataset), exist_ok=True)
    fn = get_data_file(dataset, elem, nelec, nspin, nexc, "msg")
    species_dict = {
            key: ndarray_to_bytes(val) if isinstance(val, np.ndarray) else val
            for key, val in species_dict.items()
        }
    with open(fn, "wb") as f:
        f.write(pack_msg(species_dict, **kwargs))


# TODO: check argument consistency (and write the code ;( )
def build_atomdb(*args, **kwargs):
    r"""Build the AtomDB database."""
    pass
