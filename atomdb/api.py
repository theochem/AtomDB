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


from dataclasses import dataclass, field, asdict

from importlib import import_module

from os import makedirs

from numpy import ndarray, frombuffer

from atomdb.config import *

from atomdb.utils import *


__all__ = [
    "Species",
    "generate_species",
    "compile_species",
    "load_species",
    "dump_species",
]


@dataclass(eq=False, order=False)
class Species:
    r"""Properties of atomic and ionic species."""
    #
    # Species info
    #
    dataset: str = field()
    species: str = field()
    natom: int = field()
    basis: str = field()
    nelec: int = field()
    nspin: int = field()
    nexc: int = field()
    #
    # Electronic and molecular orbital energies
    #
    energy_hf: float = field()
    energy_ci: float = field()
    mo_energies: ndarray = field()
    mo_occs: ndarray = field()
    #
    # Radial grid
    #
    rs: ndarray = field()
    #
    # Density
    #
    dens_up: ndarray = field()
    dens_dn: ndarray = field()
    dens_tot: ndarray = field()
    dens_mag: ndarray = field()
    #
    # Derivative of density
    #
    d_dens_up: ndarray = field()
    d_dens_dn: ndarray = field()
    d_dens_tot: ndarray = field()
    d_dens_mag: ndarray = field()
    #
    # Laplacian
    #
    lapl_up: ndarray = field()
    lapl_dn: ndarray = field()
    lapl_tot: ndarray = field()
    lapl_mag: ndarray = field()
    #
    # Kinetic energy density
    #
    ked_up: ndarray = field()
    ked_dn: ndarray = field()
    ked_tot: ndarray = field()
    ked_mag: ndarray = field()
    #
    # Density splines
    #
    dens_up_spline: interp1d = field(init=False, repr=False)
    dens_dn_spline: interp1d = field(init=False, repr=False)
    dens_tot_spline: interp1d = field(init=False, repr=False)
    dens_mag_spline: interp1d = field(init=False, repr=False)
    d_dens_up_spline: interp1d = field(init=False, repr=False)
    d_dens_dn_spline: interp1d = field(init=False, repr=False)
    d_dens_tot_spline: interp1d = field(init=False, repr=False)
    d_dens_mag_spline: interp1d = field(init=False, repr=False)
    ked_up_spline: interp1d = field(init=False, repr=False)
    ked_dn_spline: interp1d = field(init=False, repr=False)
    ked_tot_spline: interp1d = field(init=False, repr=False)
    ked_mag_spline: interp1d = field(init=False, repr=False)
    lapl_up_spline: interp1d = field(init=False, repr=False)
    lapl_dn_spline: interp1d = field(init=False, repr=False)
    lapl_tot_spline: interp1d = field(init=False, repr=False)
    lapl_mag_spline: interp1d = field(init=False, repr=False)

    @property
    def charge(self):
        r"""Charge of the species."""
        return self.natom - self.nelec

    @property
    def mult(self):
        r"""Multiplicity of the species."""
        return self.nspin + 1

    def __post_init__(self):
        r"""Construct splines from the Species Data."""
        self.dens_up_spline = cubic_interp(self.rs, self.dens_up, log=False)
        self.dens_dn_spline = cubic_interp(self.rs, self.dens_dn, log=False)
        self.dens_tot_spline = cubic_interp(self.rs, self.dens_tot, log=False)
        self.dens_mag_spline = cubic_interp(self.rs, self.dens_mag, log=False)
        self.d_dens_up_spline = cubic_interp(self.rs, self.d_dens_up, log=False)
        self.d_dens_dn_spline = cubic_interp(self.rs, self.d_dens_dn, log=False)
        self.d_dens_tot_spline = cubic_interp(self.rs, self.d_dens_tot, log=False)
        self.d_dens_mag_spline = cubic_interp(self.rs, self.d_dens_mag, log=False)
        self.ked_up_spline = cubic_interp(self.rs, self.ked_up, log=True)
        self.ked_dn_spline = cubic_interp(self.rs, self.ked_dn, log=True)
        self.ked_tot_spline = cubic_interp(self.rs, self.ked_tot, log=True)
        self.ked_mag_spline = cubic_interp(self.rs, self.ked_mag, log=True)
        self.lapl_up_spline = cubic_interp(self.rs, self.lapl_up, log=False)
        self.lapl_dn_spline = cubic_interp(self.rs, self.lapl_dn, log=False)
        self.lapl_tot_spline = cubic_interp(self.rs, self.lapl_tot, log=False)
        self.lapl_mag_spline = cubic_interp(self.rs, self.lapl_mag, log=False)


def generate_species(element, charge, mult, nexc=0, dataset=DEFAULT_DATASET):
    r"""Generate the raw data for a species in the given dataset."""
    return import_module(f'atomdb.datasets.{dataset}.generate').generate_species(
        element, charge, mult, nexc=nexc, dataset=dataset,
    )


def compile_species(element, charge, mult, nexc=0, dataset=DEFAULT_DATASET, bound=(0.01, 0.5), num=100):
    r"""Compile a species database entry from the given dataset's raw data."""
    return import_module(f'atomdb.datasets.{dataset}.compile').compile_species(
        element, charge, mult, nexc=nexc, dataset=dataset, bound=bound, num=num,
    )


def load_species(element, charge, mult, nexc=0, dataset=DEFAULT_DATASET):
    r"""Load an atomic or ionic species from the AtomDB database."""
    elem = get_element_symbol(element)
    nelec = get_element_number(elem) - charge
    nspin = mult - 1
    with open(get_data_file(dataset, elem, nelec, nspin, nexc, "msg"), "rb") as f:
        species_dict = unpack_msg(f)
    for key, val in species_dict.items():
        if isinstance(val, bytes):
            species_dict[key] = frombuffer(val)
    return Species(**species_dict)


def dump_species(species_obj, dataset):
    r"""Dump an atomic or ionic species to the AtomDB database."""
    makedirs(get_file(dataset), exist_ok=True)
    fn = get_data_file(
        dataset, species_obj.species, species_obj.nelec, species_obj.nspin, species_obj.nexc, "msg",
    )
    species_dict = {
        key: ndarray_to_bytes(val) if isinstance(val, ndarray) else val
        for key, val in asdict(species_obj).items() if not key.endswith('_spline')
    }
    with open(fn, "wb") as f:
        f.write(pack_msg(species_dict))
