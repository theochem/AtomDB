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

r"""AtomDB, a database of atomic and ionic properties."""


from dataclasses import dataclass, field, asdict

from importlib import import_module

from json import JSONEncoder, dumps

from os import environ, makedirs

from os.path import dirname, join

from sys import platform

from msgpack import Packer, Unpacker

from numpy import ndarray, frombuffer, exp, log

from scipy.interpolate import interp1d

from csv import DictReader

from itertools import islice

from .units import angstrom, amu


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_DATAPATH",
    "Species",
    "load",
    "compile",
    "datafile",
    "element_number",
    "element_symbol",
    "get_element_data",
]


DEFAULT_DATASET = "hci"
r"""Default dataset to query."""


DEFAULT_DATAPATH = environ.get("ATOMDB_DATAPATH", join(dirname(__file__), "datasets/"))
r"""The path for raw and compiled AtomDB data files."""


ELEMENTS = (
    # fmt: off
    "\0", "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    # fmt: on
)
r"""Tuple of the symbols for each of the 118 elements. The zeroth element is a placeholder."""


# The correct way to convert numpy arrays to bytes is different on Mac/"Darwin"
if platform == "darwin":
    # Mac
    def _array_to_bytes(array):
        r"""Convert a numpy.ndarray instance to bytes."""
        return array.tobytes()


else:
    # Linux and friends
    def _array_to_bytes(array):
        r"""Convert a numpy.ndarray instance to bytes."""
        return array.data if array.flags["C_CONTIGUOUS"] else array.tobytes()


@dataclass(eq=False, order=False)
class SpeciesData():
    r"""Properties of atomic and ionic species corresponding to fields in MessagePack files."""
    #
    # Species info
    #
    dataset: str = field()
    elem: str = field()
    natom: int = field()
    basis: str = field()
    nelec: int = field()
    nspin: int = field()
    nexc: int = field()
    #
    # Element properties
    #
    cov_radii: dict = field()
    vdw_radii: dict = field()
    mass: float = field()
    #
    # Electronic and molecular orbital energies
    #
    energy: float = field(default=None)
    mo_energies: ndarray = field(default=None)
    mo_occs: ndarray = field(default=None)
    #
    # Radial grid
    #
    rs: ndarray = field(default=None)
    #
    # Density
    #
    dens_up: ndarray = field(default=None)
    dens_dn: ndarray = field(default=None)
    dens_tot: ndarray = field(default=None)
    dens_mag: ndarray = field(default=None)
    #
    # Derivative of density
    #
    d_dens_up: ndarray = field(default=None)
    d_dens_dn: ndarray = field(default=None)
    d_dens_tot: ndarray = field(default=None)
    d_dens_mag: ndarray = field(default=None)
    #
    # Laplacian
    #
    lapl_up: ndarray = field(default=None)
    lapl_dn: ndarray = field(default=None)
    lapl_tot: ndarray = field(default=None)
    lapl_mag: ndarray = field(default=None)
    #
    # Kinetic energy density
    #
    ked_up: ndarray = field(default=None)
    ked_dn: ndarray = field(default=None)
    ked_tot: ndarray = field(default=None)
    ked_mag: ndarray = field(default=None)


class Species(SpeciesData):
    r"""Properties of atomic and ionic species."""

    def __init__(self, *args, **kwargs):
        r"""Initialize a Species Instance."""
        # Initialize superclass
        SpeciesData.__init__(self, *args, **kwargs)
        #
        # Attributes declared here are not considered as part of the dataclasses interface,
        # and therefore are not included in the output of dataclasses.asdict(species_instance)
        #
        # Charge and multiplicity
        #
        self.charge = self.natom - self.nelec
        self.mult = self.nspin + 1
        #
        # Density splines
        #
        if self.dens_up is None:
            self.dens_up_spline = cubic_interp(self.rs, self.dens_up, log=False)
        if self.dens_dn is not None:
            self.dens_dn_spline = cubic_interp(self.rs, self.dens_dn, log=False)
        if self.dens_tot is not None:
            self.dens_tot_spline = cubic_interp(self.rs, self.dens_tot, log=False)
        if self.dens_mag is not None:
            self.dens_mag_spline = cubic_interp(self.rs, self.dens_mag, log=False)
        if self.d_dens_up is not None:
            self.d_dens_up_spline = cubic_interp(self.rs, self.d_dens_up, log=False)
        if self.d_dens_dn is not None:
            self.d_dens_dn_spline = cubic_interp(self.rs, self.d_dens_dn, log=False)
        if self.d_dens_tot is not None:
            self.d_dens_tot_spline = cubic_interp(self.rs, self.d_dens_tot, log=False)
        if self.d_dens_mag is not None:
            self.d_dens_mag_spline = cubic_interp(self.rs, self.d_dens_mag, log=False)
        if self.ked_up is not None:
            self.ked_up_spline = cubic_interp(self.rs, self.ked_up, log=True)
        if self.ked_dn is not None:
            self.ked_dn_spline = cubic_interp(self.rs, self.ked_dn, log=True)
        if self.ked_tot is not None:
            self.ked_tot_spline = cubic_interp(self.rs, self.ked_tot, log=True)
        if self.ked_mag is not None:
            self.ked_mag_spline = cubic_interp(self.rs, self.ked_mag, log=True)
        if self.lapl_up is not None:
            self.lapl_up_spline = cubic_interp(self.rs, self.lapl_up, log=False)
        if self.lapl_dn is not None:
            self.lapl_dn_spline = cubic_interp(self.rs, self.lapl_dn, log=False)
        if self.lapl_tot is not None:
            self.lapl_tot_spline = cubic_interp(self.rs, self.lapl_tot, log=False)
        if self.lapl_mag is not None:
            self.lapl_mag_spline = cubic_interp(self.rs, self.lapl_mag, log=False)

    def to_dict(self):
        r"""Return the dictionary representation of the Species instance."""
        return asdict(self)

    def to_json(self):
        r"""Return the JSON string representation of the Species instance."""
        return dumps(asdict(self), cls=Species._JSONEncoder)

    class _JSONEncoder(JSONEncoder):
        r"""JSON encoder handling simple `numpy.ndarray` objects (for `Species.dump`)."""

        def default(self, obj):
            r"""Default encode function."""
            return obj.tolist() if isinstance(obj, ndarray) else JSONEncoder.default(self, obj)

    @staticmethod
    def _msgfile(elem, basis, charge, mult, nexc, dataset, datapath):
        r"""Return the filename of a database entry MessagePack file."""
        return join(datapath, f"{dataset.lower()}/db/{elem}_{basis.lower()}_{charge}_{mult}_{nexc}.msg")

    def _dump(self, datapath):
        r"""Dump the Species instance to a MessagePack file in the database."""
        # Get database entry filename
        fn = Species._msgfile(self.elem, self.basis, self.charge, self.mult, self.nexc, self.dataset, datapath)
        # Convert numpy arrays to raw bytes for dumping as msgpack
        msg = {k: _array_to_bytes(v) if isinstance(v, ndarray) else v for k, v in asdict(self).items()}
        # Dump msgpack entry to database
        with open(fn, "wb") as f:
            f.write(pack_msg(msg))


def load(elem, basis, charge, mult, nexc=0, dataset=DEFAULT_DATASET, datapath=DEFAULT_DATAPATH):
    r"""Load an atomic or ionic species from the AtomDB database."""
    # Load database msgpack entry
    with open(Species._msgfile(elem, basis, charge, mult, nexc, dataset, datapath), "rb") as f:
        msg = unpack_msg(f)
    # Convert raw bytes back to numpy arrays, initialize the Species instance, return it
    return Species(**{k: frombuffer(v) if isinstance(v, bytes) else v for k, v in msg.items()})


def compile(elem, basis, charge, mult, nexc=0, dataset=DEFAULT_DATASET, datapath=DEFAULT_DATAPATH):
    r"""Compile an atomic or ionic species into the AtomDB database."""
    # Ensure directories exist
    makedirs(join(datapath, f"{dataset}/db"), exist_ok=True)
    makedirs(join(datapath, f"{dataset}/raw"), exist_ok=True)
    # Import the compile script for the appropriate dataset
    submodule = import_module(f"atomdb.datasets.{dataset}")
    # Compile the Species instance and dump the database entry
    # species = submodule.run(elem, basis, charge, mult, nexc, dataset, datapath)._dump(datapath)
    submodule.run(elem, basis, charge, mult, nexc, dataset, datapath)._dump(datapath)


def datafile(suffix, elem, basis, charge, mult, nexc=0, dataset=None, datapath=DEFAULT_DATAPATH):
    r"""Return the filename of a raw data file."""
    # Check that all non-optional arguments are specified
    if dataset is None:
        raise ValueError("Argument `dataset` cannot be unspecified")
    # Format the filename specified and return it
    suffix = f"{'' if suffix.startswith('.') else '_'}{suffix.lower()}"
    return join(datapath, f"{dataset.lower()}/raw/{elem}_{basis.lower()}_{charge}_{mult}_{nexc}{suffix}")


def element_number(elem):
    r"""Return the element number from the given element symbol."""
    return ELEMENTS.index(elem) if isinstance(elem, str) else elem


def element_symbol(elem):
    r"""Return the element symbol from the given element number."""
    return elem if isinstance(elem, str) else ELEMENTS[elem]


def pack_msg(msg):
    r"""Pack an object to MessagePack binary format."""
    return Packer(use_bin_type=True).pack(msg)


def unpack_msg(msg):
    r"""Unpack an object from MessagePack binary format."""
    return Unpacker(msg, use_list=False, strict_map_key=True).unpack()


class interp1d_log(interp1d):
    r"""Interpolate over a 1-D grid."""

    def __init__(self, x, y, **kwargs):
        r"""Initialize the interp1d_log instance."""
        interp1d.__init__(self, x, log(y), **kwargs)

    def __call__(self, x):
        r"""Compute the interpolation at some x-values."""
        return exp(interp1d.__call__(self, x))


def cubic_interp(x, y, log=False):
    r"""Create an interpolated cubic spline for the given data."""
    cls = interp1d_log if log else interp1d
    return cls(x, y, kind="cubic", copy=False, fill_value="extrapolate", assume_sorted=True)


def get_element_data(elem):
    r"""Get properties from elements.csv."""
    z = element_number(elem)
    convertors = {
        'angstrom': (lambda s: float(s)*angstrom),
        '2angstrom': (lambda s: float(s)*angstrom/2),
        'angstrom**3': (lambda s: float(s)*angstrom**3),
        'amu': (lambda s: float(s)*amu),
    }

    with open(join(dirname(__file__), "data/elements.csv"), "r") as f:
        # Skip information about data provenance
        data = list(DictReader(islice(f, 93, None)))
        # Parse properties
        units = data[0]
        cov_radii = {k.split("_")[-1]: v for k, v in data[z].items() if "cov_radius" in k}
        cov_radii = {k: float(v)*angstrom if v is not '' else None for k, v in cov_radii.items()}
        vdw_radii = {k: v for k, v in data[z].items() if "vdw_radius" in k}
        vdw_radii = {k.split("_")[-1]: convertors[units[k]](v) if v is not '' else None for k, v in vdw_radii.items()}
        mass = float(data[z]['mass']) * amu
        return cov_radii, vdw_radii, mass
       
