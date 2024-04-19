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

from glob import glob

from importlib import import_module

import json

from numbers import Integral

from os import makedirs, path

from msgpack import packb, unpackb

from msgpack_numpy import encode, decode

import numpy as np

from numpy import ndarray

from scipy.interpolate import CubicSpline

from atomdb.utils import DEFAULT_DATASET, DEFAULT_DATAPATH
from atomdb.periodic import element_symbol


__all__ = [
    "Species",
    "compile",
    "dump",
    "load",
    "raw_datafile",
]


def default_required(name, typeof):
    r"""Default factory for required fields."""

    def f():
        raise KeyError(f"Field {name} of type {typeof} was not found")

    return f


def default_vector():
    r"""Default factory for 1-dimensional ``np.ndarray``."""

    return np.zeros(0).reshape(0)


def default_matrix():
    r"""Default factory for 2-dimensional ``np.ndarray``."""
    return np.zeros(0).reshape(1, 0)


def scalar(method):
    r"""Expose a SpeciesData field."""
    name = method.__name__

    @property
    def wrapper(self):
        rf"""{method.__doc__}"""
        return getattr(self._data, name)

    return wrapper


def _remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


def spline(method):
    r"""Expose a SpeciesData field via the ``DensitySpline`` interface."""
    name = _remove_suffix(method.__name__, "_func")

    def wrapper(self, spin="t", index=None, log=False):
        rf"""{method.__doc__}"""
        # Validate `spin` variable
        if spin not in ("t", "a", "b", "m"):
            raise ValueError(
                f"Invalid `spin` parameter '{spin}'; " "choose one of ('t'| 'a' | 'b' | 'm')"
            )

        # Set names for {a,b,tot} arrays
        name_tot = f"{name}_tot"
        if self.spinpol == 1:
            name_a, name_b = f"mo_{name}_a", f"mo_{name}_b"
        else:
            name_a, name_b = f"mo_{name}_b", f"mo_{name}_a"

        # Extract arrays
        if index is None and spin == "t":
            arr = getattr(self._data, name_tot)
        elif spin == "t":
            arr = getattr(self._data, name_a) + getattr(self._data, name_b)
            # FIXME: This is a hack to change the array to the correct 2D shape
            arr = arr.reshape(self.ao.nbasis, -1)
        elif spin == "a":
            arr = getattr(self._data, name_a)
            arr = arr.reshape(self.ao.nbasis, -1)
        elif spin == "b":
            arr = getattr(self._data, name_b)
            arr = arr.reshape(self.ao.nbasis, -1)
        elif spin == "m":
            arr = getattr(self._data, name_a) - getattr(self._data, name_b)
            arr = arr.reshape(self.ao.nbasis, -1)
        # Select specific orbitals
        if index is not None:
            arr = arr[index]
        # Colapse the orbital dimension to get total density values
        if arr.ndim > 1:
            arr = arr.sum(axis=0)  # (N,)

        # Return cubic spline
        return DensitySpline(self._data.rs, arr, log=log)

    return wrapper


class DensitySpline:
    r"""Interpolate density using a cubic spline over a 1-D grid."""

    def __init__(self, x, y, log=False):
        r"""Initialize the CubicSpline instance."""
        self._log = log
        self._obj = CubicSpline(
            x,
            np.log(y) if log else y,
            axis=0,
            bc_type="not-a-knot",
            extrapolate=True,
        )

    def __call__(self, x, deriv=0):
        r"""
        Compute the interpolation at some x-values.

        Parameters
        ----------
        x: ndarray(M,)
            Points to be interpolated.
        deriv: int, default=0
            Order of spline derivative to evaluate. Must be 0, 1, or 2.

        Returns
        -------
        ndarray(M,)
            Interpolated values (1-D array).

        """
        if not (0 <= deriv <= 2):
            raise ValueError(f"Invalid derivative order {deriv}; must be 0 <= `deriv` <= 2")
        elif self._log:
            y = np.exp(self._obj(x))
            if deriv == 1:
                # d(ρ(r)) = d(log(ρ(r))) * ρ(r)
                dlogy = self._obj(x, nu=1)
                y = dlogy.flatten() * y
            elif deriv == 2:
                # d^2(ρ(r)) = d^2(log(ρ(r))) * ρ(r) + [d(ρ(r))]^2/ρ(r)
                dlogy = self._obj(x, nu=1)
                d2logy = self._obj(x, nu=2)
                y = d2logy.flatten() * y + dlogy.flatten() ** 2 * y
        else:
            y = self._obj(x, nu=deriv)
        return y


class JSONEncoder(json.JSONEncoder):
    r"""JSON encoder handling simple `numpy.ndarray` objects."""

    def default(self, obj):
        r"""Default encode function."""
        if isinstance(obj, ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)


class _AtomicOrbitals(object):
    """Atomic orbitals class."""

    def __init__(self, data) -> None:
        self.occs_a = data.mo_occs_a
        self.occs_b = data.mo_occs_b
        self.energy_a = data.mo_energy_a
        self.energy_b = data.mo_energy_b
        self.norba = len(self.energy_a) if self.energy_a is not None else None
        self.norbb = len(self.energy_b) if self.energy_a is not None else None
        self.nbasis = self.norba  # number of spatial basis functions


@dataclass(eq=False, order=False)
class SpeciesData:
    r"""Database entry fields for atomic and ionic species."""

    # Species info
    elem: str = field(default_factory=default_required("elem", "str"))
    atnum: int = field(default_factory=default_required("atnum", "int"))
    nelec: int = field(default_factory=default_required("nelec", "int"))
    nspin: int = field(default_factory=default_required("nspin", "int"))
    nexc: int = field(default_factory=default_required("nexc", "int"))

    # Scalar properties
    atmass: float = field(default=None)
    cov_radius: float = field(default=None)
    vdw_radius: float = field(default=None)
    at_radius: float = field(default=None)
    polarizability: float = field(default=None)
    dispersion: float = field(default=None)

    # Scalar energy and CDFT-related properties
    energy: float = field(default=None)
    ip: float = field(default=None)
    mu: float = field(default=None)
    eta: float = field(default=None)

    # Basis set name
    obasis_name: str = field(default=None)

    # Radial grid
    rs: ndarray = field(default_factory=default_vector)

    # Orbital energies
    mo_energy_a: ndarray = field(default_factory=default_vector)
    mo_energy_b: ndarray = field(default_factory=default_vector)

    # Orbital occupations
    mo_occs_a: ndarray = field(default_factory=default_vector)
    mo_occs_b: ndarray = field(default_factory=default_vector)

    # Orbital densities
    mo_dens_a: ndarray = field(default_factory=default_matrix)
    mo_dens_b: ndarray = field(default_factory=default_matrix)
    dens_tot: ndarray = field(default_factory=default_matrix)

    # Orbital density gradients
    mo_d_dens_a: ndarray = field(default_factory=default_matrix)
    mo_d_dens_b: ndarray = field(default_factory=default_matrix)
    d_dens_tot: ndarray = field(default_factory=default_matrix)

    # Orbital density Laplacian
    mo_dd_dens_a: ndarray = field(default_factory=default_matrix)
    mo_dd_dens_b: ndarray = field(default_factory=default_matrix)
    dd_dens_tot: ndarray = field(default_factory=default_matrix)

    # Orbital kinetic energy densities
    mo_ked_a: ndarray = field(default_factory=default_matrix)
    mo_ked_b: ndarray = field(default_factory=default_matrix)
    ked_tot: ndarray = field(default_factory=default_matrix)


class Species:
    r"""Properties of atomic and ionic species."""

    def __init__(self, dataset, fields, spinpol=1):
        r"""Initialize a ``Species`` instance."""
        self._dataset = dataset.lower()
        self._data = SpeciesData(**fields)
        self.spinpol = spinpol
        self.ao = _AtomicOrbitals(self._data)

    def get_docstring(self):
        r"""Docstring of the species' dataset."""
        return import_module(f"atomdb.datasets.{self._dataset}").__doc__

    def to_dict(self):
        r"""Return the dictionary representation of the Species instance."""
        return asdict(self._data)

    def to_json(self):
        r"""Return the JSON string representation of the Species instance."""
        return json.dumps(asdict(self._data), cls=JSONEncoder)

    @property
    def dataset(self):
        r"""Dataset."""
        return self._dataset

    @property
    def charge(self):
        r"""Charge."""
        return self._data.atnum - self._data.nelec

    @property
    def nspin(self):
        r"""Spin number :math:`N_S = N_α - N_β`."""
        return self._data.nspin * self._spinpol

    @property
    def mult(self):
        r"""Multiplicity :math:`M = \left|N_S\right| + 1`."""
        return self._data.nspin + 1

    @property
    def spinpol(self):
        r"""Spin polarization direction (±1) of the species."""
        return self._spinpol

    @spinpol.setter
    def spinpol(self, spinpol):
        r"""Spin polarization direction setter."""
        if not isinstance(spinpol, Integral):
            raise TypeError("`spinpol` attribute must be an integral type")

        spinpol = int(spinpol)

        if abs(spinpol) != 1:
            raise ValueError("`spinpol` must be +1 or -1")

        self._spinpol = spinpol

    @scalar
    def elem(self):
        r"""Element symbol."""
        pass

    @scalar
    def obasis_name(self):
        r"""Basis name."""
        pass

    @scalar
    def atnum(self):
        r"""Atomic number."""
        pass

    @scalar
    def nelec(self):
        r"""Number of electrons."""
        pass

    @scalar
    def atmass(self):
        r"""Atomic mass in atomic units.
        
        Returns
        -------
        mass : dict
            Two options are available: the isotopically averaged mass 'stb', and the mass of the most
            common isotope 'nist'.

        """
        pass

    @scalar
    def cov_radius(self):
        r"""Covalent radius (derived from crystallographic data)."""
        pass

    @scalar
    def vdw_radius(self):
        r"""Van der Waals radii."""
        pass

    @scalar
    def at_radius(self):
        r"""Atomic radius."""
        pass

    @scalar
    def polarizability(self):
        r"""Isolated atom dipole polarizability."""
        pass

    @property
    def dispersion_c6(self):
        r"""Isolated atom C6 dispersion coefficients."""
        if self._data.dispersion is None:
            return None
        return self._data.dispersion["C6"]

    @scalar
    def nexc(self):
        r"""Excitation number."""
        pass

    @scalar
    def energy(self):
        r"""Energy."""
        pass

    @scalar
    def ip(self):
        r"""Ionization potential."""
        pass

    @scalar
    def mu(self):
        r"""Chemical potential."""
        pass

    @scalar
    def eta(self):
        r"""Chemical hardness."""
        pass

    @spline
    def dens_func(self):
        r"""
        Return a cubic spline of the electronic density.

        Parameters
        ----------
        spin : str, default="t"
            Type of occupied spin orbitals.
            Can be either "t" (for alpha + beta), "a" (for alpha),
            "b" (for beta), or "m" (for alpha - beta).
        index : sequence of int, optional
            Sequence of integers representing the spin orbitals.
            These are indexed from 0 to the number of basis functions.
            By default, all orbitals of the given spin(s) are included.
        log : bool, default=False
            Whether the logarithm of the density is used for interpolation.

        Returns
        -------
        DensitySpline
            A DensitySpline instance for the density and its derivatives.
            Given a set of radial points, it can evaluate densities and
            derivatives up to order 2.

        """
        pass

    @spline
    def d_dens_func(self):
        """
        Return a cubic spline of the first derivative of the electronic density.

        The derivarive of the density as a function of the distance to the atomic center
        (a set of points along a 1-D grid) is modeled by a cubic spline. The property can
        be computed for the alpha, beta, alpha + beta, and alpha - beta components of the
        electron density.

        Parameters
        ----------
        spin : str, optional
            Type of occupied spin orbitals which can be either "a" (for alpha), "b" (for
            beta), "ab" (for alpha + beta), and, "m" (for alpha - beta), by default 'ab'
        index : sequence of int, optional
            Sequence of integers representing the spin orbitals which are indexed
            from 1 to the number basis functions. If ``None``, all orbitals of the given
            spin(s) are included
        log : bool, optional
            Whether the logarithm of the density property is used for interpolation

        Returns
        -------
        Callable[[np.ndarray(N,), int] -> np.ndarray(N,)]
            a callable function evaluating the derivative of the density given a set of radial
            points (1-D array).

        """
        pass

    @spline
    def dd_dens_func(self):
        r"""
        Return a cubic spline of the electronic density Laplacian.

        Parameters
        ----------
        spin : str, default="t"
            Type of occupied spin orbitals.
            Can be either "t" (for alpha + beta), "a" (for alpha),
            "b" (for beta), or "m" (for alpha - beta).
        index : sequence of int, optional
            Sequence of integers representing the spin orbitals.
            These are indexed from 0 to the number of basis functions.
            By default, all orbitals of the given spin(s) are included.
        log : bool, default=False
            Whether the logarithm of the density is used for interpolation.

        Returns
        -------
        DensitySpline
            A DensitySpline instance for the density and its derivatives.
            Given a set of radial points, it can evaluate densities and
            derivatives up to order 2.

        """
        pass

    @spline
    def ked_func(self):
        r"""
        Return a cubic spline of the kinetic energy density.

        Parameters
        ----------
        spin : str, default="t"
            Type of occupied spin orbitals.
            Can be either "t" (for alpha + beta), "a" (for alpha),
            "b" (for beta), or "m" (for alpha - beta).
        index : sequence of int, optional
            Sequence of integers representing the spin orbitals.
            These are indexed from 0 to the number of basis functions.
            By default, all orbitals of the given spin(s) are included.
        log : bool, default=False
            Whether the logarithm of the density is used for interpolation.

        Returns
        -------
        DensitySpline
            A DensitySpline instance for the density and its derivatives.
            Given a set of radial points, it can evaluate densities and
            derivatives up to order 2.

        """
        pass


def compile(
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""Compile an atomic or ionic species into the AtomDB database."""
    # Ensure directories exist
    makedirs(path.join(datapath, dataset.lower(), "db"), exist_ok=True)
    makedirs(path.join(datapath, dataset.lower(), "raw"), exist_ok=True)
    # Import the compile script for the appropriate dataset
    submodule = import_module(f"atomdb.datasets.{dataset}.run")
    # Compile the Species instance and dump the database entry
    species = submodule.run(elem, charge, mult, nexc, dataset, datapath)
    dump(species, datapath=datapath)


def dump(*species, datapath=DEFAULT_DATAPATH):
    r"""Dump the Species instance(s) to a MessagePack file in the database."""
    for s in species:
        fn = datafile(
            s._data.elem, s.charge, s.mult, nexc=s.nexc, dataset=s.dataset, datapath=datapath
        )
        with open(fn, "wb") as f:
            f.write(packb(asdict(s._data), default=encode))


def load(
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""Load one or many atomic or ionic species from the AtomDB database."""
    fn = datafile(
        elem,
        charge,
        mult,
        nexc=nexc,
        dataset=dataset,
        datapath=datapath,
    )
    if Ellipsis in (elem, charge, mult, nexc):
        obj = []
        for file in glob(fn):
            with open(file, "rb") as f:
                obj.append(Species(dataset, unpackb(f.read(), object_hook=decode)))
    else:
        with open(fn, "rb") as f:
            obj = Species(dataset, unpackb(f.read(), object_hook=decode))
    return obj


def datafile(
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""Return the name of the database file for a species."""
    elem = "*" if elem is Ellipsis else element_symbol(elem)
    charge = "*" if charge is Ellipsis else f"{charge:03d}"
    mult = "*" if mult is Ellipsis else f"{mult:03d}"
    nexc = "*" if nexc is Ellipsis else f"{nexc:03d}"
    return path.join(datapath, dataset.lower(), "db", f"{elem}_{charge}_{mult}_{nexc}.msg")


def raw_datafile(
    suffix,
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""Return the name of the database file for a species."""
    elem = "*" if elem is Ellipsis else element_symbol(elem)
    charge = "*" if charge is Ellipsis else f"{charge:03d}"
    mult = "*" if mult is Ellipsis else f"{mult:03d}"
    nexc = "*" if nexc is Ellipsis else f"{nexc:03d}"
    return path.join(datapath, dataset.lower(), "raw", f"{elem}_{charge}_{mult}_{nexc}{suffix}")
