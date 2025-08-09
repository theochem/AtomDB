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

import json
import re
from dataclasses import asdict, dataclass, field
from importlib import import_module
from os import makedirs, path

import numpy as np
import pooch
import requests
from numpy import ndarray
from scipy.interpolate import CubicSpline

from atomdb.periodic_test import element_symbol_map, PROPERTY_NAME_MAP, get_scalar_data, ElementAttr
from atomdb.utils import DEFAULT_DATAPATH, DEFAULT_DATASET, DEFAULT_REMOTE
from importlib_resources import files
import tables as pt
from numbers import Integral

datasets_hdf5_file = files("atomdb.datasets").joinpath("datasets_data.h5")
DATASETS_H5FILE = pt.open_file(datasets_hdf5_file, mode="a")


__all__ = [
    "Species",
    "compile_species",
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
        # Checking if the property is not in PROPERTY_NAME_MAP, if not then fetch it from SpeciesData
        if name not in PROPERTY_NAME_MAP:
            return getattr(self._data, name)

        get_scalar_data(name, self._data.atnum, self._data.nelec)

    wrapper.__doc__ = method.__doc__
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

    # conserve the docstring of the method
    wrapper.__doc__ = method.__doc__
    return wrapper


class DensitySpline:
    r"""Interpolate density using a cubic spline over a 1-D grid."""

    def __init__(self, x, y, log=False):
        r"""Initialize the CubicSpline instance."""
        self._log = log
        self._obj = CubicSpline(
            x,
            # Clip y values to >= ε^2 if using log because they have to be above 0;
            # having them be at least ε^2 seems to work based on my testing
            np.log(y.clip(min=np.finfo(float).eps ** 2)) if log else y,
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
            # Get y = exp(log y). We'll handle errors from small log y values later.
            with np.errstate(over="ignore"):
                y = np.exp(self._obj(x))
            if deriv == 1:
                # d(ρ(r)) = d(log(ρ(r))) * ρ(r)
                dlogy = self._obj(x, nu=1)
                y = dlogy.flatten() * y
            elif deriv == 2:
                # d^2(ρ(r)) = d^2(log(ρ(r))) * ρ(r) + [d(ρ(r))]^2/ρ(r)
                dlogy = self._obj(x, nu=1)
                d2logy = self._obj(x, nu=2)
                y = d2logy.flatten() * y + dlogy.flatten() ** 2 / y
        else:
            y = self._obj(x, nu=deriv)
        # Handle errors from the y = exp(log y) operation -- set NaN to zero
        np.nan_to_num(y, nan=0.0, copy=False)
        # Cutoff value: assume y(x) is zero where x > final given point x_n
        y[x > self._obj.x[-1]] = 0
        return y


class JSONEncoder(json.JSONEncoder):
    r"""JSON encoder handling simple `numpy.ndarray` objects."""

    def default(self, obj):
        r"""Default encode function."""
        if isinstance(obj, ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)


class _AtomicOrbitals:
    """Atomic orbitals class."""

    def __init__(self, data) -> None:
        self.occs_a = data.mo_occs_a
        self.occs_b = data.mo_occs_b
        self.energy_a = data.mo_energy_a
        self.energy_b = data.mo_energy_b
        self.norba = len(self.energy_a) if self.energy_a is not None else None
        self.norbb = len(self.energy_b) if self.energy_a is not None else None
        self.nbasis = self.norba  # number of spatial basis functions


class Species:
    r"""Properties of atomic and ionic species."""

    def __init__(self, dataset, fields, spinpol=1):
        r"""Initialize a ``Species`` instance.

        Parameters
        ----------
        dataset : String
            Type of dataset to load.
        fields: dict
            Dictionary containing all fields to initialize `SpecfiesData` class object
        spinpol: int
            Spin polarization

        """
        self._dataset = dataset.lower()
        # converting fields from dict to DefinitionClass
        submodule = import_module(f"atomdb.datasets.{dataset}.run")
        fields = submodule.DefinitionClass(**fields)

        self._data = fields
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
        r"""Name of the type of dataset.

        Returns
        -------
        dataset : string

        """
        return self._dataset

    @property
    def charge(self):
        r"""Charge

        .. math::

            Q = Z - N

        Where Q is the charge, Z the atomic number and N the total number of electrons stored in
        SpeciesData object.

        Returns
        -------
        charge : int

        """
        return self._data.atnum - self._data.nelec

    @property
    def nspin(self):
        r"""Spin number

        .. math::
            N_S = N_α - N_β

        Where :math:`N_S` is the spin number and :math:`N_α` and :math:`N_β` number of alpha and \
            beta electrons.

        Returns
        -------
        nspin : int


        """
        return self._data.nspin * self._spinpol

    @property
    def mult(self):
        r"""Multiplicity

        .. math::

            M = \left|N_S\right| + 1

        Where :math:`\left|N_S\right|` is the spin number.

        Returns
        -------
        mult : int

        """
        return self._data.nspin + 1

    @property
    def spinpol(self):
        r"""Spin polarization direction (±1).

        Returns
        -------
        spinpol : int

        """
        return self._spinpol

    @spinpol.setter
    def spinpol(self, spinpol):
        r"""Spin polarization direction setter.

        Parameters
        ----------
        spinpol : `Integral` class from numbers package

        Raises
        ------
        TypeError
            If spinpol is not `Integral` type
            If absolute value is not equal to 1

        """
        if not isinstance(spinpol, Integral):
            raise TypeError("`spinpol` attribute must be an integral type")

        spinpol = int(spinpol)

        if abs(spinpol) != 1:
            raise ValueError("`spinpol` must be +1 or -1")

        self._spinpol = spinpol

    @scalar
    def elem(self):
        r"""Element symbol.

        Returns
        -------
        elem : string

        """
        pass

    @scalar
    def obasis_name(self):
        r"""Basis name.

        Returns
        -------
        obasis_name : string

        """
        pass

    @scalar
    def atnum(self):
        r"""Atomic number.

        Returns
        -------
        atnum : int

        """
        pass

    @scalar
    def nelec(self):
        r"""Number of electrons.

        Returns
        -------
        nelec : int

        """
        pass

    @scalar
    def atmass(self):
        r"""Atomic mass in atomic units.

        Returns
        -------
        atmass : dict
            Two options are available: the isotopically averaged mass 'stb', and the mass of the most
            common isotope 'nist'. For references corresponding to each type check
            https://github.com/theochem/AtomDB/blob/master/atomdb/data/data_info.csv

        """
        pass

    @scalar
    def cov_radius(self):
        r"""Covalent radius (derived from crystallographic data).

        Returns
        -------
        cov_radius : dict, float
            The return dictionary contains crystallographic covalent radii types `cordero`,
            `bragg` and `slater`. For references corresponding to each type check
            https://github.com/theochem/AtomDB/blob/master/atomdb/data/data_info.csv

        """
        pass

    @scalar
    def vdw_radius(self):
        r"""Van der Waals radii.

        Returns
        -------
        vdw_radius : dict, float
            The return dictionary contains van der wals radii types
            `bondi`, `truhlar`, `rt`, `batsanov`, `dreiding`, `uff` and `mm3`. For
            references corresponding to each type check
            https://github.com/theochem/AtomDB/blob/master/atomdb/data/data_info.csv

        """
        pass

    @scalar
    def at_radius(self):
        r"""Atomic radius.

        Returns
        -------
        at_radius : dict, float
            The return dictionary contains atomic radii types
            `wc` and `cr`. For references corresponding to each type check
            https://github.com/theochem/AtomDB/blob/master/atomdb/data/data_info.csv

        """
        pass

    @scalar
    def polarizability(self):
        r"""Isolated atom dipole polarizability.

        Returns
        -------
        polarizability : dict, float
            The return dictionary contains isolated atom dipole polarizability types
            `crc` and `chu`. For references corresponding to each type check
            https://github.com/theochem/AtomDB/blob/master/atomdb/data/data_info.csv

        """
        pass

    @scalar
    def dispersion_c6(self):
        r"""Isolated atom C6 dispersion coefficients.

        Returns
        -------
        dispersion_c6 : dict, float
            The return dictionary contains isolated atom C6 dispersion coefficients types
            `chu`. For references corresponding to each type check
            https://github.com/theochem/AtomDB/blob/master/atomdb/data/data_info.csv

        """
        pass

    @scalar
    def nexc(self):
        r"""Excitation number.

        Returns
        -------
        nexc : int

        """
        pass

    @scalar
    def energy(self):
        r"""Energy.

        Returns
        -------
        energy : float

        """
        pass

    @scalar
    def ip(self):
        r"""Ionization potential.

        Returns
        -------
        ip : float

        """
        pass

    @scalar
    def mu(self):
        r"""Chemical potential.

        Returns
        -------
        mu : float

        """
        pass

    @scalar
    def eta(self):
        r"""Chemical hardness.

        Returns
        -------
        eta : float

        """
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
        Return a cubic spline of the second derivative of the electronic density.

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
        Callable[[np.ndarray(N,), int] -> np.ndarray(N,)]
            a callable function evaluating the second derivative of the density given a set of radial
            points (1-D array).

        """
        pass

    def dd_dens_lapl_func(self, spin="t", index=None, log=False):
        r"""
        Return the function for the electronic density Laplacian.

        .. math::

            \nabla^2 \rho(\mathbf{r}) = \frac{d^2 \rho(r)}{dr^2} + \frac{2}{r} \frac{d \rho(r)}{dr}

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
        Callable[np.ndarray(N,) -> np.ndarray(N,)]
            a callable function evaluating the Laplacian of the density given a set of radial
            points (1-D array).

        Notes
        -----
        When this function is evaluated at a point close to zero, the Laplacian becomes undefined.
        In this case, this function returns zero.

        """
        # Obtain cubic spline functions for the first and second derivatives of the density
        d_dens_sp_spline = self.d_dens_func(spin=spin, index=index, log=log)
        dd_dens_spline = self.dd_dens_func(spin=spin, index=index, log=log)

        # Define the Laplacian function
        def densityspline_like_func(rs):
            # Avoid division by zero and handle small values of r
            with np.errstate(divide="ignore"):
                laplacian = dd_dens_spline(rs) + 2 * d_dens_sp_spline(rs) / rs
                laplacian = np.where(rs < 1e-10, 0.0, laplacian)
            return laplacian

        return densityspline_like_func

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


def compile_species(
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
):
    r"""Compile an atomic or ionic species into the AtomDB database.

    Parameters
    ----------
    elem : str
        Element symbol.
    charge : int
        Charge.
    mult : int
        Multiplicity.
    nexc : int, optional
        Excitation level, by default 0.
    dataset : str, optional
        Dataset name, by default DEFAULT_DATASET.
    datapath : str, optional
        Path to the local AtomDB cache, by default DEFAULT_DATAPATH variable value.

    """
    # import the selected dataset compile script and get fields
    submodule = import_module(f"atomdb.datasets.{dataset}.run")
    fields = submodule.run(elem, charge, mult, nexc, dataset, datapath)

    # dump the data to the HDF5 file
    dump(fields, dataset, mult)


def dump(fields, dataset, mult):
    r"""Dump the compiled species data to an HDF5 file in the AtomDB database.

    Parameters
    ----------
    fields (dataclass): A dataclass containing the fields to store in the HDF5 file.
    dataset (str): Name of the dataset.
    mult (int): Multiplicity.
    """

    # Save data to the HDF5 file
    element_folder_creator = import_module(f"atomdb.datasets.{dataset}.h5file_creator")
    element_folder_creator.create_hdf5_file(DATASETS_H5FILE, fields, dataset, mult)


def load(
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
    remotepath=DEFAULT_REMOTE,
):
    r"""Load one or many atomic or ionic species from the AtomDB database.

    Parameters
    ----------
    elem : str
        Element symbol.
    charge : int
        Charge.
    mult : int
        Multiplicity.
    nexc : int, optional
        Excitation level, by default 0.
    dataset : str, optional
        Dataset name, by default DEFAULT_DATASET.
    datapath : str, optional
        Path to the local AtomDB cache, by default DEFAULT_DATAPATH variable value.
    remotepath : str, optional
        Remote URL for AtomDB datasets, by default DEFAULT_REMOTE variable value

    Returns
    -------
    Object of class Species

    """

    # Construct the dataset path
    dataset_path = f"/Datasets/{dataset}"

    # import the selected dataset HDF5 file creator to access property configurations
    dataset_submodule = import_module(f"atomdb.datasets.{dataset}.h5file_creator")
    DATASET_PROPERTY_CONFIGS = getattr(dataset_submodule, f"{dataset.upper()}_PROPERTY_CONFIGS")

    # Handle wildcard case for loading multiple species
    if Ellipsis in (elem, charge, mult, nexc):
        data_paths = datafile(elem, charge, mult, nexc=nexc, dataset=dataset)
        # create a list to hold all species objects
        obj = []
        for data_path in data_paths:
            elem = data_path.split("/")[-2]
            # Construct the specific data path for each species
            fields = get_species_data(data_path, elem, DATASET_PROPERTY_CONFIGS)
            obj.append(Species(dataset, fields))
    else:
        # Construct the specific data path for a single species
        data_path = f"{dataset_path}/{elem}/{elem}_{charge:03d}_{mult:03d}_{nexc:03d}"
        # get the species data and then create a species object
        fields = get_species_data(data_path, elem, DATASET_PROPERTY_CONFIGS)
        obj = Species(dataset, fields)

    return obj


def datafile(
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
    remotepath=DEFAULT_REMOTE,
):
    r"""Return the paths to the database files for a species in AtomDB.

    Parameters
    ----------
    elem : str
        Element symbol.
    charge : int
        Charge.
    mult : int
        Multiplicity.
    nexc : int, optional
        Excitation level, by default 0.
    dataset : str, optional
        Dataset name, by default DEFAULT_DATASET.
    datapath : str, optional
        Path to the local AtomDB cache, by default DEFAULT_DATAPATH variable value.
    remotepath : str, optional
        Remote URL for AtomDB datasets, by default DEFAULT_REMOTE variable value

    Returns
    -------
    list
        paths to the database file of a species in AtomDB.

    """
    group_paths = []
    conditions = []

    # Access the dataset folder in the HDF5 file
    dataset_path = f"/Datasets/{dataset}"
    dataset_folder = DATASETS_H5FILE.get_node(dataset_path)

    if elem is not Ellipsis:
        conditions.append(f'(elem == b"{elem}")')  # b for bytes comparison
    if charge is not Ellipsis:
        conditions.append(f"(charge == {charge})")
    if mult is not Ellipsis:
        conditions.append(f"(mult == {mult})")
    if nexc is not Ellipsis:
        conditions.append(f"(nexc == {nexc})")

    if conditions:
        query_result = " & ".join(conditions) if conditions else None

        for elem, elem_folder in dataset_folder._v_groups.items():
            for species_folder in elem_folder._v_groups:
                properties_folder = DATASETS_H5FILE.get_node(
                    f"/Datasets/{dataset}/{elem}/{species_folder}/Properties"
                )
                species_info_table = properties_folder._f_get_child("species_info")

                matched_species = list(species_info_table.where(query_result))
                if matched_species:
                    group_paths.append(f"{dataset_path}/{elem}/{species_folder}")

    # if there are no conditions, return all species
    else:
        for elem, elem_folder in dataset_folder._v_groups.items():
            for species_folder in elem_folder._v_groups:
                group_paths.append(f"{dataset_path}/{elem}/{species_folder}")

    return group_paths


def get_species_data(folder_path, elem, DATASET_PROPERTY_CONFIGS):
    r"""Retrieve species data from the specified HDF5 folder path.

    Parameters
    ----------
    folder_path : str
        Path to the HDF5 folder containing the species data.
    elem : str
        Element symbol.
    DATASET_PROPERTY_CONFIGS : list
        list of configuration dictionaries.

    Returns
    -------
    dict
        the extracted species data fields.
    """
    fields = {}
    dataset_folder = DATASETS_H5FILE.get_node(folder_path)

    species_info_table = dataset_folder.Properties._f_get_child("species_info")
    species_info_row = species_info_table[0]

    # Iterate through property configurations to extract data from datasets_data.h5
    for config in DATASET_PROPERTY_CONFIGS:
        if "SpeciesInfo" in config:
            # Extract species info data
            prop_name = config["SpeciesInfo"]
            value = species_info_row[prop_name]
            if config["type"] == "string":
                value = value.decode("utf-8")
            fields[config["SpeciesInfo"]] = value

        elif "property" in config:
            # Extract single value properties
            table = dataset_folder.Properties._f_get_child(config["table_name"])
            value = table[0]["value"]
            if config["type"] == "string":
                value = value.decode("utf-8")
            fields[config["property"]] = value

        elif "array_property" in config:
            # Extract array properties
            table = dataset_folder.Properties._f_get_child(config["table_name"])
            fields[config["array_property"]] = table[0]["value"]

        elif "Carray_property" in config:
            # Extract Carray properties
            table = dataset_folder._f_get_child(config["folder"])._f_get_child(config["table_name"])
            fields[config["Carray_property"]] = table[:]

    fields["atnum"] = element_symbol_map[elem][ElementAttr.atnum]

    # Add scalar properties
    for prop in ("atmass", "cov_radius", "vdw_radius", "at_radius", "polarizability", "dispersion"):
        fields[prop] = get_scalar_data(prop, fields["atnum"], fields["nelec"])

    return fields


def raw_datafile(
    suffix,
    elem,
    charge,
    mult,
    nexc=0,
    dataset=DEFAULT_DATASET,
    datapath=DEFAULT_DATAPATH,
    remotepath=DEFAULT_REMOTE,
):
    r"""Returns the local path to the raw data file of a species

    This function retrieves the raw data file of a species from the AtomDB cache if present. If the
    file is not found, it is downloaded from the remote URL.

    Parameters
    ----------
    suffix : str
        File extension of the raw data file.
    elem : str
        Element symbol.
    charge : int
        Charge.
    mult : int
        Multiplicity.
    nexc : int, optional
        Excitation level, by default 0.
    dataset : str, optional
        Dataset name, by default DEFAULT_DATASET.
    datapath : str, optional
        Path to the local AtomDB cache, by default DEFAULT_DATAPATH variable value.
    remotepath : str, optional
        Remote URL for AtomDB datasets, by default DEFAULT_REMOTE variable value.

    Returns
    -------
    str
        Path to the raw data file.
    """
    # elem = "*" if elem is Ellipsis else element_symbol(elem) --> why using element_symbol here
    elem = "*" if elem is Ellipsis else elem
    charge = "*" if charge is Ellipsis else f"{charge:03d}"
    mult = "*" if mult is Ellipsis else f"{mult:03d}"
    nexc = "*" if nexc is Ellipsis else f"{nexc:03d}"

    # try to retrieve the file from the remote URL
    try:
        raw_file = pooch.retrieve(
            url=f"{remotepath}{dataset.lower()}/raw/{elem}_{charge}_{mult}_{nexc}{suffix}",
            known_hash=None,
            path=path.join(datapath, dataset.lower(), "raw"),
            fname=f"{elem}_{charge}_{mult}_{nexc}{suffix}",
        )
    # if the file is not found, use the local file
    except (requests.exceptions.HTTPError, ValueError):
        raw_file = path.join(
            datapath, dataset.lower(), "raw", f"{elem}_{charge}_{mult}_{nexc}{suffix}"
        )

    return raw_file
