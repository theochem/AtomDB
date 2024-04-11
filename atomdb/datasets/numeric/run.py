# -*- coding: utf-8 -*-
# AtomDB is an extended periodic table database containing experimental
# and/or computational information on stable ground state
# and/or excited states of neutral and charged atomic species.
#
# Copyright (C) 2014-2015 The AtomDB Development Team
#
# This file is part of AtomDB.
#
# AtomDB is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AtomDB is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
#
"""Numerical Hartree-Fock Data on a Grid."""


import os

import numpy as np

import atomdb

from atomdb.utils import MULTIPLICITIES

from atomdb.periodic import Element


def load_numerical_hf_data():
    """Load data from desnity.out file into a `SpeciesTable`."""

    from io import StringIO

    def helper_skip():
        """Skip the header for each species."""
        for _ in range(4):
            f.readline()

    def helper_data():
        """Read the grid, density, gradient, laplacian values into arrays."""
        data = ""
        for i in range(number_points):
            data += f.readline()
        data = np.loadtxt(StringIO(data))
        assert data.shape == (number_points, 4)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    species = {}
    with open(os.path.join(os.path.dirname(__file__), "raw/density.out"), "r") as f:
        line = f.readline()
        while line:
            if line.startswith(" 1st line is atomic no"):
                helper_skip()
                line = f.readline()
            if line.startswith("     "):
                # dictionary to store attributes of the new atomic species
                kwargs = {}

                atomic_number, number_electrons, number_points = [
                    int(item) for item in line.split()
                ]
                energy = [float(item) for item in f.readline().split()]
                assert len(energy) == 5
                assert abs(sum(energy[:-1]) - energy[-1]) < 1.0e-8  # 1.e-12 didn"t work?!

                # store energy component values
                kwargs["energy_components"] = dict(
                    [
                        ("T", energy[0]),
                        ("Vne", energy[1]),
                        ("J", energy[2]),
                        ("Ex", energy[3]),
                        ("E", energy[4]),
                    ]
                )

                grid, density, gradient, laplacian = helper_data()
                assert grid.shape == density.shape == gradient.shape == laplacian.shape

                # store grid dependent values
                kwargs.update(
                    {"grid": grid, "density": density, "gradient": gradient, "laplacian": laplacian}
                )

                # add the new atomic species
                species[(atomic_number, number_electrons)] = kwargs
                line = f.readline()

    return species


DOCSTRING = """Numeric Dataset

Load data from desnity.out file into a `SpeciesTable`.

"""


def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Compile the densities from Slater orbitals database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")
    if charge < -1:
        raise ValueError("The charge must be greater than or equal to -1.")
    if charge == elem:
        raise ValueError("The atomic number and charge cannot be the same.")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    atnum = atomdb.element_number(elem)
    nelec = atnum - charge
    nspin = mult - 1
    n_up = (nelec + nspin) // 2
    n_dn = (nelec - nspin) // 2
    obasis_name = None

    # Check that the input multiplicity corresponds to the most stable electronic configuration.
    # For charged species take the multiplicity from the neutral isoelectronic species.
    expected_mult = MULTIPLICITIES[(atnum, charge)]
    if mult != expected_mult:
        raise ValueError(
            f"Multiplicity {mult} not available for {elem} with charge = {charge}. "
            f"Expected multiplicity is {expected_mult}."
        )

    species_table = load_numerical_hf_data()
    data = species_table[(atnum, nelec)]

    #
    # Element periodic properties
    #
    atom = Element(elem)
    atmass = atom.mass["stb"]
    cov_radius, vdw_radius, at_radius, polarizability, dispersion_c6 = [
        None,
    ] * 5
    if charge == 0:
        # overwrite values for neutral atomic species
        cov_radius, vdw_radius, at_radius = (atom.cov_radius, atom.vdw_radius, atom.at_radius)
        polarizability = atom.pold
        dispersion_c6 = atom.c6

    # Get electronic structure data
    energy = data["energy_components"]["E"]

    # Make grid
    points = data["grid"]

    # Compute densities and derivatives
    dens_tot = data["density"]
    d_dens_tot = data['gradient']

    # Compute laplacian and kinetic energy density
    lapl_tot = data['laplacian']
    ked_tot = None

    # Return Species instance
    fields = dict(
        elem=elem,
        atnum=atnum,
        obasis_name=obasis_name,
        nelec=nelec,
        nspin=nspin,
        nexc=nexc,
        atmass=atmass,
        cov_radius=cov_radius,
        vdw_radius=vdw_radius,
        at_radius=at_radius,
        polarizability=polarizability,
        dispersion_c6=dispersion_c6,
        energy=energy,
        rs=points,
        dens_tot=dens_tot,
        d_dens_tot=d_dens_tot,
        dd_dens_tot=lapl_tot,
        ked_tot=ked_tot,
    )
    return atomdb.Species(dataset, fields)
