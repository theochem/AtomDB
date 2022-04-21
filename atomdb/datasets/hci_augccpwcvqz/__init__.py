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

r"""HCI compile function."""

import numpy as np

from iodata import load_one

from gbasis.wrappers import from_iodata

from gbasis.evals.density import evaluate_density as eval_dens
from gbasis.evals.density import evaluate_deriv_density as eval_d_dens
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density as eval_pd_ked

import atomdb


__all__ = [
    "run",
]


BOUND = (0.01, 1.0)


NPOINTS = 100

BASIS = 'aug-ccpwCVQZ'


# def run(elem, charge, mult, nexc, basis, dataset, datapath):
def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Run an HCI computation and compile the AtomDB database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    natom = atomdb.element_number(elem)
    nelec = natom - charge
    nspin = mult - 1
    n_up = (nelec + nspin) // 2
    n_dn = (nelec - nspin) // 2
    basis = BASIS

    # Load restricted Hartree-Fock SCF
    scfdata = load_one(atomdb.datafile(".molden", elem, charge, mult, nexc, dataset, datapath))
    _mo_energy = scfdata.mo.energies
    _mo_occ = scfdata.mo.occs
    mo_coeff = scfdata.mo.coeffs

    # Load HCI data
    data = np.load(atomdb.datafile(".ci.npz", elem, charge, mult, nexc, dataset, datapath))
    energy = data['energy'][0]

    # Prepare data for computing Species properties
    dm1_up, dm1_dn = data['rdm1']
    dm1_tot = dm1_up + dm1_dn
    dm1_mag = dm1_up - dm1_dn

    # Make grid
    rs = np.linspace(*BOUND, NPOINTS)
    grid = np.zeros((NPOINTS, 3))
    grid[:, 0] = rs

    # Compute densities and derivatives
    obasis, coord_types = from_iodata(scfdata)
    order = np.array([1, 0, 0])
    dens_up = eval_dens(dm1_up, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    dens_dn = eval_dens(dm1_dn, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    dens_tot = eval_dens(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    dens_mag = eval_dens(dm1_mag, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    d_dens_up = eval_d_dens(order, dm1_up, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    d_dens_dn = eval_d_dens(order, dm1_dn, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    d_dens_tot = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    d_dens_mag = eval_d_dens(order, dm1_mag, obasis, grid, coord_type=coord_types, transform=mo_coeff)

    # Compute laplacian and kinetic energy density
    order = np.array([2, 0, 0])
    lapl_up = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    lapl_dn = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    lapl_tot = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    lapl_mag = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    ked_up = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    ked_dn = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    ked_tot = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    ked_mag = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    #
    # Element properties
    #
    cov_radii, vdw_radii, mass = atomdb.get_element_data(elem)
    #
    # Conceptual-DFT properties (TODO)
    #
    ip=None
    mu=None
    eta=None

    # Return Species instance
    return atomdb.Species(
        dataset,
        elem,
        natom,
        basis,
        nelec,
        nspin,
        nexc,
        cov_radii,
        vdw_radii,
        mass,
        energy,
        _mo_energy,
        _mo_occ,
        ip,
        mu,
        eta,
        rs,
        dens_up,
        dens_dn,
        dens_tot,
        dens_mag,
        d_dens_up,
        d_dens_dn,
        d_dens_tot,
        d_dens_mag,
        lapl_up,
        lapl_dn,
        lapl_tot,
        lapl_mag,
        ked_up,
        ked_dn,
        ked_tot,
        ked_mag,
    )
