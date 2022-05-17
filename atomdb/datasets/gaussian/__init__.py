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

r"""Gaussian density compile function."""

import os

import numpy as np

from gbasis.wrappers import from_iodata

from gbasis.evals.density import evaluate_density as eval_dens
from gbasis.evals.density import evaluate_deriv_density as eval_d_dens
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density as eval_pd_ked

from iodata import load_one

import atomdb


__all__ = [
    "run",
]


BOUND = (0.01, 1.0)

NPOINTS = 100


def _load_fchk(n_atom, element, n_elec, multi, basis_name, data_path):
    bname = basis_name.lower().replace("-", "").replace("*", "p").replace("+", "d")
    prefix = f"atom_{str(n_atom).zfill(3)}_{element}"
    tag = f"N{str(n_elec).zfill(2)}_M{multi}"
    method = f"uhf_{bname}_g09"
    # fchkpath = os.path.join(os.path.dirname(__file__), f"raw/{prefix}_{tag}_{method}.fchk")
    fchkpath = os.path.join(data_path, f"gaussian/raw/{prefix}_{tag}_{method}.fchk")
    return load_one(fchkpath)


def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Compile the AtomDB database entry for densities from Gaussian wfn."""
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

    # Load data from fchk
    basis = 'def2-svpd'
    data = _load_fchk(natom, elem, nelec, mult, basis, datapath)

    # Unrestricted Hartree-Fock SCF results
    energy = data.energy
    mo_energy = data.mo.energies
    mo_occ = data.mo.occs
    mo_coeffs = data.mo.coeffs # (nbasis, norba + norbb)
    norba = data.mo.norba
    coeffs_a, coeffs_b = mo_coeffs[:, :norba], mo_coeffs[:, norba:]

    # Prepare data for computing Species properties
    dm1_up = np.dot(coeffs_a * mo_occ[:norba], coeffs_a.T)
    dm1_dn = np.dot(coeffs_b * mo_occ[norba:], coeffs_b.T)
    dm1_tot = dm1_up + dm1_dn
    dm1_mag = dm1_up - dm1_dn

    # Make grid
    rs = np.linspace(*BOUND, NPOINTS)
    grid = np.zeros((NPOINTS, 3))
    grid[:, 0] = rs

    # Compute densities and derivatives
    order = np.array([1, 0, 0])
    obasis, coord_types = from_iodata(data)
    dens_up = eval_dens(dm1_up, obasis, grid, coord_type=coord_types)
    dens_dn = eval_dens(dm1_dn, obasis, grid, coord_type=coord_types)
    dens_tot = eval_dens(dm1_tot, obasis, grid, coord_type=coord_types)
    dens_mag = eval_dens(dm1_mag, obasis, grid, coord_type=coord_types)
    d_dens_up = eval_d_dens(order, dm1_up, obasis, grid, coord_type=coord_types)
    d_dens_dn = eval_d_dens(order, dm1_dn, obasis, grid, coord_type=coord_types)
    d_dens_tot = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types)
    d_dens_mag = eval_d_dens(order, dm1_mag, obasis, grid, coord_type=coord_types)

    # Density spherical average (TODO)

    # Compute laplacian and kinetic energy density
    order = np.array([2, 0, 0])
    lapl_up = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types)
    lapl_dn = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types)
    lapl_tot = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types)
    lapl_mag = eval_d_dens(order, dm1_tot, obasis, grid, coord_type=coord_types)
    ked_up = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types)
    ked_dn = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types)
    ked_tot = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types)
    ked_mag = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types)
    #
    # Element properties
    #
    cov_radii, vdw_radii, mass = atomdb.get_element_data(elem)
    #
    # Conceptual-DFT properties (TODO)
    #
    # NOTE: Only the alpha component of the MOs is used bellow
    mo_energy_occ_up = mo_energy[:norba][:n_up]
    mo_energy_virt_up = mo_energy[:norba][n_up:]
    ip = -mo_energy_occ_up[-1] # - energy_HOMO_alpha
    ea = -mo_energy_virt_up[0] # - energy_LUMO_alpha
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
        mo_energy,
        mo_occ,
        ip,
        ea,
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
