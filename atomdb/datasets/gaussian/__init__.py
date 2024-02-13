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
from gbasis.evals.density import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis

from grid.onedgrid import UniformInteger
from grid.rtransform import ExpRTransform
from grid.atomgrid import AtomGrid

from iodata import load_one

import atomdb


__all__ = [
    "run",
]


# Parameters to generate an atomic grid from uniform radial grid
# Use 170 points, lmax = 21 for the Lebedev grid since our basis
# don't go beyond l=10 in the spherical harmonics.
BOUND = (1e-5, 2e1)  # (r_min, r_max)

NPOINTS = 1000

SIZE = 170  # Lebedev grid sizes

DEGREE = 21  #  Lebedev grid degrees


DOCSTRING = """Gaussian basis densities (UHF) Dataset

Electronic structure and density properties evaluated with def2-svpd basis set

"""


def _load_fchk(n_atom, element, n_elec, multi, basis_name, data_path):
    bname = basis_name.lower().replace("-", "").replace("*", "p").replace("+", "d")
    prefix = f"atom_{str(n_atom).zfill(3)}_{element}"
    tag = f"N{str(n_elec).zfill(2)}_M{multi}"
    method = f"uhf_{bname}_g09"
    # fchkpath = os.path.join(os.path.dirname(__file__), f"raw/{prefix}_{tag}_{method}.fchk")
    fchkpath = os.path.join(data_path, f"gaussian/raw/{prefix}_{tag}_{method}.fchk")
    return load_one(fchkpath)


def eval_orbs_density(one_density_matrix, orb_eval):
    r"""Return each orbital density evaluated at a set of points

    rho_i(r) = \sum_j P_ij \phi_i(r) \phi_j(r)

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix (1DM) from K orbitals
    orb_eval : np.ndarray(K_orb, N)
        orbitals evaluated at a set of grid points (N).
        These orbitals must be the basis used to evaluate the 1DM.

    Returns
    -------
    orb_dens : np.ndarray(K_orb, N)
        orbitals density at a set of grid points (N)
    """
    #
    # Following lines were taken from Gbasis eval.py module (L60-L61)
    #
    density = one_density_matrix.dot(orb_eval)
    density *= orb_eval
    return density


def eval_orb_ked(one_density_matrix, basis, points, transform=None, coord_type="spherical"):
    "Adapted from Gbasis"
    orbt_ked = 0
    for orders in np.identity(3, dtype=int):
        deriv_orb_eval_one = evaluate_deriv_basis(
            basis, points, orders, transform=transform, coord_type=coord_type
        )
        deriv_orb_eval_two = deriv_orb_eval_one  # if orders_one == orders_two
        density = one_density_matrix.dot(deriv_orb_eval_two)
        density *= deriv_orb_eval_one
        orbt_ked += density
    return 0.5 * orbt_ked


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
    basis = "def2-svpd"
    data = _load_fchk(natom, elem, nelec, mult, basis, datapath)

    # Unrestricted Hartree-Fock SCF results
    energy = data.energy
    norba = data.mo.norba
    mo_e_up = data.mo.energies[:norba]
    mo_e_dn = data.mo.energies[norba:]
    occs_up = data.mo.occs[:norba]
    occs_dn = data.mo.occs[norba:]
    mo_coeffs = data.mo.coeffs  # ndarray(nbasis, norba + norbb)
    coeffs_a = mo_coeffs[:, :norba]
    coeffs_b = mo_coeffs[:, norba:]

    # Prepare data for computing Species properties
    # density matrix in AO basis
    dm1_up = np.dot(coeffs_a * occs_up, coeffs_a.T)
    dm1_dn = np.dot(coeffs_b * occs_dn, coeffs_b.T)
    dm1_tot = dm1_up + dm1_dn

    # Make grid
    onedg = UniformInteger(NPOINTS)  # number of uniform grid points.
    rgrid = ExpRTransform(*BOUND).transform_1d_grid(onedg)  # radial grid
    atgrid = AtomGrid(rgrid, degrees=[DEGREE], sizes=[SIZE], center=np.array([0.0, 0.0, 0.0]))

    # Compute densities
    obasis, coord_types = from_iodata(data)
    orb_eval = evaluate_basis(obasis, atgrid.points, coord_type=coord_types, transform=None)
    orb_dens_up = eval_orbs_density(dm1_up, orb_eval)
    orb_dens_dn = eval_orbs_density(dm1_dn, orb_eval)
    dens_tot = eval_dens(dm1_tot, obasis, atgrid.points, coord_type=coord_types, transform=None)

    # Compute kinetic energy density
    orb_ked_up = eval_orb_ked(dm1_up, obasis, atgrid.points, transform=None, coord_type=coord_types)
    orb_ked_dn = eval_orb_ked(dm1_dn, obasis, atgrid.points, transform=None, coord_type=coord_types)
    ked_tot = eval_pd_ked(dm1_tot, obasis, atgrid.points, coord_type=coord_types, transform=None)

    # Spherically average densities and orbital densities
    dens_spherical_avg = atgrid.spherical_average(dens_tot)
    dens_splines_up = [atgrid.spherical_average(dens) for dens in orb_dens_up]
    dens_splines_dn = [atgrid.spherical_average(dens) for dens in orb_dens_dn]
    ked_spherical_avg = atgrid.spherical_average(ked_tot)
    ked_splines_up = [atgrid.spherical_average(dens) for dens in orb_ked_up]
    ked_splines_dn = [atgrid.spherical_average(dens) for dens in orb_ked_dn]
    # Evaluate interpolated densities in uniform radial grid
    rs = rgrid.points
    dens_avg_tot = dens_spherical_avg(rs)
    orb_dens_avg_up = np.array([spline(rs) for spline in dens_splines_up])
    orb_dens_avg_dn = np.array([spline(rs) for spline in dens_splines_dn])
    ked_avg_tot = ked_spherical_avg(rs)
    orb_ked_avg_up = np.array([spline(rs) for spline in ked_splines_up])
    orb_ked_avg_dn = np.array([spline(rs) for spline in ked_splines_dn])

    # Get information about the element
    cov_radii, vdw_radii, mass = atomdb.get_element_data(elem)
    if charge != 0:
        cov_radii, vdw_radii = [None, None]  # overwrite values for charged species

    # Conceptual-DFT properties (WIP)
    # NOTE: Only the alpha component of the MOs is used bellow
    mo_energy_occ_up = mo_e_up[:n_up]
    mo_energy_virt_up = mo_e_up[n_up:]
    ip = -mo_energy_occ_up[-1]  # - energy_HOMO_alpha
    ea = -mo_energy_virt_up[0]  # - energy_LUMO_alpha
    mu = None
    eta = None

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
        mo_e_up,
        mo_e_dn,
        occs_up,
        occs_dn,
        ip,
        mu,
        eta,
        rs=rs,
        _orb_dens_up=orb_dens_avg_up.flatten(),
        _orb_dens_dn=orb_dens_avg_dn.flatten(),
        dens_tot=dens_avg_tot,
        _orb_ked_up=orb_ked_avg_up.flatten(),
        _orb_ked_dn=orb_ked_avg_dn.flatten(),
        ked_tot=ked_avg_tot,
    )
