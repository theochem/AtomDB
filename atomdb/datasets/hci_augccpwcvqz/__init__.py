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
from gbasis.evals.density import evaluate_basis

import atomdb


__all__ = [
    "run",
]


BOUND = (0.01, 1.0)


NPOINTS = 100

BASIS = 'aug-ccpwCVQZ'


DOCSTRING = """Heat-bath Configuration Interaction (HCI) Dataset

Electronic structure and density properties evaluated with aug-ccpwCVQZ basis set

"""


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
    norba = scfdata.mo.norba
    mo_e_up = scfdata.mo.energies[:norba]
    mo_e_dn = scfdata.mo.energies[norba:]
    occs_up = scfdata.mo.occs[:norba]
    occs_dn = scfdata.mo.occs[norba:]
    # _mo_energies = np.array([_mo_e_up, _mo_e_dn])  # (energy_a, energy_b)
    # _mo_occs = np.array([occs_up, occs_dn])  # (occs_a, occs_b)
    mo_coeff = scfdata.mo.coeffs

    # Load HCI data
    data = np.load(atomdb.datafile(".ci.npz", elem, charge, mult, nexc, dataset, datapath))
    energy = data['energy'][0]

    # Prepare data for computing Species properties
    dm1_up, dm1_dn = data['rdm1']
    dm1_tot = dm1_up + dm1_dn

    # Make grid
    rs = np.linspace(*BOUND, NPOINTS)
    grid = np.zeros((NPOINTS, 3))
    grid[:, 0] = rs

    # Compute densities
    obasis, coord_types = from_iodata(scfdata)
    orb_eval = evaluate_basis(obasis, grid, coord_type=coord_types, transform=mo_coeff)
    orb_dens_up = eval_orbs_density(dm1_up, orb_eval)
    orb_dens_dn = eval_orbs_density(dm1_dn, orb_eval)
    orb_dens_tot = orb_dens_up + orb_dens_dn
    dens_tot = np.sum(orb_dens_tot, axis=0)
    # dens_tot = eval_dens(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)

    # Compute kinetic energy density
    # TODO: replace bellow by kinetic energy density per orbital
    ked_up = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    ked_dn = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)
    ked_tot = eval_pd_ked(dm1_tot, obasis, grid, coord_type=coord_types, transform=mo_coeff)

    # Density and KED spherical average (TODO)

    #
    # Element properties
    #
    cov_radii, vdw_radii, mass = atomdb.get_element_data(elem)
    if charge != 0:
        cov_radii, vdw_radii = [None, None]  # overwrite values for charged species
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
        mo_e_up,
        mo_e_dn,
        occs_up,
        occs_dn,
        ip,
        mu,
        eta,
        rs=rs,
        orb_dens_up=orb_dens_up,
        orb_dens_dn=orb_dens_dn,
        dens_tot=dens_tot,
        ked_up=ked_up,
        ked_dn=ked_dn,
        ked_tot=ked_tot,
    )
