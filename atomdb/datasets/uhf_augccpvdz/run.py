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

r"""UHF compile function."""

import numpy as np

from pyscf import gto, scf
from pyscf.tools import molden

from iodata import load_one

from gbasis.wrappers import from_iodata

from gbasis.evals.density import evaluate_density as eval_dens
from gbasis.evals.density import evaluate_deriv_density as eval_d_dens
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density as eval_pd_ked
from gbasis.evals.density import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis

from grid.onedgrid import UniformInteger
from grid.rtransform import ExpRTransform
from grid.atomgrid import AtomGrid
from dataclasses import dataclass
from typing import Optional, Dict
from atomdb.periodic_test import element_symbol_map, get_scalar_data

import atomdb


__all__ = [
    "run",
]


# Parameters to generate an atomic grid from uniform radial grid
# Use 170 points, lmax = 21 for the Lebedev grid since our basis
# don't go beyond l=10 in the spherical harmonics.
BOUND = (1e-5, 2e1)  # (r_min, r_max)

NPOINTS = 100

SIZE = 170  # Lebedev grid sizes

DEGREE = 21  #  Lebedev grid degrees


BASIS = "aug-cc-pVDZ"


DOCSTRING = """UHF Dataset

Electronic structure and density properties evaluated with aug-cc-pVDZ basis set

"""


@dataclass
class DefinitionClass:
    """Data structure for the Slater dataset."""

    # species info
    elem: str
    atnum: int
    nelec: int
    nspin: int
    nexc: int
    nbasis: int
    charge: int
    mult: int
    obasis_name: str

    # properties (all from multiple sources Dict[str, float] )
    atmass: Optional[Dict[str, float]]
    cov_radius: Optional[Dict[str, float]]
    vdw_radius: Optional[Dict[str, float]]
    at_radius: Optional[Dict[str, float]]
    polarizability: Optional[Dict[str, float]]
    dispersion: Optional[Dict[str, float]]

    # [float]
    energy: Optional[float]
    ip: Optional[float]
    mu: Optional[float]
    eta: Optional[float]

    # [np.ndarray]
    mo_energy_a: Optional[np.ndarray]
    mo_energy_b: Optional[np.ndarray]
    mo_occs_a: Optional[np.ndarray]
    mo_occs_b: Optional[np.ndarray]

    # Radial grid
    rs: np.ndarray = Optional[np.ndarray]

    # Density
    mo_dens_a: np.ndarray = Optional[np.ndarray]
    mo_dens_b: np.ndarray = Optional[np.ndarray]
    dens_tot: np.ndarray = Optional[np.ndarray]

    # KED
    mo_ked_a: np.ndarray = Optional[np.ndarray]
    mo_ked_b: np.ndarray = Optional[np.ndarray]
    ked_tot: np.ndarray = Optional[np.ndarray]


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


def eval_orb_ked(one_density_matrix, basis, points, transform=None):
    "Adapted from Gbasis"
    orbt_ked = 0
    for orders in np.identity(3, dtype=int):
        deriv_orb_eval_one = evaluate_deriv_basis(basis, points, orders, transform=transform)
        deriv_orb_eval_two = deriv_orb_eval_one  # orders_one == orders_two
        density = one_density_matrix.dot(deriv_orb_eval_two)
        density *= deriv_orb_eval_one
        orbt_ked += density
    return 0.5 * orbt_ked


def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Run an HCI computation and compile the AtomDB database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    atnum = element_symbol_map[elem][0]
    nelec = atnum - charge
    nspin = mult - 1
    n_up = (nelec + nspin) // 2
    n_dn = (nelec - nspin) // 2
    obasis_name = BASIS

    # Load restricted Hartree-Fock SCF
    mol = gto.Mole()
    mol.build(
        atom=[[elem, (0, 0, 0)]],
        basis=obasis_name,
        charge=charge,
        spin=nspin,
    )
    mf = scf.UHF(mol)
    mf.kernel()
    molden.from_scf(
        mf,
        atomdb.raw_datafile(".molden", elem, charge, mult, nexc, dataset, datapath),
        ignore_h=False,
    )

    scfdata = load_one(atomdb.raw_datafile(".molden", elem, charge, mult, nexc, dataset, datapath))
    norba = scfdata.mo.norba
    mo_e_up = scfdata.mo.energies[:norba]
    mo_e_dn = scfdata.mo.energies[norba:]
    occs_up = scfdata.mo.occs[:norba]
    occs_dn = scfdata.mo.occs[norba:]
    mo_coeff = scfdata.mo.coeffs

    energy = mf.e_tot

    # Prepare data for computing Species properties
    dm1_up, dm1_dn = mf.make_rdm1()
    dm1_tot = dm1_up + dm1_dn

    # Make grid
    onedg = UniformInteger(NPOINTS)  # number of uniform grid points.
    rgrid = ExpRTransform(*BOUND).transform_1d_grid(onedg)  # radial grid
    atgrid = AtomGrid(rgrid, degrees=[DEGREE], sizes=[SIZE], center=np.array([0.0, 0.0, 0.0]))

    # Compute densities
    obasis = from_iodata(scfdata)
    orb_eval = evaluate_basis(obasis, atgrid.points)
    orb_dens_up = eval_orbs_density(dm1_up, orb_eval)
    orb_dens_dn = eval_orbs_density(dm1_dn, orb_eval)
    # orb_dens_tot = orb_dens_up + orb_dens_dn
    # dens_tot = np.sum(orb_dens_tot, axis=0)
    dens_tot = eval_dens(dm1_tot, obasis, atgrid.points)

    # Compute kinetic energy density
    orb_ked_up = eval_orb_ked(dm1_up, obasis, atgrid.points)
    orb_ked_dn = eval_orb_ked(dm1_dn, obasis, atgrid.points)
    ked_tot = eval_pd_ked(dm1_tot, obasis, atgrid.points)

    # Density and KED spherical average
    dens_spherical_avg = atgrid.spherical_average(dens_tot)
    ked_spherical_avg = atgrid.spherical_average(ked_tot)
    dens_splines_up = [atgrid.spherical_average(dens) for dens in orb_dens_up]
    dens_splines_dn = [atgrid.spherical_average(dens) for dens in orb_dens_dn]
    ked_splines_up = [atgrid.spherical_average(dens) for dens in orb_ked_up]
    ked_splines_dn = [atgrid.spherical_average(dens) for dens in orb_ked_dn]
    # Evaluate interpolated densities in a uniform radial grid
    rs = rgrid.points
    dens_avg_tot = dens_spherical_avg(rs)
    orb_dens_avg_up = np.array([spline(rs) for spline in dens_splines_up])
    orb_dens_avg_dn = np.array([spline(rs) for spline in dens_splines_dn])
    ked_avg_tot = ked_spherical_avg(rs)
    orb_ked_avg_up = np.array([spline(rs) for spline in ked_splines_up])
    orb_ked_avg_dn = np.array([spline(rs) for spline in ked_splines_dn])

    #
    # Element properties
    #
    atmass = get_scalar_data("atmass", atnum, nelec)

    # get scalar data
    cov_radius, vdw_radius, at_radius, polarizability, dispersion = [
        None,
    ] * 5
    if charge == 0:
        cov_radius = get_scalar_data("cov_radius", atnum, nelec)
        vdw_radius = get_scalar_data("vdw_radius", atnum, nelec)
        at_radius = get_scalar_data("at_radius", atnum, nelec)
        polarizability = get_scalar_data("polarizability", atnum, nelec)
        dispersion = get_scalar_data("dispersion", atnum, nelec)

    #
    # Conceptual-DFT properties (TODO)
    #
    ip = None
    mu = None
    eta = None

    # Return Species instance
    fields = DefinitionClass(
        elem=elem,
        charge=charge,
        mult=mult,
        atnum=atnum,
        nbasis=norba,
        obasis_name=obasis_name,
        nelec=nelec,
        nspin=nspin,
        nexc=nexc,
        atmass=atmass,
        cov_radius=cov_radius,
        vdw_radius=vdw_radius,
        at_radius=at_radius,
        polarizability=polarizability,
        dispersion=dispersion,
        energy=energy,
        mo_energy_a=mo_e_up,
        mo_energy_b=mo_e_dn,
        mo_occs_a=occs_up,
        mo_occs_b=occs_dn,
        ip=ip,
        mu=mu,
        eta=eta,
        rs=rs,
        # Density
        mo_dens_a=orb_dens_avg_up.flatten(),
        mo_dens_b=orb_dens_avg_dn.flatten(),
        dens_tot=dens_avg_tot,
        # KED
        mo_ked_a=orb_ked_avg_up.flatten(),
        mo_ked_b=orb_ked_avg_dn.flatten(),
        ked_tot=ked_avg_tot,
    )
    return fields
