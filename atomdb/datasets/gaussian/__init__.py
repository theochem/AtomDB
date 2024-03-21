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
from gbasis.evals.density import evaluate_density_gradient
from gbasis.evals.density import evaluate_density_hessian
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density as eval_pd_ked
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.eval import evaluate_basis

from grid.onedgrid import UniformInteger
from grid.rtransform import ExpRTransform
from grid.atomgrid import AtomGrid

from iodata import load_one

import atomdb

from atomdb.periodic import Atom


__all__ = [
    "run",
]


# Parameters to generate an atomic grid from uniform radial grid
# Use 170 points, lmax = 21 for the Lebedev grid since our basis
# don't go beyond l=10 in the spherical harmonics.
BOUND = (1e-30, 2e1)  # (r_min, r_max)

NPOINTS = 1000

SIZE = 170  # Lebedev grid sizes

DEGREE = 21  #  Lebedev grid degrees


DOCSTRING = """Gaussian basis densities (UHF) Dataset

Electronic structure and density properties evaluated with def2-svpd basis set

"""


def _load_fchk(n_atom, element, n_elec, multi, basis_name, data_path):
    r"""Load Gaussian fchk file and return the iodata object

    This function finds the fchk file in the data directory corresponding to the given parameters,
    loads it and returns the iodata object.

    Parameters
    ----------
    n_atom : int
        Atomic number
    element : str
        Chemical symbol of the species
    n_elec : int
        Number of electrons
    multi : int
        Multiplicity
    basis_name : str
        Basis set name
    data_path : str
        Path to the data directory

    Returns
    -------
    iodata : iodata.IOData
        Iodata object containing the data from the fchk file
    """
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


def eval_orbs_radial_d_density(one_density_matrix, basis, points, transform=None):
    """Compute the radial derivative of the density orbital components.

    For a set of points, compute the radial derivative of the density component for each orbital
    given the basis set and the basis transformation matrix.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
    basis : gbasis.basis.Basis
        Basis set used to evaluate the radial derivative of the density
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the derivatives
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.

    Returns
    -------
    radial_orb_d_dens : np.ndarray(K, N)
        Radial derivative of the density at the set of points for each orbital component.
    """
    # compute the basis values for the points output shape (K, N)
    basis_val = evaluate_basis(basis, points, transform=transform)

    # compute unitary vectors for the directions of the points
    unitvect_pts = points / np.linalg.norm(points, axis=1)[:, None]

    # array to store the radial derivative of the density (orbital components)
    output = np.zeros_like(basis_val)

    # orders for the cartesian directions
    orders_one = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))

    # for each cartesian direction
    for ind, orders in enumerate(orders_one):
        # compute the derivative of each orbital for the cartesian direction
        deriv_comp = evaluate_deriv_basis(basis, points, orders, transform)
        # compute matrix product of 1RDM and d|phi_i|^2/dx(or y or z) for each point
        density = 2 * one_density_matrix @ basis_val * deriv_comp
        # project derivative components to the radial component
        density *= unitvect_pts[:, ind]

        output += density
    return output


def eval_orbs_radial_dd_density(one_density_matrix, basis, points, transform=None):
    """Compute the radial second derivative of the density orbital components.

    For a set of points, compute the radial second derivative of the density component for each
    orbital given the basis set and the basis transformation matrix.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix in terms of the given basis set.
    basis : gbasis.basis.Basis
        Basis set used to evaluate the radial derivative of the density
    points : np.ndarray(N, 3)
        Cartesian coordinates of the points in space (in atomic units) where the derivatives
        are evaluated.
        Rows correspond to the points and columns correspond to the :math:`x, y, \text{and} z`
        components.

    Returns
    -------
    radial_dd_orb_dens : np.ndarray(K, N)
        Radial second derivative of the density at the set of points for each orbital component.
    """
    # compute unitary vectors for the directions of the points
    unitvect_pts = points / np.linalg.norm(points, axis=1)[:, None]

    # compute the basis values for the points output shape (K, N)
    orb_val = evaluate_basis(basis, points, transform=transform)

    # compute first order derivatives of the basis for the points
    orb_d_x = evaluate_deriv_basis(basis, points, np.array([1, 0, 0]), transform)
    orb_d_y = evaluate_deriv_basis(basis, points, np.array([0, 1, 0]), transform)
    orb_d_z = evaluate_deriv_basis(basis, points, np.array([0, 0, 1]), transform)

    # assemble the gradient of basis functions for the points
    orb_1d = np.array([orb_d_x, orb_d_y, orb_d_z])

    # array to store the radial second derivative of the orbital components of density, shape (K, N)
    output = np.zeros((one_density_matrix.shape[0], points.shape[0]))

    # for each distinct element of the Hessian
    for i in range(3):
        for j in range(i + 1):
            # cartesian orders for the hessian element [i, j]
            hess_order = np.array([0, 0, 0])
            hess_order[i] += 1
            hess_order[j] += 1

            # derivative of the basis for the points with order hess_order
            orb_dd_ij = evaluate_deriv_basis(basis, points, hess_order, transform)

            # compute hessian of orbital contributions to the density
            #  2 * (dphi/di * 1RDM @ dphi/dj +  phi * 1RDM @ Hij)
            dd_rho_orb_ij = 2 * (
                np.einsum("il,ij,jl -> jl", orb_dd_ij, one_density_matrix, orb_val)
                + np.einsum("il,ij,jl -> jl", orb_1d[i], one_density_matrix, orb_1d[j])
            )

            # project the hessian of the orbital contributions to the density to the radial component
            increment = np.einsum(
                "i,ji,i -> ji", unitvect_pts[:, i], dd_rho_orb_ij, unitvect_pts[:, j]
            )
            # add the contribution to the output
            output += increment
            # if element not in the diagonal, add the symmetric contribution
            if i != j:
                output += increment
    return output


def eval_radial_d_density(one_density_matrix, basis, points):
    """Compute the radial derivative of the density.

    For a set of points, compute the radial derivative of the density
    given the one-electron density matrix and the basis set.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix (1DM) from K orbitals
    basis : gbasis.basis.Basis
        Basis set used to evaluate the radial derivative of the density
    points : np.ndarray(N, 3)
        Set of points where the radial derivative of the density is evaluated

    Returns
    -------
    radial_d_density : np.ndarray(N)
        Radial derivative of the density at the set of points
    """
    rho_grad = evaluate_density_gradient(one_density_matrix, basis, points)
    # compute unitary vectors for the directions of the points
    unitvect_pts = points / np.linalg.norm(points, axis=1)[:, None]
    # compute the radial derivative of the density
    return np.einsum("ij,ij->i", unitvect_pts, rho_grad)


def eval_radial_dd_density(one_density_matrix, basis, points):
    """Compute the radial derivative of the density.

    For a set of points, compute the radial derivative of the density
    given the one-electron density matrix and the basis set.

    Parameters
    ----------
    one_density_matrix : np.ndarray(K_orb, K_orb)
        One-electron density matrix (1DM) from K orbitals
    basis : gbasis.basis.Basis
        Basis set used to evaluate the radial derivative of the density
    points : np.ndarray(N, 3)
        Set of points where the radial derivative of the density is evaluated

    Returns
    -------
    radial_dd_density : np.ndarray(N)
        Radial derivative of the density at the set of points
    """
    rho_hess = evaluate_density_hessian(one_density_matrix, basis, points)
    # compute unitary vectors for the directions of the points
    unitvect_pts = points / np.linalg.norm(points, axis=1)[:, None]
    # compute the radial second derivative of the density
    return np.einsum("ij,ijk,ik->i", unitvect_pts, rho_hess, unitvect_pts)


def eval_orb_ked(one_density_matrix, basis, points, transform=None):
    "Adapted from Gbasis"
    orbt_ked = 0
    for orders in np.identity(3, dtype=int):
        deriv_orb_eval_one = evaluate_deriv_basis(basis, points, orders, transform=transform)
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

    # check for inconsistencies in filenames
    if not np.allclose(np.array([n_up, n_dn]), np.array([sum(occs_up), sum(occs_dn)])):
        raise ValueError(f"Inconsistent data in fchk file for N: {natom}, M: {mult} CH: {charge}")

    # Prepare data for computing Species properties
    # density matrix in AO basis
    dm1_up = np.dot(coeffs_a * occs_up, coeffs_a.T)
    dm1_dn = np.dot(coeffs_b * occs_dn, coeffs_b.T)
    dm1_tot = data.one_rdms["scf"]

    # Make grid
    onedg = UniformInteger(NPOINTS)  # number of uniform grid points.
    rgrid = ExpRTransform(*BOUND).transform_1d_grid(onedg)  # radial grid
    atgrid = AtomGrid(rgrid, degrees=[DEGREE], sizes=[SIZE], center=np.array([0.0, 0.0, 0.0]))

    # Evaluate properties on the grid:
    # --------------------------------
    # total and spin-up orbital, and spin-down orbital densities
    obasis = from_iodata(data)
    orb_eval = evaluate_basis(obasis, atgrid.points, transform=None)
    orb_dens_up = eval_orbs_density(dm1_up, orb_eval)
    orb_dens_dn = eval_orbs_density(dm1_dn, orb_eval)
    dens_tot = eval_dens(dm1_tot, obasis, atgrid.points, transform=None)

    # total, spin-up orbital, and spin-down orbital first (radial) derivatives of the density
    d_dens_tot = eval_radial_d_density(dm1_tot, obasis, atgrid.points)
    orb_d_dens_up = eval_orbs_radial_d_density(dm1_up, obasis, atgrid.points, transform=None)
    orb_d_dens_dn = eval_orbs_radial_d_density(dm1_dn, obasis, atgrid.points, transform=None)

    # total, spin-up orbital, and spin-down orbital second (radial) derivatives of the density
    dd_dens_tot = eval_radial_dd_density(dm1_tot, obasis, atgrid.points)
    orb_dd_dens_up = eval_orbs_radial_dd_density(dm1_up, obasis, atgrid.points, transform=None)
    orb_dd_dens_dn = eval_orbs_radial_dd_density(dm1_dn, obasis, atgrid.points, transform=None)

    # total, spin-up orbital, and spin-down orbital kinetic energy densities
    ked_tot = eval_pd_ked(dm1_tot, obasis, atgrid.points, transform=None)
    orb_ked_up = eval_orb_ked(dm1_up, obasis, atgrid.points, transform=None)
    orb_ked_dn = eval_orb_ked(dm1_dn, obasis, atgrid.points, transform=None)

    # Spherically average properties:
    # --------------------------------
    # total, spin-up orbital, and spin-down orbital densities
    dens_spherical_avg = atgrid.spherical_average(dens_tot)
    dens_splines_up = [atgrid.spherical_average(dens) for dens in orb_dens_up]
    dens_splines_dn = [atgrid.spherical_average(dens) for dens in orb_dens_dn]

    # total, spin-up orbital, and spin-down orbital radial derivatives of the density
    d_dens_spherical_avg = atgrid.spherical_average(d_dens_tot)
    d_dens_splines_up = [atgrid.spherical_average(d_dens) for d_dens in orb_d_dens_up]
    d_dens_splines_dn = [atgrid.spherical_average(d_dens) for d_dens in orb_d_dens_dn]

    # total, spin-up orbital, and spin-down orbital radial second derivatives of the density
    dd_dens_spherical_avg = atgrid.spherical_average(dd_dens_tot)
    dd_dens_splines_up = [atgrid.spherical_average(dd_dens) for dd_dens in orb_dd_dens_up]
    dd_dens_splines_dn = [atgrid.spherical_average(dd_dens) for dd_dens in orb_dd_dens_dn]

    # total, spin-up orbital, and spin-down orbital kinetic energy densities
    ked_spherical_avg = atgrid.spherical_average(ked_tot)
    ked_splines_up = [atgrid.spherical_average(dens) for dens in orb_ked_up]
    ked_splines_dn = [atgrid.spherical_average(dens) for dens in orb_ked_dn]

    # Evaluate interpolated densities in uniform radial grid:
    # -------------------------------------------------------
    rs = rgrid.points
    # total, spin-up orbital, and spin-down orbital densities
    dens_avg_tot = dens_spherical_avg(rs)
    orb_dens_avg_up = np.array([spline(rs) for spline in dens_splines_up])
    orb_dens_avg_dn = np.array([spline(rs) for spline in dens_splines_dn])

    # total, spin-up orbital, and spin-down orbital radial derivatives of the density
    d_dens_avg_tot = d_dens_spherical_avg(rs)
    orb_d_dens_avg_up = np.array([spline(rs) for spline in d_dens_splines_up])
    orb_d_dens_avg_dn = np.array([spline(rs) for spline in d_dens_splines_dn])

    # total, spin-up orbital, and spin-down orbital radial second derivatives of the density
    dd_dens_avg_tot = dd_dens_spherical_avg(rs)
    orb_dd_dens_avg_up = np.array([spline(rs) for spline in dd_dens_splines_up])
    orb_dd_dens_avg_dn = np.array([spline(rs) for spline in dd_dens_splines_dn])

    # total, spin-up orbital, and spin-down orbital kinetic energy densities
    ked_avg_tot = ked_spherical_avg(rs)
    orb_ked_avg_up = np.array([spline(rs) for spline in ked_splines_up])
    orb_ked_avg_dn = np.array([spline(rs) for spline in ked_splines_dn])

    # Get information about the element
    atom = Atom(elem)
    cov_radii = atom.cov_radius
    vdw_radii = atom.vdw_radius
    mass = atom.mass["stb"]
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
        DOCSTRING,
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
        # Density
        _orb_dens_up=orb_dens_avg_up.flatten(),
        _orb_dens_dn=orb_dens_avg_dn.flatten(),
        dens_tot=dens_avg_tot,
        # Density gradient
        _orb_d_dens_up=orb_d_dens_avg_up.flatten(),
        _orb_d_dens_dn=orb_d_dens_avg_dn.flatten(),
        d_dens_tot=d_dens_avg_tot,
        # Density laplacian
        _orb_dd_dens_up=orb_dd_dens_avg_up.flatten(),
        _orb_dd_dens_dn=orb_dd_dens_avg_dn.flatten(),
        dd_dens_tot=dd_dens_avg_tot,
        # KED
        _orb_ked_up=orb_ked_avg_up.flatten(),
        _orb_ked_dn=orb_ked_avg_dn.flatten(),
        ked_tot=ked_avg_tot,
    )
