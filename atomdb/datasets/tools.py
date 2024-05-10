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

r"""Utility functions for database compilation scripts."""

import numpy as np

from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.density import evaluate_density_gradient
from gbasis.evals.density import evaluate_density_hessian


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
