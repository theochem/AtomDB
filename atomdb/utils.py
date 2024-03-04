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

r"""Tool functions."""

import numpy as np


def grad2ddens(gradient, points):
    """Evalaute the radial component of the gradient of the density as

    .. math::
    
        \frac{\partial \rho (\mathbf{r_n})}{\partial r} = \nabla \rho (\mathbf{r_n}) \cdot \mathbf{\hat{r}_n}
    
    where :math:`\nabla \rho (\mathbf{r_n})` is the gradient of the density evaluated at the grid points
    :math:`\mathbf{r_n}` and :math:`\mathbf{\hat{r}_n}` is the radial unit vector.

    Parameters
    ----------
    gradient : (N, 3) array
        Gradient of the density evaluated at the grid points.
    points : (N, 3) array
        Grid points.
    
    Returns
    -------
    (N,) array
        Radial component of the gradient of the density.
    """
    return np.einsum("ij,ij->i", gradient, points) / np.linalg.norm(points, axis=1)
