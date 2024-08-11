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

import os

import numpy as np

from iodata import load_one

from gbasis.wrappers import from_iodata

from gbasis.evals.density import evaluate_density as eval_dens
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density as eval_pd_ked
from gbasis.evals.density import evaluate_basis

from grid.onedgrid import UniformInteger
from grid.rtransform import ExpRTransform
from grid.atomgrid import AtomGrid

import atomdb

from atomdb.periodic import Element
from atomdb.datasets.tools import (
    eval_orbs_density,
    eval_orbs_radial_d_density,
    eval_orbs_radial_dd_density,
    eval_orb_ked,
    eval_radial_d_density,
    eval_radial_dd_density,
    eval_orb_ked,
)


__all__ = [
    "run",
]


# Parameters to generate an atomic grid from uniform radial grid
# Use 170 points, lmax = 21 for the Lebedev grid since our basis
# don't go beyond l=10 in the spherical harmonics.
BOUND = (1e-10, 2e1)  # (r_min, r_max)

NPOINTS = 200

SIZE = 170  # Lebedev grid sizes

DEGREE = 21  #  Lebedev grid degrees


BASIS = "aug-ccpwCVQZ"

def raw_filepath(suffix, n_atom, charge, mult, nexc, basis, dataset, data_path):
    G1G2 = [1, 2, 3, 4, 11, 12, 19, 20]  # Group 1 and 2 elements
    elem = f"{n_atom:04d}"
    charge = f"q{charge:03d}"
    mult = f"m{mult:02d}"
    nexc = f"k{nexc:02d}"

    if n_atom in G1G2:
        basis = "aug-cc-pVQZ"  # basis set for Group 1 and 2 elements
    bname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")

    tag = f"{elem}_{charge}_{mult}_{nexc}"
    method = f"sp_hci_{bname}_5e-4"
    rawpath = os.path.join(data_path, f"{dataset}/raw/{tag}_{method}{suffix}")
    return rawpath


def run(elem, charge, mult, nexc, dataset, datapath):
    r"""Run an HCI computation and compile the AtomDB database entry."""
    # Check arguments
    if nexc != 0:
        raise ValueError("Nonzero value of `nexc` is not currently supported")

    # Set up internal variables
    elem = atomdb.element_symbol(elem)
    atnum = atomdb.element_number(elem)
    nelec = atnum - charge
    nspin = mult - 1
    n_up = (nelec + nspin) // 2
    n_dn = (nelec - nspin) // 2
    obasis_name = BASIS

    # Load restricted Hartree-Fock SCF
    if False:
        rawpath = raw_filepath(".molden", atnum, charge, mult, nexc, BASIS, dataset, datapath)
        scfdata = load_one(rawpath)
        norba = scfdata.mo.norba
        mo_e_up = scfdata.mo.energies[:norba]
        mo_e_dn = mo_e_up  # since only alpha MO information in .molden
        occs_up = scfdata.mo.occs  # ndarray(nbasis, nmos)
        occs_dn = None
        mo_coeff = scfdata.mo.coeffs
    # Load restricted Hartree-Fock SCF instead from fchk
    if True:
        rawpath= raw_filepath(".fchk",  atnum, charge, mult, nexc, BASIS, dataset, datapath)
        scfdata = load_one(rawpath) 
        norba =    scfdata.mo.norba # yes
        mo_e_up =  scfdata.mo.energies[:norba] # yes
        mo_e_dn =  scfdata.mo.energies[norba:] # yes
        occs_up =  scfdata.mo.occs[:norba] # yes
        occs_dn =  scfdata.mo.occs[norba:] # ?
        mo_coeff = scfdata.mo.coeffs  # thats different now, ndarray(nbasis, norba + norbb)
    


    # Load HCI data
    rawpath = raw_filepath(".ci.npz", atnum, charge, mult, nexc, BASIS, dataset, datapath)
    data = np.load(rawpath)
    energy = data["energy"]

    # Prepare data for computing Species properties
    # density matrix in MO basis
    dm1_up, dm1_dn = data["rdm1"]
    dm1_tot = dm1_up + dm1_dn

    # Make grid
    onedg = UniformInteger(NPOINTS)  # number of uniform grid points.
    rgrid = ExpRTransform(*BOUND).transform_1d_grid(onedg)  # radial grid
    atgrid = AtomGrid(rgrid, degrees=[DEGREE], sizes=[SIZE], center=np.array([0.0, 0.0, 0.0]))

    # Evaluate properties on the grid:
    # --------------------------------
    # total and spin-up orbital, and spin-down orbital densities

    obasis = from_iodata(scfdata)
    orb_eval = evaluate_basis(obasis, atgrid.points, transform=mo_coeff.T)
    orb_dens_up = eval_orbs_density(dm1_up, orb_eval)
    orb_dens_dn = eval_orbs_density(dm1_dn, orb_eval)
    dens_tot = eval_dens(dm1_tot, obasis, atgrid.points, transform=mo_coeff.T)

    # total, spin-up orbital, and spin-down orbital first (radial) derivatives of the density
    d_dens_tot = eval_radial_d_density(dm1_tot, obasis, atgrid.points)
    orb_d_dens_up = eval_orbs_radial_d_density(dm1_up, obasis, atgrid.points, transform=mo_coeff.T)
    orb_d_dens_dn = eval_orbs_radial_d_density(dm1_dn, obasis, atgrid.points, transform=mo_coeff.T)

    # total, spin-up orbital, and spin-down orbital first (radial) derivatives of the density
    dd_dens_tot = eval_radial_dd_density(dm1_tot, obasis, atgrid.points)
    orb_dd_dens_up = eval_orbs_radial_dd_density(
        dm1_up, obasis, atgrid.points, transform=mo_coeff.T
    )
    orb_dd_dens_dn = eval_orbs_radial_dd_density(
        dm1_dn, obasis, atgrid.points, transform=mo_coeff.T
    )

    # total, spin-up orbital, and spin-down orbital kinetic energy densities
    ked_tot = eval_pd_ked(dm1_tot, obasis, atgrid.points, transform=mo_coeff.T)
    orb_ked_up = eval_orb_ked(dm1_up, obasis, atgrid.points, transform=mo_coeff.T)
    orb_ked_dn = eval_orb_ked(dm1_dn, obasis, atgrid.points, transform=mo_coeff.T)

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

    # Evaluate interpolated densities in a uniform radial grid
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
    atom = Element(elem)
    atmass = atom.mass
    cov_radius, vdw_radius, at_radius, polarizability, dispersion = [
        None,
    ] * 5
    # overwrite values for neutral atomic species
    if charge == 0:
        cov_radius, vdw_radius, at_radius = (atom.cov_radius, atom.vdw_radius, atom.at_radius)
        polarizability = atom.pold
        dispersion = {"C6": atom.c6}

    # Conceptual-DFT properties (TODO)
    ip = None
    mu = None
    eta = None

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
        # Density gradient
        mo_d_dens_a=orb_d_dens_avg_up.flatten(),
        mo_d_dens_b=orb_d_dens_avg_dn.flatten(),
        d_dens_tot=d_dens_avg_tot,
        # Density laplacian
        mo_dd_dens_a=orb_dd_dens_avg_up.flatten(),
        mo_dd_dens_b=orb_dd_dens_avg_dn.flatten(),
        dd_dens_tot=dd_dens_avg_tot,
        # KED
        mo_ked_a=orb_ked_avg_up.flatten(),
        mo_ked_b=orb_ked_avg_dn.flatten(),
        ked_tot=ked_avg_tot,
    )
    return atomdb.Species(dataset, fields)
