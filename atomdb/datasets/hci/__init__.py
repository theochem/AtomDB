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

from tempfile import NamedTemporaryFile

import numpy as np

from basis_set_exchange import get_basis

from pyscf import gto, scf

from pyscf.tools import fcidump

import pyci

from gbasis.parsers import parse_nwchem, make_contractions

from gbasis.evals.density import evaluate_density as eval_dens
from gbasis.evals.density import evaluate_deriv_density as eval_d_dens
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density as eval_pd_ked

import atomdb


__all__ = [
    "run",
]


NTHREAD = 1


EPSILON = 1.0e-4


CONV_TOL = 1.0e-12


BOUND = (0.01, 1.0)


NPOINTS = 100


def run(elem, basis, charge, mult, nexc, dataset, datapath):
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

    # Get basis set
    nwchem = get_basis(basis, elements=natom, fmt="nwchem", header=False)

    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = f"{elem} 0.0 0.0 0.0"
    mol.charge = natom - nelec
    mol.spin = nspin
    mol.basis = gto.basis.parse(nwchem)
    mol.build()

    # Run restricted Hartree-Fock SCF
    hf = scf.RHF(mol)
    hf.conv_tol = 1.0e-12
    hf.run(verbose=4)
    mo_energy = hf.mo_energy
    mo_occ = hf.mo_occ
    mo_coeff = hf.mo_coeff.transpose()

    # Save Hartree-Fock data
    with open(atomdb.datafile("hf.txt", elem, basis, charge, mult, nexc, dataset, datapath), "w") as f:
        print(f"{basis}\n{hf.e_tot:24.16e}", file=f)
    with open(atomdb.datafile("basis.nwchem", elem, basis, charge, mult, nexc, dataset, datapath), "w") as f:
        print(nwchem, file=f)
    np.savez(
        atomdb.datafile("hf.npz", elem, basis, charge, mult, nexc, dataset, datapath),
        mo_energy=mo_energy,
        mo_occ=mo_occ,
        mo_coeff=mo_coeff,
    )

    # Generate molecular integrals
    fcidump_file = NamedTemporaryFile()
    fcidump.from_scf(hf, fcidump_file.name, tol=0.0)

    # Run HCI
    pyci.set_num_threads(NTHREAD)
    ham = pyci.hamiltonian(fcidump_file.name)
    fwfn = pyci.fullci_wfn(ham.nbasis, n_up, n_dn)
    pyci.add_excitations(fwfn, 0, 1)
    eigenvals, eigenvecs = pyci.solve(ham, fwfn, n=1, tol=CONV_TOL)
    dets_added = len(fwfn)
    while dets_added:
        dets_added = pyci.add_hci(ham, fwfn, eigenvecs[0], eps=EPSILON)
        eigenvals, eigenvecs = pyci.solve(ham, fwfn, n=1, tol=CONV_TOL)
    energy = eigenvals[0]

    # Compute HCI RDMs
    rdm1, rdm2 = pyci.compute_rdms(fwfn, eigenvecs[0])

    # Save HCI data
    fwfn.to_file(atomdb.datafile("hci.ci", elem, basis, charge, mult, nexc, dataset, datapath))
    np.savez(
        atomdb.datafile("hci.npz", elem, basis, charge, mult, nexc, dataset, datapath),
        energy=eigenvals,
        coeff=eigenvecs,
        rdm1=rdm1,
        rdm2=rdm2,
    )

    # Prepare data for computing Species properties
    dm1_up, dm1_dn = rdm1
    dm1_tot = dm1_up + dm1_dn
    dm1_mag = dm1_up - dm1_dn

    # Make grid
    rs = np.linspace(*BOUND, NPOINTS)
    grid = np.zeros((NPOINTS, 3))
    grid[:, 0] = rs

    # Compute densities and derivatives
    nwchem_file = atomdb.datafile("basis.nwchem", elem, basis, charge, mult, nexc, dataset, datapath)
    obasis = make_contractions(parse_nwchem(nwchem_file), [elem], np.array([[0, 0, 0]]))
    order = np.array([1, 0, 0])
    dens_up = eval_dens(dm1_up, obasis, grid, coord_type="spherical", transform=mo_coeff)
    dens_dn = eval_dens(dm1_dn, obasis, grid, coord_type="spherical", transform=mo_coeff)
    dens_tot = eval_dens(dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    dens_mag = eval_dens(dm1_mag, obasis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_up = eval_d_dens(order, dm1_up, obasis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_dn = eval_d_dens(order, dm1_dn, obasis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_tot = eval_d_dens(order, dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_mag = eval_d_dens(order, dm1_mag, obasis, grid, coord_type="spherical", transform=mo_coeff)

    # Compute laplacian and kinetic energy density
    order = np.array([2, 0, 0])
    lapl_up = eval_d_dens(order, dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    lapl_dn = eval_d_dens(order, dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    lapl_tot = eval_d_dens(order, dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    lapl_mag = eval_d_dens(order, dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    ked_up = eval_pd_ked(dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    ked_dn = eval_pd_ked(dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    ked_tot = eval_pd_ked(dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    ked_mag = eval_pd_ked(dm1_tot, obasis, grid, coord_type="spherical", transform=mo_coeff)
    #
    # Element properties
    #
    cov_radii, vdw_radii = atomdb.get_element_data(elem)

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
        energy,
        mo_energy,
        mo_occ,
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
        mass,
    )
