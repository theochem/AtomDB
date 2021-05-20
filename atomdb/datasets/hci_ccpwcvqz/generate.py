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

r"""HCI generate function."""


import sys

import numpy as np

from pyscf import gto, scf
from pyscf.tools import fcidump

import pyci

from atomdb.utils import get_element, get_element_name


__all__ = [
    "generate_species",
]


# TODO: make arguments consistent.
def generate_species(species, nelec, nspin, nexc, epsilon=1.0e-4, conv_tol=1.0e-12, nthread=1):
    r"""Run an HCI computation."""
    basis_name = 'cc-pwcvqz' # TODO: get rid of this argument
    species = get_element(species)
    natom = get_element_number(species)
    n_up = (nelec + nspin) // 2
    n_dn = (nelec - nspin) // 2
    title = f"{species}_N{nelec}_S{nspin}"
    #
    # Build PySCF molecule
    #
    mol = gto.Mole()
    mol.atom = f"""
    {species} 0.0  0.0 0.0
    """
    mol.charge = natom - nelec
    mol.spin = nspin
    mol.basis = basis_name
    mol.build()
    #
    # Run restricted Hartree-Fock SCF
    #
    hf = scf.RHF(mol)
    hf.conv_tol = 1.0e-12
    hf.run(verbose=4)
    #
    # Save Hartree-Fock data
    #
    np.save(f"{title}_{basis_name}_hf_mo_energies.npy", hf.mo_energy)
    np.save(f"{title}_{basis_name}_hf_mo_occ.npy", hf.mo_occ)
    np.save(f"{title}_{basis_name}_hf_mo_coeff.npy", hf.mo_coeff)
    fcidump.from_scf(hf, f"{title}_{basis_name}_hf.fcidump", tol=1e-16)
    with open(f"{title}_{basis_name}_hf.txt", "w") as f:
        print(f"H-F energy: {hf.e_tot:24.16e}", file=f)
    #
    # Run HCI
    #
    pyci.set_num_threads(nthread)
    ham = pyci.hamiltonian(f"{title}_{basis_name}_hf.fcidump")
    fwfn = pyci.fullci_wfn(ham.nbasis, n_up, n_dn)
    pyci.add_excitations(fwfn, 0, 1)
    evals, evecs = pyci.solve(ham, fwfn, n=1, tol=conv_tol)
    dets_added = len(fwfn)
    while dets_added:
        dets_added = pyci.add_hci(ham, fwfn, evecs[0], eps=epsilon)
        evals, evecs = pyci.solve(ham, fwfn, n=1, tol=conv_tol)
    #
    # Compute RDMs
    #
    rdm1, rdm2 = pyci.compute_rdms(fwfn, evecs[0])
    rdm1.shape, rdm2.shape
    #
    # Save HCI data
    #
    fwfn.to_file(f"{title}_{basis_name}_hcisd.ci")
    np.save(f"{title}_{basis_name}_hcisd_energies.npy", evals)
    np.save(f"{title}_{basis_name}_hcisd_coeffs.npy", evecs)
    np.save(f"{title}_{basis_name}_hcisd_rdm1.npy", rdm1)
    np.save(f"{title}_{basis_name}_hcisd_rdm2.npy", rdm1)
