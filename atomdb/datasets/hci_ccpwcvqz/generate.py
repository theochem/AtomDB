#epsilon This file is part of AtomDB.
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


from os import makedirs

import numpy as np

from basis_set_exchange import get_basis

from pyscf import gto, scf
from pyscf.tools import fcidump

import pyci

from atomdb.config import *
from atomdb.utils import *


__all__ = [
    "generate_species",
]


BASIS_NAME = 'cc-pwcvqz'


EPSILON = 1.0e-4


CONV_TOL = 1.0e-12


NTHREAD=1


def generate_species(element, charge, mult, nexc=0, dataset=DEFAULT_DATASET):
    r"""Run an HCI computation."""
    #
    # Set up internal variables
    #
    elem = get_element_symbol(element)
    natom = get_element_number(elem)
    nelec = natom - charge
    nspin = mult - 1
    n_up = (nelec + nspin) // 2
    n_dn = (nelec - nspin) // 2
    basis = get_basis(BASIS_NAME, elements=natom, fmt='nwchem', header=False)
    title = f"{elem}_N{nelec}_S{nspin}_E{nexc}"
    #
    # Build PySCF molecule
    #
    mol = gto.Mole()
    mol.atom = f"""
    {elem} 0.0  0.0 0.0
    """
    mol.charge = natom - nelec
    mol.spin = nspin
    mol.basis = gto.basis.parse(basis)
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
    rawpath = get_file(f"{dataset}/raw_data/")
    makedirs(rawpath, exist_ok=True)
    with open(f"{rawpath}/{title}_basis_name.txt", "w") as f:
        print(BASIS_NAME, file=f)
    with open(f"{rawpath}/{title}_basis.nwchem", "w") as f:
        print(basis, file=f)
    with open(f"{rawpath}/{title}_hf_energy.txt", "w") as f:
        print(f"H-F energy: {hf.e_tot:24.16e}", file=f)
    np.save(f"{rawpath}/{title}_hf_mo_energies.npy", hf.mo_energy)
    np.save(f"{rawpath}/{title}_hf_mo_occ.npy", hf.mo_occ)
    np.save(f"{rawpath}/{title}_hf_mo_coeff.npy", hf.mo_coeff)
    fcidump.from_scf(hf, f"{rawpath}/{title}_hf.fcidump", tol=1e-16)
    #
    # Run HCI
    #
    pyci.set_num_threads(NTHREAD)
    ham = pyci.hamiltonian(f"{rawpath}/{title}_hf.fcidump")
    fwfn = pyci.fullci_wfn(ham.nbasis, n_up, n_dn)
    pyci.add_excitations(fwfn, 0, 1)
    evals, evecs = pyci.solve(ham, fwfn, n=1, tol=CONV_TOL)
    dets_added = len(fwfn)
    while dets_added:
        dets_added = pyci.add_hci(ham, fwfn, evecs[0], eps=EPSILON)
        evals, evecs = pyci.solve(ham, fwfn, n=1, tol=CONV_TOL)
    #
    # Compute RDMs
    #
    rdm1, rdm2 = pyci.compute_rdms(fwfn, evecs[0])
    rdm1.shape, rdm2.shape
    #
    # Save HCI data
    #
    fwfn.to_file(f"{rawpath}/{title}_hcisd.ci")
    np.save(f"{rawpath}/{title}_hcisd_energies.npy", evals)
    np.save(f"{rawpath}/{title}_hcisd_coeffs.npy", evecs)
    np.save(f"{rawpath}/{title}_hcisd_rdm1.npy", rdm1)
    np.save(f"{rawpath}/{title}_hcisd_rdm2.npy", rdm1)
