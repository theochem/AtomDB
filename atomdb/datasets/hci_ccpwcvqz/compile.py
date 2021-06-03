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

from gbasis.parsers import parse_nwchem, make_contractions
from gbasis.evals.density import evaluate_density
from gbasis.evals.density import evaluate_deriv_density
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density

from atomdb.utils import get_element_symbol, get_element_number, get_file, get_raw_data_file
from atomdb.api import *

__all__ = [
    "compile_species",
]


# TODO: removed basis_name argument, make arguments consistent.
def compile_species(dataset, species, nelec, nspin, nexc, bound=(0.01, 0.5), num=100):
    r"""Initialize a Species instance from an HCI computation."""
    #
    # Load raw data from computation
    #
    species = get_element_symbol(species)
    natom = get_element_number(species)
    mo_coeff_file = get_raw_data_file(dataset, species, nelec, nspin, nexc, "hf_mo_coeff.npy")
    mo_coeff = np.load(mo_coeff_file).transpose()
    dm1_file = get_raw_data_file(dataset, species, nelec, nspin, nexc, "hcisd_rdm1.npy")
    dm1_up, dm1_dn = np.load(dm1_file)
    dm1_tot = dm1_up + dm1_dn
    dm1_mag = dm1_up - dm1_dn
    basis_name = get_file(f"{dataset}/raw_data/basis.txt")
    with open(basis_name, 'r') as f:
        basis_name = f.readline().strip()
    basis = parse_nwchem(get_file(f"{dataset}/raw_data/basis.nwchem"))
    basis = make_contractions(basis, [species], np.array([[0, 0, 0]]))
    eci_file = get_raw_data_file(dataset, species, nelec, nspin, nexc, "hcisd_energies.npy")
    energy_ci = np.load(eci_file)
    hf_file = get_raw_data_file(dataset, species, nelec, nspin, nexc, "hf.txt")
    with open(hf_file, 'r') as f:
        energy_hf = float(f.readline()[12:])
    mos_file = get_raw_data_file(dataset, species, nelec, nspin, nexc, "hf_mo_energies.npy")
    mo_energies = np.load(mos_file)
    mo_occs_file = get_raw_data_file(dataset, species, nelec, nspin, nexc, "hf_mo_occ.npy")
    mo_occs = np.load(mo_occs_file)
    #
    # Make grid
    #
    rs = np.linspace(*bound, num)
    grid = np.zeros((num, 3))
    grid[:, 0] = rs
    #
    # Compute densities and derivatives
    #
    order = np.array([1, 0, 0])
    dens_up = evaluate_density(dm1_up, basis, grid, coord_type="spherical", transform=mo_coeff)
    dens_dn = evaluate_density(dm1_dn, basis, grid, coord_type="spherical", transform=mo_coeff)
    dens_tot = evaluate_density(dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    dens_mag = evaluate_density(dm1_mag, basis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_up = evaluate_deriv_density(order, dm1_up, basis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_dn = evaluate_deriv_density(order, dm1_dn, basis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_tot = evaluate_deriv_density(order, dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    d_dens_mag = evaluate_deriv_density(order, dm1_mag, basis, grid, coord_type="spherical", transform=mo_coeff)
    #
    # Compute laplacian and kinetic energy density
    #
    order = np.array([2, 0, 0])
    lapl_up = evaluate_deriv_density(order, dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    lapl_dn = evaluate_deriv_density(order, dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    lapl_tot = evaluate_deriv_density(order, dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    lapl_mag = evaluate_deriv_density(order, dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    ked_up = evaluate_posdef_kinetic_energy_density(dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    ked_dn = evaluate_posdef_kinetic_energy_density(dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    ked_tot = evaluate_posdef_kinetic_energy_density(dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    ked_mag = evaluate_posdef_kinetic_energy_density(dm1_tot, basis, grid, coord_type="spherical", transform=mo_coeff)
    #
    # Return Species instance
    #
    return Species(
        species, natom, basis_name, nelec, nspin,
        energy_ci, energy_hf, mo_energies, mo_occs,
        rs, dens_up, dens_dn, dens_tot, dens_mag,
        d_dens_up, d_dens_dn, d_dens_tot, d_dens_mag,
        lapl_up, lapl_dn, lapl_tot, lapl_mag,
        ked_up, ked_dn, ked_tot, ked_mag,
    )


if __name__ == "__main__":
    from atomdb.utils import get_element
    NATOM = 5
    NELEC = 5
    NSPIN = 1
    DSET= "hci_ccpwcvqz"
    NEXC = 0
    SPECIES = get_element(NATOM)
    RMIN = 0.0
    RMAX = 0.2
    N_POINTS = 500
    atprop = compile_species(DSET, SPECIES, NELEC, NSPIN, NEXC, bound=(RMIN, RMAX), num=N_POINTS)
    print("atprop.to_dict():")
    print("----------------")
    print(atprop.to_dict())
    print("----------------")
    print("DONE")
