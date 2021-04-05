from dataclasses import dataclass

import numpy as np

from gbasis.parsers import parse_nwchem, make_contractions
from gbasis.evals.density import evaluate_density
from gbasis.evals.density import evaluate_deriv_density
from gbasis.evals.density import evaluate_posdef_kinetic_energy_density

from .utils import cubic_interpolation, get_datafile, get_raw_datafile


__all__ = [
    "compile_dbentry",
    "DBEntry",
]


@dataclass
class DBEntry:
    r"""Database entry class."""
    #
    # Species info
    #
    species: str
    basis: str
    nelec: int
    nspin: int
    #
    # Electronic and molecular orbital energies
    #
    energy_ci: float
    mo_energies: np.array
    mo_occs: np.array
    #
    # Radial grid
    #
    rs: np.ndarray
    #
    # Density
    #
    dens_up: np.ndarray
    dens_dn: np.ndarray
    dens_tot: np.ndarray
    dens_mag: np.ndarray
    #
    # Derivative of density
    #
    d_dens_up: np.ndarray
    d_dens_dn: np.ndarray
    d_dens_tot: np.ndarray
    d_dens_mag: np.ndarray
    #
    # Laplacian
    #
    lapl_up: np.ndarray
    lapl_dn: np.ndarray
    lapl_tot: np.ndarray
    lapl_mag: np.ndarray
    #
    # Kinetic energy density
    #
    ked_up: np.ndarray
    ked_dn: np.ndarray
    ked_tot: np.ndarray
    ked_mag: np.ndarray


def compile_dbentry(dataset, species, nelec, nspin, basis_name, bound=(0.01, 0.5), num=100):
    r"""Initialize a DBEntry instance."""
    #
    # Load raw data from computation
    #
    mo_coeff_file = get_raw_datafile("gbs_hf_mo_coeff.npy", dataset, species, basis_name, nelec, nspin)
    mo_coeff = np.load(mo_coeff_file).transpose()
    dm1_file = get_raw_datafile("gbs_hcisd_rdm1.npy", dataset, species, basis_name, nelec, nspin)
    dm1_up, dm1_dn = np.load(dm1_file)
    dm1_tot = dm1_up + dm1_dn
    dm1_mag = dm1_up - dm1_dn
    basis = parse_nwchem(get_datafile(f"raw/{dataset}/{basis_name}.nwchem"))
    basis = make_contractions(basis, [species], np.array([[0, 0, 0]]))
    eci_file = get_raw_datafile("gbs_hcisd_energies.npy", dataset, species, basis_name, nelec, nspin)
    energy_ci = np.load(eci_file)
    # energy_hf = get_raw_datafile("gbs_hf.txt", dataset, species, basis_name, nelec, nspin)
    mos_file = get_raw_datafile("gbs_hf_mo_energies.npy", dataset, species, basis_name, nelec, nspin)
    mo_energies = np.load(mos_file)
    mo_occs_file = get_raw_datafile("gbs_hf_mo_occ.npy", dataset, species, basis_name, nelec, nspin)
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
    # Return DBEntry instance
    #
    return DBEntry(
        species, basis_name, nelec, nspin,
        energy_ci, mo_energies, mo_occs,
        rs, dens_up, dens_dn, dens_tot, dens_mag,
        d_dens_up, d_dens_dn, d_dens_tot, d_dens_mag,
        lapl_up, lapl_dn, lapl_tot, lapl_mag,
        ked_up, ked_dn, ked_tot, ked_mag,
    )


if __name__ == "__main__":
    from .utils import get_element
    NATOM = 5
    NELEC = 5
    NSPIN = 1
    DSET= "hci_1.0e-4"
    BASIS = "cc-pwcvqz"
    SPECIES = get_element(NATOM)
    RMIN = 0.0
    RMAX = 0.2
    N_POINTS = 500
    atprop = compile_dbentry(DSET, SPECIES, NELEC, NSPIN, BASIS, bound=(RMIN, RMAX), num=N_POINTS)
    print("DONE")