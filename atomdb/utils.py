import csv
import os
import numpy as np
import h5py as h5
from scipy import constants

# Define physical constants
ANGSTROM = 100 * constants.pico * constants.m_e * constants.c * constants.alpha / constants.hbar
AMU = constants.gram / (constants.Avogadro * constants.m_e)
CMINV = 2 * constants.centi * constants.Rydberg
EV = constants.eV / (2 * constants.Rydberg * constants.h * constants.c)

# File paths for database and CSV storage
MODULE_DATAPATH = os.path.join(os.path.dirname(__file__), "data")
HDF5_NIST_FILE = os.path.join(MODULE_DATAPATH, "database_beta_1.3.0.h5")
MULT_TABLE_CSV = os.path.join(MODULE_DATAPATH, "multiplicities_table.csv")

def load_multiplicities(max_atomic_number=100):
    """
    Reads multiplicities from a CSV file and returns a dictionary.

    Parameters:
        max_atomic_number (int): Maximum atomic number to be processed.

    Returns:
        dict: Multiplicities indexed by (atomic number, charge).
    """
    multiplicities = {}
    
    try:
        with open(MULT_TABLE_CSV, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            charges = list(map(int, next(reader)[1:]))  # Read charge values
            
            for row in reader:
                atomic_number = int(row[0])
                values = list(map(int, row[1:]))
                for charge, multiplicity in zip(charges, values):
                    if multiplicity > 0:
                        multiplicities[(atomic_number, charge)] = multiplicity
    except FileNotFoundError:
        print(f"Warning: {MULT_TABLE_CSV} not found. Multiplicities not loaded.")
    
    return multiplicities

def generate_multiplicity_csv(max_atomic_number=100):
    """
    Generates and writes a multiplicity table to a CSV file.

    The data is read from the HDF5 file, and missing values are set to zero.

    Parameters:
        max_atomic_number (int): Highest atomic number to include in the table.

    Raises:
        ValueError: If `max_atomic_number` exceeds 100.
    """
    if max_atomic_number > 100:
        raise ValueError("Max atomic number must be ≤ 100.")

    min_charge, max_charge = -2, max_atomic_number

    # Initialize numpy array for table storage
    table = np.zeros((max_atomic_number, 1 + max_charge - min_charge), dtype=int)
    table[:, 0] = np.arange(1, max_atomic_number + 1)

    if not os.path.exists(HDF5_NIST_FILE):
        raise FileNotFoundError(f"Missing data file: {HDF5_NIST_FILE}")

    with h5.File(HDF5_NIST_FILE, "r") as h5_file:
        for atomic_number, row in enumerate(table[:, 1:], start=1):
            for col, charge in enumerate(range(min_charge, max_charge)):
                num_electrons = atomic_number - charge
                if num_electrons <= 0:
                    continue

                try:
                    element_group = h5_file[f"{atomic_number:03d}"][f"{num_electrons:03d}"]
                    multiplicities = element_group["Multi"][...].astype(int)
                    energies = element_group["Energy"][...].astype(float)

                    if multiplicities.size and energies.size:
                        row[col] = multiplicities[np.argmin(energies)]  # Select most stable species
                except KeyError:
                    # Missing data; default to zero
                    row[col] = 0

    # Write to CSV file
    with open(MULT_TABLE_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["", "charge"] + [""] * (max_charge - min_charge - 1))
        writer.writerow(["atnum"] + list(map(str, range(min_charge, max_charge))))
        writer.writerows(table)

# Load multiplicities into a dictionary
MULTIPLICITIES = load_multiplicities()
