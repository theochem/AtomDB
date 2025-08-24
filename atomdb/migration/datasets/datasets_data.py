"""
running this file will recreate datasets_data.h5, that will lead to create an empty datasets folder once again
"""

from importlib_resources import files
import tables as pt

hdf5_file = files("atomdb.datasets").joinpath("datasets_data.h5")

with pt.open_file(hdf5_file, mode="w", title="Datasets Data Files") as h5file:
    # create the root folder 'datasets'
    datasets_folder = h5file.create_group("/", "Datasets", "Datasets Data")

    # create a folder for each dataset to hold its data files
    h5file.create_group(datasets_folder, "slater", "Slater dataset")
    h5file.create_group(datasets_folder, "gaussian", "Gaussian dataset")
    h5file.create_group(datasets_folder, "hci", "HCI dataset")
    h5file.create_group(datasets_folder, "nist", "NIST dataset")
    h5file.create_group(datasets_folder, "numeric", "Numeric dataset")
    h5file.create_group(datasets_folder, "uhf_augccpvdz", "UHF aug-cc-pVDZ dataset")
