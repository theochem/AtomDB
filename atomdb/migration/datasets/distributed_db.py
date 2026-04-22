from importlib_resources import files
from atomdb.species import get_versioned_h5file
from atomdb.db import DATASETS_NAMES
import tables as pt
import warnings
import atexit

# Suppresses NaturalNameWarning warnings from PyTables.
warnings.filterwarnings("ignore", category=pt.NaturalNameWarning)

datasets_hdf5_file = get_versioned_h5file()
DATASETS_H5FILE = pt.open_file(datasets_hdf5_file, mode="a")
atexit.register(DATASETS_H5FILE.close)


for dataset_name in DATASETS_NAMES:
    dataset_path = f"/Datasets/{dataset_name}"
    dataset_file = files("atomdb.datasets.datasets_files").joinpath(f"{dataset_name}_v000.h5")

    with pt.open_file(dataset_file, "w") as h5file:
        DATASETS_H5FILE.copy_node(dataset_path, h5file.root, recursive=True)
        print(f"Created {dataset_file}")
