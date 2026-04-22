import glob
import os
import uuid
import tables as pt
import tempfile
import atexit
from importlib_resources import files
import warnings

__all__ = [
    "DATASETS_NAMES",
    "create_global_db",
    "GlobalDB",
]
# Suppresses NaturalNameWarning warnings from PyTables.
warnings.filterwarnings("ignore", category=pt.NaturalNameWarning)


DATASETS_NAMES = {
    "slater",
    "gaussian",
    "hci",
    "nist",
    "numeric",
    "uhf_augccpvdz",
}


class GlobalDB:
    """A global database that uses external links to connect to individual dataset files"""

    def __init__(self):
        """Initialize the GlobalDB instance with an in-memory HDF5 file."""

        # Creates a temporary in-memory HDF5 file
        self._tempdir = tempfile.mkdtemp()

        file_path = f"{os.path.join(self._tempdir, uuid.uuid4().hex)}.h5"
        self.h5file = pt.open_file(
            file_path, mode="a", driver="H5FD_CORE", driver_core_backing_store=0
        )
        # The file will be automatically closed when the program exits.
        atexit.register(self.close)

    def __del__(self):
        """Destructor that ensures proper closing of the hdf5 file."""
        self.close()

    def close(self):
        """Close the HDF5 file."""
        self.h5file.close()

    def attach_dataset(self, dataset, version=None):
        """Attach a dataset to the global database using an external link.

        Parameters
        ----------
        dataset : str
            Name of the dataset.
        version : int, optional
        """
        dataset_file = self._get_last_h5file_version(dataset, version)

        # Verify the target file exists before creating the link

        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file {dataset_file} not found")
        self.h5file.create_external_link("/", dataset, f"{dataset_file}:/")

    def _get_last_h5file_version(self, dataset, version):
        """Find the appropriate dataset file for the requested version.

        Parameters
        ----------
        dataset : str
            Name of the dataset.
        version : int, optional

        Returns
        -------
        str
            Path to the dataset file.
        """
        dataset_dir = files("atomdb.datasets.datasets_files")

        if version is not None:
            dataset_file = str(dataset_dir.joinpath(f"{dataset}_v{version:03d}.h5"))
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Requested version v{version:03d} not found")

        else:
            dataset_files_paths = str(dataset_dir.joinpath(f"{dataset}_v*.h5"))
            # Find all versioned files
            hdf5_files = glob.glob(dataset_files_paths)

            # Return last version
            hdf5_files.sort()
            dataset_file = hdf5_files[-1]

        return dataset_file


def create_global_db(version=None):
    """Create and initialize a global database with all datasets.

    Returns
    -------
    global_db
        An initialized GlobalDB instance with external links to all datasets.
    """
    global_db = GlobalDB()

    for dataset in DATASETS_NAMES:
        try:
            global_db.attach_dataset(dataset, version)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: could not attach {dataset} dataset: {e}")

    return global_db
