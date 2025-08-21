import warnings
import numpy as np
from importlib_resources import files
import tables as pt
from dataclasses import asdict
from atomdb.datasets.gaussian.run import NPOINTS
from atomdb.periodic_test import element_symbol_map, ElementAttr


# Suppresses NaturalNameWarning warnings from PyTables.
warnings.filterwarnings("ignore", category=pt.NaturalNameWarning)

max_norba = 41

GAUSSIAN_PROPERTY_CONFIGS = [
    {
        "SpeciesInfo": "elem",
        "type": "string",
    },
    {
        "SpeciesInfo": "nexc",
        "type": "int",
    },
    {
        "SpeciesInfo": "charge",
        "type": "int",
    },
    {
        "SpeciesInfo": "mult",
        "type": "int",
    },
    {
        "SpeciesInfo": "nelec",
        "type": "int",
    },
    {
        "SpeciesInfo": "nspin",
        "type": "int",
    },
    {
        "SpeciesInfo": "energy",
        "type": "float",
    },
    {
        "SpeciesInfo": "ip",
        "type": "float",
    },
    {
        "SpeciesInfo": "mu",
        "type": "float",
    },
    {
        "SpeciesInfo": "eta",
        "type": "float",
    },
    {
        "SpeciesInfo": "nbasis",
        "type": "int",
    },
    {
        "property": "obasis_name",
        "table_name": "obasis_name",
        "description": "Orbital basis name",
        "type": "string",
    },
    {
        "array_property": "mo_energy_a",
        "table_name": "mo_energy_a",
        "description": "Alpha MO Energies",
    },
    {
        "array_property": "mo_energy_b",
        "table_name": "mo_energy_b",
        "description": "Beta MO Energies",
    },
    {
        "array_property": "mo_occs_a",
        "table_name": "mo_occs_a",
        "description": "Alpha MO Occupations",
    },
    {
        "array_property": "mo_occs_b",
        "table_name": "mo_occs_b",
        "description": "Alpha MO Energies",
    },
    {"Carray_property": "rs", "table_name": "rs", "folder": "RadialGrid", "spins": "no"},
    {
        "Carray_property": "mo_dens_a",
        "table_name": "mo_dens_a",
        "folder": "Density",
        "spins": "yes",
    },
    {
        "Carray_property": "mo_dens_b",
        "table_name": "mo_dens_b",
        "folder": "Density",
        "spins": "yes",
    },
    {"Carray_property": "dens_tot", "table_name": "dens_tot", "folder": "Density", "spins": "no"},
    {
        "Carray_property": "mo_d_dens_a",
        "table_name": "mo_d_dens_a",
        "folder": "DensityGradient",
        "spins": "yes",
    },
    {
        "Carray_property": "mo_d_dens_b",
        "table_name": "mo_d_dens_b",
        "folder": "DensityGradient",
        "spins": "yes",
    },
    {
        "Carray_property": "d_dens_tot",
        "table_name": "d_dens_tot",
        "folder": "DensityGradient",
        "spins": "no",
    },
    {
        "Carray_property": "mo_dd_dens_a",
        "table_name": "mo_dd_dens_a",
        "folder": "DensityLaplacian",
        "spins": "yes",
    },
    {
        "Carray_property": "mo_dd_dens_b",
        "table_name": "mo_dd_dens_b",
        "folder": "DensityLaplacian",
        "spins": "yes",
    },
    {
        "Carray_property": "dd_dens_tot",
        "table_name": "dd_dens_tot",
        "folder": "DensityLaplacian",
        "spins": "no",
    },
    {
        "Carray_property": "mo_ked_a",
        "table_name": "mo_ked_a",
        "folder": "KineticEnergyDensity",
        "spins": "yes",
    },
    {
        "Carray_property": "mo_ked_b",
        "table_name": "mo_ked_b",
        "folder": "KineticEnergyDensity",
        "spins": "yes",
    },
    {
        "Carray_property": "ked_tot",
        "table_name": "ked_tot",
        "folder": "KineticEnergyDensity",
        "spins": "no",
    },
]


class IntPropertyDescription(pt.IsDescription):
    value = pt.Int32Col()


class StringPropertyDescription(pt.IsDescription):
    value = pt.StringCol(25)


class FloatPropertyDescription(pt.IsDescription):
    value = pt.Float64Col()


# static definition
class ArrayPropertyDescription(pt.IsDescription):
    value = pt.Float64Col(shape=(max_norba,))


class SpeciesInfo(pt.IsDescription):
    """Schema for SpeciesInfo table."""

    elem = pt.StringCol(25)
    charge = pt.Int32Col()
    mult = pt.Int32Col()
    nexc = pt.Int32Col()
    nelec = pt.Int32Col()
    nspin = pt.Int32Col()
    nbasis = pt.Int32Col()
    energy = pt.Float64Col()
    ip = pt.Float64Col()
    mu = pt.Float64Col()
    eta = pt.Float64Col()


def create_species_info_table(species_info_table_row, prop_name, prop_type, value):
    """Adds a property column to speciesInfo table.

    Args:
        table_row (dict): single row in the table that holds all the columns.
        prop_name (str): Name of the property column to add to the table.
        prop_type (str): Data type of the property ('int', 'string', or 'float').
        value: The value to store in the column.

    """
    if prop_type == "int":
        value = int(value) if value is not None else 0

    elif prop_type == "string":
        value = str(value) if value is not None else ""

    elif prop_type == "float":
        value = float(value) if value is not None else np.nan

    species_info_table_row[prop_name] = value


def create_properties_tables(hdf5_file, parent_folder, config, value):
    """Creates a table for storing properties in the HDF5 file.

    Args:
        hdf5_file (tables.File): The open HDF5 file where the table will be created.
        parent_folder (tables.Group): The parent folder in the HDF5 file where the table will be stored.
        config (dict): Configuration dictionary containing table metadata, including:
            - 'table_name': Name of the table.
            - 'description': Description of the table.
            - 'type': Data type of the property ('int', 'string', or 'float').
        value: The value to store in the table.
    """

    # Extract table metadata from config.
    table_name = config["table_name"]
    table_description = config["description"]
    type = config["type"]

    if type == "int":
        row_description = IntPropertyDescription
        value = int(value) if value is not None else 0

    elif type == "string":
        row_description = StringPropertyDescription
        value = str(value) if value is not None else ""

    elif type == "float":
        row_description = FloatPropertyDescription
        value = float(value) if value is not None else np.nan

    # Create the table and populate the data
    table = hdf5_file.create_table(parent_folder, table_name, row_description, table_description)
    row = table.row
    row["value"] = value
    row.append()
    table.flush()


def create_properties_arrays(hdf5_file, parent_folder, table_name, description, data):
    """Creates a table for storing an array property in the HDF5 file.

    Args:
        hdf5_file (tables.File): The open HDF5 file where the array will be created.
        parent_folder (tables.Group): The parent folder in the HDF5 file where the table will be stored.
        table_name (str): Name of the table to create.
        description (str): Description of the table.
        data (numpy.ndarray): The array data to store in the table.
    """
    filters = pt.Filters(complevel=5, complib="blosc2:lz4")

    # Create the table and populate the data
    table = hdf5_file.create_table(
        parent_folder, table_name, ArrayPropertyDescription, description, filters=filters
    )
    row = table.row
    padded_data = np.pad(data, (0, max_norba - len(data)), "constant", constant_values=0)
    row["value"] = padded_data
    row.append()
    table.flush()


def create_spins_array(h5file, parent_folder, key, array_data, shape):
    """Creates a  CArray for storing spin-dependent array data in the HDF5 file.

    Args:
        hdf5_file (tables.File): The open HDF5 file where the CArray will be created.
    parent_folder (tables.Group): The parent folder in the HDF5 file where the CArray will be stored.
        key (str): Name of the CArray.
        array_data (numpy.ndarray): The array data to store in the CArray.
        shape (int): The total size of the CArray.
    """
    data_length = len(array_data)
    filters = pt.Filters(complevel=5, complib="blosc2:lz4")

    # Create the CArray and populate the data
    array = h5file.create_carray(
        parent_folder, key, pt.Float64Atom(), shape=(shape,), filters=filters
    )
    array[:data_length] = array_data
    array[data_length:] = 0


def create_tot_array(h5file, parent_folder, key, array_data):
    """Creates a CArray for storing total (non-spin-dependent) array data in the HDF5 file.

    Args:
        h5file (tables.File): The open HDF5 file where the CArray will be created.
        parent_folder (tables.Group): The parent folder in the HDF5 file where the CArray will be stored.
        key (str): Name of the CArray.
        array_data (numpy.ndarray): The array data to store in the CArray.
    """
    data_length = len(array_data)
    filters = pt.Filters(complevel=5, complib="blosc2:lz4")

    # Create the CArray and populate the data
    tot_gradient_array = h5file.create_carray(
        parent_folder, key, pt.Float64Atom(), shape=(NPOINTS,), filters=filters
    )
    if data_length < NPOINTS:
        tot_gradient_array[:data_length] = array_data
        tot_gradient_array[data_length:] = 0

    else:
        tot_gradient_array[:] = array_data


def create_hdf5_file(DATASETS_H5FILE, fields, dataset, mult):
    """Creates an HDF5 folder with structured data for a specific dataset and element.

    Args:
        DATASETS_H5FILE (tables.File): An open PyTables HDF5 file object to store the data.
        fields (dataclass): A dataclass containing the fields to store in the HDF5 file.
        dataset (str): Name of the dataset.
        mult (int): Multiplicity.
    """
    fields = asdict(fields)
    dataset = dataset.lower()
    shape = NPOINTS * max_norba

    elem = fields["elem"]
    nexc = fields["nexc"]
    atnum = element_symbol_map[elem][ElementAttr.atnum]
    charge = atnum - fields["nelec"]

    # charge and mult can be calculated (instead of passing them)?
    dataset_folder = f"/Datasets/{dataset}"
    elem_folder = f"{dataset_folder}/{elem}"
    specific_elem_folder = f"{elem_folder}/{elem}_{charge:03d}_{mult:03d}_{nexc:03d}"

    # Create dataset folder if it doesn't exist
    if dataset_folder not in DATASETS_H5FILE:
        DATASETS_H5FILE.create_group("/Datasets", dataset, f"{dataset} Data")

    # Create element folder if it doesn't exist
    if elem_folder not in DATASETS_H5FILE:
        DATASETS_H5FILE.create_group(dataset_folder, elem, f"{elem} Data")

    # Create specific element folder (charge/mult/nexc) if it doesn't exist
    if specific_elem_folder not in DATASETS_H5FILE:
        DATASETS_H5FILE.create_group(
            elem_folder,
            f"{elem}_{charge:03d}_{mult:03d}_{nexc:03d}",
            f"{elem} {charge} {mult} {nexc} Data",
        )

    folders = {
        "Properties": DATASETS_H5FILE.create_group(
            specific_elem_folder, "Properties", "Properties Data"
        ),
        "RadialGrid": DATASETS_H5FILE.create_group(
            specific_elem_folder, "RadialGrid", "Radial Grid Data"
        ),
        "Density": DATASETS_H5FILE.create_group(specific_elem_folder, "Density", "Density Data"),
        "DensityGradient": DATASETS_H5FILE.create_group(
            specific_elem_folder, "DensityGradient", "Density Gradient Data"
        ),
        "DensityLaplacian": DATASETS_H5FILE.create_group(
            specific_elem_folder, "DensityLaplacian", "Density Laplacian Data"
        ),
        "KineticEnergyDensity": DATASETS_H5FILE.create_group(
            specific_elem_folder, "KineticEnergyDensity", "Kinetic Energy Density Data"
        ),
    }

    # Create basic species table and its row
    species_info_table = DATASETS_H5FILE.create_table(
        folders["Properties"], "species_info", SpeciesInfo, "Species Information"
    )
    species_info_table_row = species_info_table.row

    # Create basic property tables
    for config in GAUSSIAN_PROPERTY_CONFIGS:
        if "SpeciesInfo" in config:
            prop_name = config["SpeciesInfo"]
            create_species_info_table(
                species_info_table_row, prop_name, config["type"], fields[prop_name]
            )

        elif "property" in config:
            prop_name = config["property"]
            create_properties_tables(
                DATASETS_H5FILE, folders["Properties"], config, fields[prop_name]
            )

        # Create array property tables
        elif "array_property" in config:
            prop_name = config["array_property"]
            create_properties_arrays(
                DATASETS_H5FILE,
                folders["Properties"],
                config["table_name"],
                config["description"],
                fields[prop_name],
            )

        elif "Carray_property" in config:
            prop_name = config["Carray_property"]
            parent_folder = folders[config["folder"]]
            if config["spins"] == "yes":
                create_spins_array(
                    DATASETS_H5FILE, parent_folder, config["table_name"], fields[prop_name], shape
                )
            elif config["spins"] == "no":
                create_tot_array(
                    DATASETS_H5FILE, parent_folder, config["table_name"], fields[prop_name]
                )

    species_info_table_row.append()
    species_info_table.flush()
